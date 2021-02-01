import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch_geometric.nn import GATConv
from torch_scatter import scatter_add
from torch_geometric.utils import softmax


class ConvLayer(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, model='Gen'):
        super(ConvLayer, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, (n_outputs * 2), kernel_size, stride=stride,
                                          padding=padding,dilation=dilation))
        self.padding = padding
        self.model = model
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.trans = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if self.trans is not None:
            torch.nn.init.xavier_uniform_(self.trans.weight)

    def forward(self, x):
        y = self.conv(x)
        out = self.glu(y[:, :, :-self.padding].contiguous()) if self.model == 'Gen' else self.glu(y)
        out = self.dropout(out)
        if self.trans is not None:
            x = self.trans(x)
        return out + x


class Encoder(nn.Module):

    def __init__(self, input_size, hid_size, n_levels, kernel_size=3, dropout=0.2, model='Gen'):
        super(Encoder, self).__init__()
        layers = []
        for i in range(n_levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size if model == 'Gen' else dilation_size
            if i == 0:
                layers += [ConvLayer(input_size, hid_size, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout, model=model)]
            else:
                layers += [ConvLayer(hid_size, hid_size, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout, model=model)]
        
        self.network = (nn.Sequential)(*layers)

    def forward(self, x):
        return self.network(x)


class NNet(nn.Module):

    def __init__(self, n_in, n_out, hide=(64, 64, 8)):
        super(NNet, self).__init__()
        self.n_hide = len(hide)
        self.fcs = nn.ModuleList([weight_norm(nn.Linear(n_in, hide[i] * 2)) if i == 0 else 
                                  weight_norm(nn.Linear(hide[(i - 1)], n_out)) if i == self.n_hide else 
                                  weight_norm(nn.Linear(hide[(i - 1)], hide[i] * 2)) for i in range(self.n_hide + 1)])
        self.init_weights()

    def init_weights(self):
        for i in range(self.n_hide + 1):
            self.fcs[i].weight.data.normal_(0, 0.01)

    def forward(self, x):
        for i in range(self.n_hide):
            x = F.glu(self.fcs[i](x))
        x = self.fcs[(-1)](x)
        return x


class GEN(nn.Module):

    def __init__(self, n_props, dic_size, emb_size, hid_size=256, n_levels=5, kernel_size=3, dropout=0.2):
        super(GEN, self).__init__()
        self.emb = nn.Embedding(dic_size, emb_size, padding_idx=0)
        self.dense = NNet(n_props, 16, hide=(8, 64))
        self.encoder = Encoder((emb_size + 16), hid_size, n_levels, kernel_size, dropout=dropout, model='Gen')
        self.decoder = nn.Linear(hid_size, dic_size)

    def forward(self, input, label):
        emb = self.emb(input)
        pro = self.dense(label).unsqueeze(-1).expand(-1, -1, emb.size(1))
        y = self.encoder(torch.cat((emb.transpose(1, 2), pro), dim=1))
        o = self.decoder(y.transpose(1, 2))
        return o.contiguous()


class GlobalAttention(torch.nn.Module):

    def __init__(self, gate_nn, nn=None):
        super(GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.gate, self.x = (None, None)
        self.reset_parameters()

    def reset_parameters(self):
        self.reset(self.gate_nn)
        self.reset(self.nn)

    def reset(self, nn):

        def _reset(item):
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

        if nn is not None:
            if hasattr(nn, 'children') and len(list(nn.children())) > 0:
                for item in nn.children():
                    _reset(item)

            else:
                _reset(nn)

    def forward(self, x, batch, out_size, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[(-1)].item() + 1 if size is None else size
        gate = self.gate_nn(x).view(-1, out_size)
        x = self.nn(x) if self.nn is not None else x
        if not (gate.dim() == x.dim() and gate.size(0) == x.size(0)):
            raise AssertionError
        gate = softmax(gate, batch, num_nodes=size)
        self.gate, self.x = gate, x
        out = scatter_add((gate * x), batch, dim=0, dim_size=size)
        return out

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__, self.gate_nn, self.nn)


class PRE(torch.nn.Module):

    def __init__(self, h_size, emb_h, dim, n_levels, dropout, out_size):
        super(PRE, self).__init__()
        self.emb_h = nn.Embedding(h_size, emb_h)
        self.n_levles = n_levels
        self.convs = nn.ModuleList([GATConv(emb_h, dim, heads=8, dropout=dropout) if i == 0 else 
                                    GATConv(dim * 8, dim, heads=1, concat=False, dropout=dropout) if i == self.n_levles else 
                                    GATConv(dim * 8, dim * 2, heads=4, dropout=dropout) for i in range(self.n_levles + 1)])
        self.global_atten = GlobalAttention(nn.Linear(dim, out_size), nn.Linear(dim, out_size))
        self.out_size=out_size

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.emb_h(x.squeeze())
        for i in range(self.n_levles+1):
            x = F.relu(self.convs[i](x, edge_index))
        x = self.global_atten(x, batch, self.out_size)
        return x