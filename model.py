import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from dgllife.model.gnn.gat import GAT
from dgl.readout import softmax_nodes,sum_nodes


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


class GlobalAttentionPooling(nn.Module):
    
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn
        self.gate, self.feat = (None, None)

    def forward(self, graph, feat):
        
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata['gate'] = gate
            gate = softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')

            self.gate, self.feat = gate, feat 

            graph.ndata['r'] = feat * gate
            readout = sum_nodes(graph, 'r')
            graph.ndata.pop('r')

            return readout


class PRE(torch.nn.Module):

    def __init__(self, h_size, emb_h, dim):
        super(PRE, self).__init__()
        self.emb_h = nn.Embedding(h_size, emb_h)
        self.convs = GAT(emb_h,hidden_feats=[dim,dim*2,dim*2,dim*2,dim*2,dim],num_heads=[8,4,4,4,4,1],
                         agg_modes=['flatten','flatten','flatten','flatten','flatten','mean'])
        self.global_atten0 = GlobalAttentionPooling(nn.Linear(dim, 1), nn.Linear(dim, 1))
        self.global_atten1 = GlobalAttentionPooling(nn.Linear(dim, 1), nn.Linear(dim, 1))

    def forward(self, bg, feats):
        feats = self.emb_h(feats)
        feats = self.convs(bg,feats)
        x0 = self.global_atten0(bg,feats)
        x1 = self.global_atten1(bg,feats)
        return torch.cat((x0,x1),dim=-1)