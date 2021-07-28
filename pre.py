import argparse
import config
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import PRE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import dgl


def train(train_iter):
    model.train()
    total_loss = 0
    for data in train_iter:
        smiles, bg, labels, masks = data
        bg, labels, masks = bg.to(device), labels.to(device), masks.to(device)
        node_feats = bg.ndata.pop('atomic')
        # edge_feats = bg.edata.pop('e').to(device)
        
        optimizer.zero_grad()
        outputs = model(bg, node_feats.to(device))
        loss = (F.mse_loss(outputs,labels[:,args.property_n])* (masks != 0).float()).mean()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()  

    return total_loss/len(train_iter)


def evaluate(data_iter):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for data in data_iter:
            smiles, bg, labels, masks = data
            bg, labels, masks = bg.to(device), labels.to(device), masks.to(device)
            node_feats = bg.ndata.pop('atomic')
            # edge_feats = bg.edata.pop('e').to(device)
            
            outputs = model(bg, node_feats.to(device))
            loss = (F.mse_loss(outputs,labels[:,args.property_n])* (masks != 0).float()).mean()
            total_loss += loss.item()

    return total_loss/len(data_iter)


def test(data_iter):
    with torch.no_grad():
        model.eval()
        total_mae_loss0 = 0
        total_mae_loss1 = 0
        total_mse_loss0 = 0
        total_mse_loss1 = 0
        for data in data_iter:
            smiles, bg, labels, masks = data
            bg, labels, masks = bg.to(device), labels.to(device), masks.to(device)
            node_feats = bg.ndata.pop('atomic')
            # edge_feats = bg.edata.pop('e').to(device)

            outputs=model(bg, node_feats.to(device))
            total_mae_loss0 +=(outputs[:,0] - labels[:,0]).abs().mean().item()
            total_mae_loss1 +=(outputs[:,1] - labels[:,1]).abs().mean().item()
            total_mse_loss0 += F.mse_loss(outputs[:,0],labels[:,0]).item()
            total_mse_loss1 += F.mse_loss(outputs[:,1],labels[:,1]).item()
    return total_mae_loss0/len(data_iter),total_mae_loss1/len(data_iter),total_mse_loss0/len(data_iter),total_mse_loss1/len(data_iter)

def collate_molgraphs(data):

    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction Modeling',parents=[config.parser])
    parser.add_argument('--save_name', type=str, default='pre_hl.pt',help='the name of save model')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)

    train_set, val_set, test_set  = torch.load("data/opv_graph.pt")
    train_iter = DataLoader(train_set, args.batch_size, shuffle=True, collate_fn=collate_molgraphs)
    val_iter = DataLoader(val_set, args.batch_size, shuffle=False, collate_fn=collate_molgraphs)
    test_iter = DataLoader(test_set, args.batch_size, shuffle=False, collate_fn=collate_molgraphs)
    
    model = PRE(h_size=args.h_size,emb_h=args.emb_h,dim=args.hid_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    
    best_vloss =10
    writer = SummaryWriter()

    for epoch in range(1,args.epochs+1):
        start_time = time.time()
        train_loss = train(train_iter)
        val_loss = evaluate(val_iter)

        print('-' * 80)
        print('epoch: {:4d} | time: {:4.4f}s | train loss: {:4.6f} | valid loss: {:4.6f}'.format
              (epoch, time.time() - start_time, train_loss, val_loss))

        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Valid Loss', val_loss, epoch)
        
        if val_loss < best_vloss:
            print('-' * 80)
            print('Save model!')
            torch.save(model, 'results/'+ args.save_name)
            best_vloss = val_loss

    writer.close()

    model = torch.load("results/"+args.save_name)
    test_mae_loss0,test_mae_loss1,test_mse_loss0,test_mse_loss1=test(test_iter)
    print('=' * 40)
    print('End of training | HOMO MAE {:4.6f} | HOMO RMSE {:4.6f}'.format(test_mae_loss0,test_mse_loss0**0.5))
    print('End of training | LUMO MAE {:4.6f} | LUMO RMSE {:4.6f}'.format(test_mae_loss1,test_mse_loss1**0.5))
    print('=' * 40)
    