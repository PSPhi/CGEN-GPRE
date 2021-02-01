import argparse
import config
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import PRE
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train(train_iter):
    model.train()
    total_loss = 0
    for data in train_iter:
        data=data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data),data.y[:,args.property_n])
        loss.backward()
        total_loss += loss.item()*data.num_graphs
        optimizer.step()  

    return total_loss/len(train_iter.dataset)


def evaluate(data_iter):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        for data in data_iter:
            data = data.to(device)
            loss = F.mse_loss(model(data),data.y[:,args.property_n])
            total_loss += loss.item()*data.num_graphs

    return total_loss/len(data_iter.dataset)


def test(data_iter):
    with torch.no_grad():
        model.eval()
        total_mae_loss0 = 0
        total_mae_loss1 = 0
        total_mse_loss0 = 0
        total_mse_loss1 = 0
        for data in data_iter:
            data = data.to(device)
            output=model(data)
            total_mae_loss0 +=(output[:,0] - data.y[:,0]).abs().sum().item()
            total_mae_loss1 +=(output[:,1] - data.y[:,1]).abs().sum().item()
            total_mse_loss0 += F.mse_loss(output[:,0],data.y[:,0]).item()*data.num_graphs
            total_mse_loss1 += F.mse_loss(output[:,1],data.y[:,1]).item()*data.num_graphs
    return total_mae_loss0/len(data_iter.dataset),total_mae_loss1/len(data_iter.dataset),total_mse_loss0/len(data_iter.dataset),total_mse_loss1/len(data_iter.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction Modeling',parents=[config.parser])
    parser.add_argument('--save_name', type=str, default='pre_hl.pt',help='the name of save model')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)

    data_list = torch.load("data/opv_graph.pt")
    train_iter = DataLoader(data_list[:-10000], args.batch_size, shuffle=True)
    val_iter = DataLoader(data_list[-10000:-5000], args.batch_size, shuffle=False)
    test_iter = DataLoader(data_list[-5000:], args.batch_size, shuffle=False)
    
    model = PRE(h_size=args.h_size,emb_h=args.emb_h,dim=args.hid_size,n_levels=args.n_levels,dropout=args.drop,out_size=len(args.property_n))

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
    