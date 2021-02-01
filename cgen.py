import argparse
import config
import time
import math
import torch
from torch import nn,optim
from torch.utils.data import TensorDataset,DataLoader
from model import GEN
from torch.utils.tensorboard import SummaryWriter


def train(train_iter):
    model.train()
    total_loss = 0

    for data, labels in train_iter:
        inputs = data[:, :-1].to(device)
        targets = data[:, 1:].to(device)

        optimizer.zero_grad()
        outputs = model(inputs,labels.to(device))
        loss = criterion(outputs.view(-1, args.dic_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_iter)


def evaluate(data_iter):
    with torch.no_grad():
        model.eval()
        total_loss = 0

        for data, labels in data_iter:
            inputs = data[:, :-1].to(device)
            targets = data[:, 1:].to(device)

            outputs = model(inputs,labels.to(device))
            loss = criterion(outputs.view(-1, args.dic_size), targets.view(-1))
            total_loss += loss.item()

    return total_loss / len(data_iter)


def iterator(data_list,split_num0,split_num1,batch_size):
    train_data=TensorDataset(*map(lambda x: x[:split_num0],data_list))
    val_data=TensorDataset(*map(lambda x: x[split_num0:split_num1],data_list))
    test_data=TensorDataset(*map(lambda x: x[split_num1:],data_list))
    train_iter = DataLoader(train_data, batch_size, shuffle=True)
    val_iter = DataLoader(val_data, batch_size, shuffle=False)
    test_iter = DataLoader(test_data, batch_size, shuffle=False)
    return train_iter,val_iter,test_iter


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generative Modeling',parents=[config.parser])
    parser.add_argument('--save_name', type=str, default='cg_hl.pt',help='the name of save model')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)
    
    Inputs,Labels = torch.load("data/opv_smiles.pt")
    Labels = Labels[:,args.property_n]
    train_iter,val_iter,test_iter = iterator([Inputs,Labels],-10000,-5000,args.batch_size)

    model = GEN(n_props=len(args.property_n), dic_size=args.dic_size, emb_size=args.emsize, 
                hid_size=args.nhid, n_levels=args.levels, kernel_size=args.ksize, dropout=args.dropout)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    best_vloss = 10
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss=train(train_iter)
        val_loss = evaluate(val_iter) 
        print('-' * 100)
        print('epoch: {:4d} | time: {:4.4f}s | train loss: {:4.6f} | valid loss: {:4.6f} | valid ppl: {:8.6f}'
              .format(epoch, time.time()-start_time, train_loss, val_loss, math.exp(val_loss)))

        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Valid Loss', val_loss, epoch)

        if val_loss < best_vloss:
            print('-' * 100)
            print('Save model!')
            torch.save(model, "results/" + args.save_name)
            best_vloss = val_loss

    writer.close()

    model=torch.load("results/" + args.save_name)
    test_loss = evaluate(test_iter)
    print('=' * 60)
    print('End of training | test loss {:4.6f} | test ppl {:8.6f}'.format(test_loss, math.exp(test_loss)))
    print('=' * 60)
