import argparse

parser = argparse.ArgumentParser(add_help=False)
#General
parser.add_argument('--property_n', type=int, nargs='+', default=[0,1],
                    help='the numerical order of property (default: 0:homo,1:lumo,2:homo_calib,3:lumo_calib,4:PCE,5:PCE_calb)')#[0,1],[1]
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size (default: 32)')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate (default: 0.001)')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer type (default: Adam)')                    
parser.add_argument('--epochs', type=int, default=300, help='upper epoch limit (default: 300)')
parser.add_argument('--dic_size', type=int, default=30, help='number of character (default: 32)')
parser.add_argument('--emsize', type=int, default=32, help='size of character embeddings (default: 32)')
parser.add_argument('--ksize', type=int, default=3, help='kernel size (default: 3)')  
#Generative
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (default: 0.2)')                
parser.add_argument('--levels', type=int, default=5, help='number of levels (default: 5)')
parser.add_argument('--nhid', type=int, default=256, help='number of hidden units per layer (default: 256)')
#Predictionz
parser.add_argument('--drop', type=float, default=0, help='drop applied to layers (default: 0)')
parser.add_argument('--h_size', type=int, default=18, help='size of atoms (default: 18)')
parser.add_argument('--emb_h', type=int, default=32, help='size of atomic number embeddings (default: 32)')         
parser.add_argument('--hid_size', type=int, default=128, help='number of hidden units per layer (default: 64)')

