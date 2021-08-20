import torch
import pandas as pd
from rdkit import Chem
from dgllife.data import MoleculeCSVDataset
from functools import partial
from dgllife.utils import smiles_to_bigraph, RandomSplitter


class Dictionary(object):
    # SMILES character library
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def process(all_sms):
    # SMILES process
    all_smiles = []
    element_table = ["Cl", "Br"]
    for i in range(len(all_sms)):
        sms = all_sms[i]
        smiles = []
        j = 0
        while j < len(sms):
            sms1 = []
            # [**] as one character 
            if sms[j] == "[":
                sms1.append(sms[j])
                j = j + 1
                while sms[j] != "]":
                    sms1.append(sms[j])
                    j = j + 1
                sms1.append(sms[j])
                sms2 = ''.join(sms1)
                smiles.append(sms2)
                j = j + 1
            else:
                sms1.append(sms[j])

                if j + 1 < len(sms):
                    sms1.append(sms[j + 1])
                    sms2 = ''.join(sms1)
                else:
                    sms1.insert(0, sms[j - 1])
                    sms2 = ''.join(sms1)

                # the element in element_table as one character
                if sms2 not in element_table:
                    smiles.append(sms[j])
                    j = j + 1
                else:
                    smiles.append(sms2)
                    j = j + 2

        all_smiles.append(list(smiles))
    return all_smiles


class Corpus(object):
    # creat dictionary and tokenize
    def __init__(self, sm_list):
        self.dictionary = Dictionary()
        self.all = self.tokenize(sm_list)

    def tokenize(self, sm_list):
        self.dictionary.add_word('\n')
        all_smiles = process(sm_list)
        max_length = max([len(smiles) for smiles in all_smiles])+1

        all_ids = []
        for smiles in all_smiles:
            id = []
            words = ['&'] + smiles
            for word in words:
                self.dictionary.add_word(word)
                id += [self.dictionary.word2idx[word]]

            while len(id) < max_length:
                id += [0]

            all_ids.append(id)
        print(max_length,self.dictionary.word2idx)
        return all_ids

# Obtain the features of atoms and bonds
def featurize_atoms(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(atom.GetAtomicNum())
    return {'atomic': torch.tensor(feats).long()}

def featurize_edges(mol, self_loop=True):
    feats = []
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(num_atoms):
            e_ij = mol.GetBondBetweenAtoms(i,j)
            if e_ij is None:
                bond_type = None
            else:
                bond_type = e_ij.GetBondType()

            if i != j or self_loop:
                feats.append([float(bond_type == x)for x in (None, Chem.rdchem.BondType.SINGLE,
                              Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC)])
    return {'dist': torch.tensor(feats).long()}


if __name__ == "__main__":
    df = pd.read_csv('data/opv.csv',index_col=0)
    data = df[(df['lumo']-df['homo']>0)&(df['homo']>=-7.6)&(df['homo']<=-4.6)&(df['lumo']>=-4.6)&(df['lumo']<=-1.6)].reset_index(drop=True)

    # SMILES to Dataset for generative model
    corpus = Corpus(data['smiles'])
    print(corpus.dictionary)
    torch.save([corpus.dictionary.word2idx, corpus.dictionary.idx2word], "data/opv_dic.pt")

    torch.manual_seed(1024)
    rand_index = torch.randperm(len(data.index)).tolist()
    
    inputs=torch.LongTensor(corpus.all)[rand_index]
    targets=torch.FloatTensor(data.iloc[rand_index,2:].values)
    Inputs=inputs[(targets[:,0]>=-7)&(targets[:,0]<=-5.2)&(targets[:,1]>=-4)&(targets[:,0]<=-2.2)]
    Targets=targets[(targets[:,0]>=-7)&(targets[:,0]<=-5.2)&(targets[:,1]>=-4)&(targets[:,0]<=-2.2)]
    print(Inputs[0],Inputs.size(),Targets[0],Targets.size())
    torch.save([Inputs,Targets],"data/opv_smiles.pt")

    # SMILES to graph-based dataset for prediction model with DGL-Life 
    dataset=MoleculeCSVDataset(df=data,
                               smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                               node_featurizer=featurize_atoms,
                               edge_featurizer=None,
                               smiles_column='smiles',
                               cache_file_path='data/graph.pt') 
    
    train_set, val_set, test_set = RandomSplitter.train_val_test_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1)
    torch.save([train_set,val_set,test_set], "data/opv_graph.pt")
    
