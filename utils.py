import torch
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data


class Dictionary(object):
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
    all_smiles = []
    element_table = ["Cl", "Br"]
    for i in range(len(all_sms)):
        sms = all_sms[i]
        smiles = []
        j = 0
        while j < len(sms):
            sms1 = []
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

                if sms2 not in element_table:
                    smiles.append(sms[j])
                    j = j + 1
                else:
                    smiles.append(sms2)
                    j = j + 2

        all_smiles.append(list(smiles))
    return all_smiles


def tok(ms, word2idx):
    all_ids = []
    all_smiles = process(ms)
    max_length = max([len(smiles) for smiles in all_smiles])+1
    for smiles in all_smiles:
        ids = []
        for word in smiles:
            if word in word2idx:
                ids += [word2idx[word]]
        while len(ids) < max_length:
            ids += [0]

        all_ids.append(ids)

    return torch.LongTensor(all_ids)


class Corpus(object):
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


def smiles_transfer(df,sm_list,rand_index):
    data_list=[]
    
    for s in rand_index:
        m=Chem.MolFromSmiles(df.loc[s]['smiles'])
        #m=Chem.AddHs(m)
        n_atoms=len(m.GetAtoms())
    
        x=[]
        edges_index=[]
        edges_attr=[]
        
        for i in range(n_atoms):
            atom_i=m.GetAtomWithIdx(i)
            x.append([atom_i.GetAtomicNum()])

            for j in range(n_atoms):
                e_ij=m.GetBondBetweenAtoms(i,j)
                if e_ij is not None:
                    edges_index.append([i,j])
                    bond_type=[int(e_ij.GetBondType()==x) for x in [Chem.rdchem.BondType.SINGLE,
                               Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]]
                    edges_attr.append([bond_type.index(1)+1])

        y=[[data.loc[s]['homo'], data.loc[s]['lumo'],data.loc[s]['homo_calib'], 
            data.loc[s]['lumo_calib'],data.loc[s]['PCE'], data.loc[s]['PCE_calib']]]

        data_list += [Data(x=torch.LongTensor(x),edge_index=torch.LongTensor(edges_index).T,edges_attr=torch.LongTensor(edges_attr),
                           y=torch.Tensor(y),sm=torch.LongTensor([sm_list[s]]))]
        
    return data_list


if __name__ == "__main__":
    df = pd.read_csv('data/opv.csv',index_col=0)
    data = df[(df['lumo']-df['homo']>0)&(df['homo']>=-7.6)&(df['homo']<=-4.6)&(df['lumo']>=-4.6)&(df['lumo']<=-1.6)].reset_index(drop=True)

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

    data_list=smiles_transfer(data,corpus.all,rand_index)
    print(len(data_list),data_list[0])
    torch.save(data_list, "data/opv_graph.pt")
    
