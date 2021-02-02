import torch
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
import pandas as pd


def sample(gen_model,labels,batch_size):
    num_samples = labels.size(0)
    samples = torch.zeros(num_samples, 160).long()
    if torch.cuda.is_available()==True:
        samples=samples.cuda()

    with torch.no_grad():
        gen_model.eval()
        for i in range(0,num_samples,batch_size):
            targets = labels[i:min(i+batch_size,num_samples)]
            inputs = torch.ones(targets.size(0), 1, dtype=torch.long)
            if torch.cuda.is_available()==True:
                targets = targets.cuda()
                inputs = inputs.cuda()
                
            for j in range(160):
                out = gen_model(inputs,targets)
                final_outputs = out.contiguous()[:,-1,:]
                next_tokens = torch.multinomial(F.softmax(final_outputs,dim=-1), 1)
                inputs = torch.cat((inputs,next_tokens),-1)
                if torch.sum(next_tokens).item()==0:
                    break

            samples[i:min(i+batch_size,num_samples),:(inputs.size(1)-1)] = inputs[:,1:]

    return samples


def ts2sms(tensors):
    word2idx, idx2word = torch.load("data/opv_dic.pt")
    smiles=[]
    idxs=[]
    for i in range(len(tensors)):
        sms=''
        for t in tensors[i]:
            if t!=0:
                sms += idx2word[t] 
            else:
                break

        if bool(Chem.MolFromSmiles(sms)):
            smiles+=[sms]
            idxs+=[i]
    return smiles,idxs


def get_prop(smiles_list,all_mols,df):
    homo_list=[]
    lumo_list=[]
    for smiles in smiles_list:
        if smiles in all_mols:
            j=all_mols.index(smiles)
            homo_list+=[df.iloc[j]['homo']]
            lumo_list+=[df.iloc[j]['lumo']]
        else:
            homo_list+=[None]
            lumo_list+=[None]

    return homo_list,lumo_list


def get_simi(sms,all_sms):
    simis=[]
    all_mols = [Chem.MolFromSmiles(smi) for smi in all_sms]
    all_fps = [AllChem.GetMorganFingerprint(mol,2) for mol in all_mols]
    for sm in sms:
        fp=AllChem.GetMorganFingerprint(Chem.MolFromSmiles(sm),2) 
        simis+=[max(DataStructs.BulkDiceSimilarity(fp, all_fps))]
    return simis


def smiles_transfer(smiles_list):
    data_list=[]
    word2idx, idx2word = torch.load("data/opv_dic.pt")
    data_sm = tok(smiles_list, word2idx)
    
    for k in range(len(smiles_list)):
        m=Chem.MolFromSmiles(smiles_list[k])
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

        data_list += [Data(x=torch.LongTensor(x),edge_index=torch.LongTensor(edges_index).T,
                        edges_attr=torch.LongTensor(edges_attr),sm=data_sm[k].unsqueeze(0))]
        
    return data_list


def run_pre(data_list,model_path):
    model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    outs_h=[]
    outs_l=[]
    data_iter = DataLoader(data_list, 64, shuffle=False)
    with torch.no_grad():
        for data in data_iter:
            outs = model(data.to(device))
            outs_h += outs[:,0].view(-1).detach().cpu().numpy().tolist()
            outs_l += outs[:,1].view(-1).detach().cpu().numpy().tolist()
    return outs_h,outs_l


if __name__ == "__main__":

    Labels = torch.rand([4000,2])
    Labels[:,0] = Labels[:,0]*1.8-7
    Labels[:,1] = Labels[:,1]*1.8-4

    Labels0=torch.linspace(-7.6,-4.6,16).repeat(1,16).T
    Labels1=-torch.linspace(1.6,4.6,16).repeat(16,1).T.reshape(256,1)
    Labels=torch.cat([Labels0,Labels1],dim=-1)

    df = pd.DataFrame({'homo_tar':Labels[:,0].tolist(),'lumo_tar':Labels[:,1].tolist()})

    df0 = pd.read_csv('data/opv.csv',index_col=0)
    df0 = df0[(df0['lumo']-df0['homo'])>0].reset_index(drop=True)
    all_sms = df0['smiles'].tolist()

    gen = torch.load("results/cg_hl.pt")
    
    count=30

    for i in range(count):
        gen_samples = sample(gen, Labels, batch_size=64)
        sam_smiles,idxs = ts2sms(gen_samples) 
        data_list = smiles_transfer(sam_smiles)
        homo_pre, lumo_pre = run_pre(data_list,"results/pre_hl.pt")
        df.loc[idxs,'smiles'+str(i)]=sam_smiles
        df.loc[idxs,'homo_pre'+str(i)]=homo_pre
        df.loc[idxs,'lumo_pre'+str(i)]=lumo_pre
        df.loc[idxs,'similarity'+str(i)]=get_simi(sam_smiles,all_sms)

    df.to_csv("results/sample.csv")