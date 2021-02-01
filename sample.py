import torch
import torch.nn.functional as F
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from run import run_pre,smiles_transfer


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

    gen = torch.load("results/cg_homo&lumo.pt")
    
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