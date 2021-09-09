import sys, os, time, pickle, random
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import sparse
import yaml
from utils import config 
from sklearn.preprocessing import LabelEncoder

def get_train_val_test_loaders(seed, config_str, batch_size, num_classes, fuse, hour, train = True):
    tr, va, te = get_train_val_dataset(seed, config_str, batch_size, num_classes, fuse, hour, train = train)
    return tr, va, te

def get_train_val_dataset(seed, config_str, batch_size, num_classes, fuse, hour, train = True):
#         read in FIDDLE data 
    reader = _AHRPReader(config_str, hour)
    Xy_tr, Xy_va, Xy_te = reader.get_splits(gap=0.0, random_state=0)
    
    
    def _init_fn(worker_id):
        np.random.seed(seed)
        
    num_workers = 14


    if (train):
        tr = EHRDataset(*Xy_tr, fuse=fuse)
        va = EHRDataset(*Xy_va, fuse=fuse)
        te = EHRDataset(*Xy_te, fuse=fuse)
        tr_loader = DataLoader(tr, batch_size=int(batch_size), shuffle=True , num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)
        va_loader = DataLoader(va, batch_size=int(batch_size), shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)
        te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)
    else:
        tr_loader, va_loader = [],[]
        te = EHRDataset(*Xy_te, fuse=fuse)
        te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)

    
    
    return tr_loader, va_loader, te_loader
     
class _AHRPReader(object):
    def __init__(self, config_str, hour):
        """
        """
        
        start_time = time.time() 
        self.config_str = config_str
        self.data_directory = config(config_str + ".data_directory")
        self.s_loc = config(config_str + ".s")
#         self.pop_idx = pd.read_csv("~/Fused/train_models/" + self.data_directory  + '/pop.csv')
        self.pop_idx = pd.read_csv(self.data_directory  + '/IDs.csv')
        self.meta = pd.read_csv(config(self.config_str + ".csv_file"))
        self.hour = hour 
        
        self.s = sparse.load_npz(self.s_loc).todense()
            
#         start_time = time.time() 
#         self.config_str = config_str
#         self.pop_idx = pd.read_csv('../RDW/data_prep_resampled_regular/pop_resampled_regular.csv')
#         self.meta = pd.read_csv(config(self.config_str + ".csv_file"))
#         self.hour = hour 
#         print("Hour:", 0)
#         print("loading from:", '../RDW/' + str(int(hour)) + '_resampled_regular/X.npz')
#         self.X = sparse.load_npz('../RDW/' + str(int(hour)) + '_resampled_regular/X.npz').todense()
#         self.s = sparse.load_npz('../RDW/' + str(int(hour)) + '_resampled_regular/s.npz').todense()
            
#         print(self.X.shape)
        print("data shape:", self.s.shape)

    def _conv_str(self,ls):
        return "".join(str(int(x)) for x in ls)
    def _select_examples(self, rows):
        pts = self.pop_idx["ID"][rows]
        meta_rows = []
        labels = []
        for pt in pts:
            
            ind = self.meta.loc[self.meta["pt_id"].isin([pt])].index.values
            ind = ind[0]
            meta_rows.append(ind)
        class_labels = config(self.config_str + ".labels").split("|")
        
        labels = self.meta.iloc[meta_rows][[l for l in class_labels]].values
        threshold = config(self.config_str + ".threshold")
        if (threshold != -1):
#             print("Thresholding labels at threshold:", threshold)
            ones = labels < threshold 
            zeros = labels > threshold  
            labels[ones] = 1
            labels[zeros] = 0

        return (
            np.zeros(len(self.s[rows])), 
            self.s[rows], 
            labels,
            self.meta.iloc[meta_rows],
        )
        
    def get_splits(self, gap=0.0, random_state=None, verbose=True):
        """
        fixed splits based on patient
        """
        pts_tr = self.meta['pt_id'][self.meta['split'] == 'train'].values
        pts_va = self.meta['pt_id'][self.meta['split'] == 'valid'].values
        pts_te = self.meta['pt_id'][self.meta['split'] == 'test'].values
    
#         self.pop_idx.to_csv("pop.csv")
#         pts_tr.to_csv("pts.csv")
        tr_idx = self.pop_idx.loc[self.pop_idx['ID'].isin(pts_tr)].index.values
        va_idx = self.pop_idx.loc[self.pop_idx['ID'].isin(pts_va)].index.values
        te_idx = self.pop_idx.loc[self.pop_idx['ID'].isin(pts_te)].index.values

        try:
            import pathlib
            pathlib.Path('./output/outcome/').mkdir(parents=True, exist_ok=True)
            np.savez(open('./output/outcome/idx.npz', 'wb'), tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx)
        except:
#             print('indices not saved')
            raise
    
        Xy_tr = self._select_examples(tr_idx)
        Xy_va = self._select_examples(va_idx)
        Xy_te = self._select_examples(te_idx)
        
        
        return Xy_tr, Xy_va, Xy_te
    
    
    
class EHRDataset(Dataset):
    def __init__(self, X, s, y, meta, fuse=False):
        assert len(X) == len(s)
        assert len(X) == len(y)
        self.X = X
        self.s = s
        self.y = y
        self.fuse = fuse
        self.meta = meta 

    def __getitem__(self, index):
        if self.fuse:
            xi = self.X[index] # LxD
            si = self.s[index] # d
            L, D = self.s.shape
            
#             print("l", L, D)
#             xi = np.hstack((xi, np.tile(si, (L, 1))))
            xi = np.tile(si,(1,1))
#             print(xi.shape)
        
#             xi = self.X[index] # LxD
#             si = self.s[index] # d
#             L, D = xi.shape
#             xi = np.hstack((xi, np.tile(si, (L, 1))))
#             xi = np.tile(si,(L,1))
    
    
#             y_array = np.zeros((1,))
#             y_array[0] = self.y[index]
            if  self.meta['pt_id'].to_numpy()[index] == 465:
                print("here")
            return (
                torch.from_numpy(xi).float(),
                torch.from_numpy(self.y[index]).float(),
                self.meta['pt_id'].to_numpy()[index],
            )
        else:
            print(self.meta['pt_id'].to_numpy()[index])
            return (
                torch.from_numpy(self.X[index]).float(),
                torch.from_numpy(self.s[index]).float(),
                torch.from_numpy(self.y[index]).float(),
                self.meta['pt_id'].to_numpy()[index]
            )
    
    def __len__(self):
        return len(self.y)
    
# get_train_val_test_loaders("ehr", "all_binary_baseline", 16, 3, True) 
