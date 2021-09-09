import torch
from torch.utils.data import Dataset, DataLoader
from utils import config 
import numpy as np
import pickle
def get_fused_loader(seed, image_loader, ehr_loader, batch_size, split):
    print(type(ehr_loader))
    loader = get_dataset(seed, image_loader, ehr_loader,batch_size, split)
    return loader

def get_dataset(seed, image_loader, ehr_loader,batch_size, split):
    dataset = FusedDataset(image_loader, ehr_loader)
    
    def _init_fn(worker_id):
        np.random.seed(seed)
    
    num_workers = 7
    if split == "train":
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True , num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False , num_workers=num_workers, pin_memory=True, worker_init_fn=_init_fn)
    return loader

    
class FusedDataset(Dataset):
    def __init__(self, image_loader, ehr_loader):  
        """
        Save EHR and Image dataloaders    
        """       
        self.image_loader = image_loader
        self.ehr_loader = ehr_loader
    def __getitem__(self, idx):

#         after 
        img = np.squeeze(self.image_loader.dataset.X[idx])
        img = self.image_loader.dataset.composed(img).transpose(2,0,1)


            
#         get patient id 
        pt_id = self.image_loader.dataset.class_meta[idx]["pt_id"]
#         get patient label 
        y = self.image_loader.dataset.y[idx]
        
        
        idx_ehr = np.where(self.ehr_loader.dataset.meta["pt_id"].to_numpy() == pt_id) 
        if (len(idx_ehr)> 1):
            idx_ehr = idx_ehr[0]
        X_ehr = self.ehr_loader.dataset.X[idx_ehr]
        s_ehr = self.ehr_loader.dataset.s[idx_ehr]
#         TODO: inc this later when we use time dependent data too 
        xi = s_ehr
        if (xi.shape[0] == 0):
            print(idx_ehr, pt_id)
#         xi = np.tile(s_ehr,(1,1))
        data_ret = (torch.from_numpy(img).float(), torch.from_numpy(xi).float())

        return (
            data_ret, 
            torch.tensor(y).long(),
            pt_id
        )
    
    def __len__(self):
        return len(self.image_loader.dataset.y)
    
