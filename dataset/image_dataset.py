from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils import config
from skimage import io
from PIL import Image
from image_utils import RandomCrop, RandomFlip, RandomRotate, ImageStandardizer
# import image_utils
from torchvision import transforms, utils
from importlib import reload
import pickle
from tqdm import tqdm
import cv2
def get_train_val_test_loaders(seed, config_str, batch_size, augmentation = [], num_classes = 3, bias_te = False, data = "", labels = []):
    print("Get train_val_test_loaders")
    loaders, std = get_train_val_dataset(seed, config_str, augmentation, num_classes=num_classes, bias_te= bias_te, data = data, labels = labels)
    
    num_workers = 7
    def _init_fn(worker_id):
        np.random.seed(seed)
    
    for idx, loader in enumerate(loaders):
        if config_str == "image.michigan" or config_str == "image.mimic_test" or config_str == "fused.mimic_test":
            if idx != 2:
                loaders[idx] = loader
            else:
                loader = DataLoader(loader, batch_size = batch_size, shuffle = idx == 0, num_workers = num_workers, pin_memory = True, worker_init_fn = _init_fn)
                loaders[idx] = loader
        else:
            loader = DataLoader(loader, batch_size = batch_size, shuffle = idx == 0, num_workers = num_workers, pin_memory = True, worker_init_fn = _init_fn)
            loaders[idx] = loader

#     tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory=True, worker_init_fn=_init_fn)
#     va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers = num_workers, pin_memory=True, worker_init_fn=_init_fn)
#     te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers = num_workers, pin_memory=True, worker_init_fn=_init_fn)
        

    return loaders, std

def get_train_val_dataset(seed, config_str, augmentation, num_classes=3, bias_te = False, data = "", labels = []):
    disk = config(config_str + ".disk")
    try:
        standardize = config(config_str + ".standardize")
    except:
        standardize = False
    loaders = []
    
    loaders.append(PacsDataset(seed, config_str, 'train', augmentation, num_classes, data, labels))
    loaders.append(PacsDataset(seed, config_str, 'valid', augmentation, num_classes, data, labels))
    loaders.append(PacsDataset(seed, config_str, 'test', augmentation, num_classes, data, labels))
    
    if (bias_te):
        loaders.append(PacsDataset(seed, config_str, 'test', augmentation, num_classes, data, labels, bias_test = bias_te))
    # Standardize
    print("MAKE SURE YOU'RE USING THE RIGHT STANDARDIZER FOR THE SYNTHETIC IMAGES")
        
    if "mimic_chexpert" in config_str:
        print("no standardization")
        image_mean = [0.0, 0.0,0.0]
        image_std = [1.0, 1.0, 1.0]
    else:
        print("Standardizing to ImageNet mean and std")
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
    
    standardizer = ImageStandardizer(image_mean, image_std)


    print("Mean:", standardizer.image_mean)
    print("Std:", standardizer.image_std)
    if (not disk):
        for idx, loader in enumerate(loaders):
            loader.X = standardizer.transform(loader.X)
            loaders[idx] = loader
    else:        
        
        print("Mean:", standardizer.image_mean)
        print("Std:", standardizer.image_std)
        for idx, loader in enumerate(loaders):
            loader.standardizer = standardizer
            loaders[idx] = loader

    return loaders, standardizer



class PacsDataset(Dataset):

    def __init__(self, seed, config_str, partition, augmentation, num_classes=3, metadata = "", labels = [], bias_test = False):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()

        if partition not in ['train', 'valid', 'test']:
            raise ValueError('Partition {} does not exist'.format(partition))
        
        """
        Inititalize variables that tell me what model we're using
        """
        self.labels = labels
        np.random.seed(0)
        self.get_from_disk = config(config_str + ".disk")
        self.split = partition
        self.config_str = config_str
#         print("config_str:", config_str)
        self.image_size = int(config(self.config_str + ".image_size"))
        self.num_classes = config(self.config_str + ".num_classes")
        if (metadata == ""):
            self.metadata = pd.read_csv(config(self.config_str + '.csv_file'), index_col=0)
        else:
            self.metadata = pd.read_csv(metadata)
        self.class_labels = config(config_str + ".labels").split("|")
        if len(labels):
            for l in labels:
                self.class_labels.append(l)
                
        print("Class labels:", self.class_labels)
        
        
        """
        Check if synthetic and what to modify in the image 
        """
            
        try: 
            self.synthetic = config(self.config_str + ".synthetic_bias")
        except:
            self.synthetic = False 
            
            
        self.bias_test = bias_test
        
            
        """
        Initialize augmentation variables 
        """
        self.augmentation = augmentation 
        self.list_of_transformations = []
        self.position = "uniform" if self.split == "train" else "center"
        self.crop = RandomCrop(seed, self.image_size, self.position) 
        
        print("split:", self.split)
        
        if self.split == "train":
            if "flip" in self.augmentation:
                self.flip = RandomFlip(seed, config(self.config_str + ".flip_probability"))
                self.list_of_transformations.append(self.flip)

            if "rotate" in self.augmentation:
                print("rotating!")
                self.rotate = RandomRotate(seed, config(self.config_str + ".rotate_degrees"), self.split)
                self.list_of_transformations.append(self.rotate)
                
        self.list_of_transformations.append(self.crop)
        self.composed = transforms.Compose(self.list_of_transformations)

        """
        Reads in the necessary data from disk.
        """
#         self.class_meta, self.X, self.y, self.y_orig = self._load_data()
        self.class_meta, self.X, self.y = self._load_data()

    def __len__(self):
        if (not self.get_from_disk):
            return len(self.X)
        else:
            return len(self.class_meta)
    
    def __getitem__(self, idx):
        if (self.get_from_disk):
            curr_data = self.class_meta.iloc[idx]
            
#             curr_path = row["local_path"]
            img = self.composed(self.standardizer._transform_image(io.imread(curr_data["local_path"])/ 255)).transpose(2,0,1)
           
#             img = io.imread(curr_data["local_path"], as_gray = True)
#             print(img.shape)
            y_lab = np.array([curr_data[l] for l in self.class_labels])
            pt_id = curr_data["pt_id"]
#             if (config(self.config_str + ".threshold") != -1):
#                 threshold = config(self.config_str + ".threshold")
#                 ones = y_lab < threshold
#                 zeros = y_lab > threshold
#                 y_lab[ones] = 1
#                 y_lab[zeros] = 0
        else:
            img = np.squeeze(self.X[idx])
            img =  self.composed(img).transpose(2,0,1)
#         only modify train/validation
            y_lab = self.y[idx]
#         check synethic                              
            pt_id = self.class_meta[idx]["pt_id"]
            
#             y_orig_lab = self.y_orig[idx]
#         return torch.from_numpy(img).float(), torch.tensor(y_lab).float(), torch.tensor(y_orig_lab) ,torch.tensor(pt_id)
        return torch.from_numpy(img).float(), torch.tensor(y_lab).float(),torch.tensor(pt_id)
 

    
    def _load_data(self):
        """
        Loads a single data partition from file.
        """
#         print("loading %s..." % self.split)
        
        df = self.metadata[self.metadata.split == self.split]
        print("Getting from disk:", self.get_from_disk)
        if (self.get_from_disk):
            return df, [], []
        

        X, y = [], []
        image_paths = []
        meta = [] 


        for i, row in tqdm(df.iterrows()):
            labels = np.array([row[l] for l in self.class_labels])
            curr_path = row["local_path"]
            if curr_path[0] == "/":
                curr_path = curr_path[1:]
            try:
                
                image = io.imread(curr_path.replace("data1/home/sjabbour/Research/", "~/Chest/chest-x-ray/").replace(".png", ".jpg")) / 255
            except:
                
                print(curr_path.replace("data1/home/sjabbour/Research/", "~/Chest/chest-x-ray/").replace(".png", ".jpg"))
                image = io.imread(curr_path.replace(".png", ".jpg")) / 255
            X.append(np.array([image]))
            meta.append(row)
            y.append(labels)
        y = np.array(y)
#         y_orig = y.copy()
#         X = np.array(X)
        if (len(config(self.config_str + ".threshold")) == 1):
            threshold = config(self.config_str + ".threshold")[0]
            if threshold != -1:
                ones = y < threshold
                zeros = y > threshold
                y[ones] = 1
                y[zeros] = 0
        
        else:
            thresholds = config(self.config_str + ".threshold")
            for i, t in enumerate(thresholds):
                if t != -1:
                    ones = y[:,i] < t 
                    zeros = y[:, i] > t
                    y[:,i][ones] = 1
                    y[:,i][zeros] = 0
  
        return meta, X, y


    