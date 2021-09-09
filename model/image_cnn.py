'''
CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.cnn import CNN
'''
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
# import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision import datasets, models, transforms
from math import sqrt
# from train_common import config


def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, 'config'):
        with open('config.json') as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split('.'):
        node = node[part]
    return node

class CNN_PACS(nn.Module):
    def __init__(self, data_type, model, pretrain, device = [], params = [], pretrain_file = ""):
        super().__init__()
        self.config_str = data_type + "." + model
        num_classes = config(data_type + "." + model + '.num_classes')
        
        if (pretrain_file == ""):
            try: 
                pretrain_file = config(self.config_str + '.pretrain_file')
            except:
                pretrain_file = False

        if (pretrain_file):
            print("using pretrained densenet:", pretrain_file)

            self.model = models.densenet121(pretrained=pretrain)


            if (pretrain_file):
                print("Pretrained densenet file:", pretrain_file)
                if torch.cuda.is_available():
                    checkpoint = torch.load(pretrain_file)
                else:
                    print("loading on CPU, gpu not available")
                    checkpoint = torch.load(pretrain_file,map_location=torch.device('cpu'))

                state_dict = checkpoint['state_dict']
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                pretrain_classes = config(self.config_str + '.pretrain_classes')

                self.model.classifier = nn.Linear(1024, pretrain_classes)
                for k, v in state_dict.items():
                    if 'module' in k:
                        k = k.replace('module.','')
                    new_state_dict[k]=v
                self.model.load_state_dict(new_state_dict)
                if num_classes == pretrain_classes:
                    print("keeping classifier weights!")
                else:
                    self.model.classifier = nn.Linear(1024, num_classes)
        
        else:
            print("Using ImageNet initialized model")
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(1024, num_classes)
      
class CNN_PACS_classifier(nn.Module):
    def __init__(self, data_type, model, pretrain):
        super().__init__()
        self.config_str = data_type + "." + model
        num_classes = config(data_type + "." + model + '.num_classes')
        
        print("using pretrained densenet:", pretrain)
        self.model = models.densenet121(pretrained=pretrain)

        try: 
            pretrain_file = config(self.config_str + '.pretrain_file')
            print("Pretrained densenet file:", pretrain_file)

            checkpoint = torch.load(pretrain_file)
            state_dict = checkpoint['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            self.model.classifier = nn.Linear(1024, 14)
            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.','')
                new_state_dict[k]=v

            self.model.load_state_dict(new_state_dict)
            print("loaded pretrained chexpert from:", pretrain_file)
        except:
            pass


        self.fc1 = nn.Linear(14, num_classes)
        
    def forward(self, image):
    
        X = self.fc1(F.relu(self.model(image))) 
        return X 
            
class CNN_aux(nn.Module):
    def __init__(self, data_type, model, pretrain):
        super().__init__()
        self.model_name = model 
        self.config_str = data_type + "." + model
        num_classes = config(data_type + "." + model + '.num_classes')
        
        print("using pretrained densenet:", pretrain)
        self.model = models.densenet121(pretrained=pretrain)

        try: 
            pretrain_file = config(self.config_str + '.pretrain_file')
            checkpoint = torch.load(pretrain_file)
            state_dict = checkpoint['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            self.model.classifier = nn.Linear(1024, 14)
            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.','')
                new_state_dict[k]=v

            self.model.load_state_dict(new_state_dict)
            print("loaded pretrained chexpert from:", pretrain_file)
        except:
            pass

                
        
        self.model.classifier = nn.Linear(1024, num_classes)
    
        
#         batch norm 
        self.norm = nn.BatchNorm2d(64)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=3)

        self.flatten = nn.Flatten()
        if (self.model_name == "bias_predict_clarity"):
            self.classifier = nn.Linear(112896, 773)

        elif (self.model_name == "bias_predict_ehr"):
            self.classifier = nn.Linear(112896, 164)
        elif (self.model_name == "bias_predict_chf"):
            self.classifier = nn.Linear(112896, 177)
        elif(self.model_name == "bias_predict_all"):
            self.classifier = nn.Linear(112896, 1112)


        
        
    def forward(self, image_and_data):
        
        # First convolution
        image = image_and_data[0]
#         data = image_and_data[1]
        
#         ehr pass 
        ehr = self.model.features.conv0(image)
        ehr = self.model.features.norm0(ehr)
        ehr = self.model.features.relu0(ehr)
        ehr = self.model.features.pool0(ehr)
        ehr_prediction = self.classifier(self.flatten(self.pool(self.norm(ehr))))
        
#         ehr_prediction = self.ehr_pass(image)
        
        image_prediction = self.model(image)
        
        
        return (image_prediction, ehr_prediction)
    
    
class CNN_aux_end(nn.Module):
    def __init__(self, data_type, model, pretrain):
        super().__init__()
        self.model_name = model 
        self.config_str = data_type + "." + model
        self.num_classes = config(data_type + "." + model + '.num_classes')
        
        print("using pretrained densenet:", pretrain)
        self.model = models.densenet121(pretrained=pretrain)

        try: 
            pretrain_file = config(self.config_str + '.pretrain_file')
            print("using densenet:", pretrain_file)
            checkpoint = torch.load(pretrain_file)
            state_dict = checkpoint['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            self.model.classifier = nn.Linear(1024, 14)
            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.','')
                new_state_dict[k]=v

            self.model.load_state_dict(new_state_dict)
            print("loaded pretrained chexpert from:", pretrain_file)
        except:
            pass

                
        
        self.model.classifier = nn.Linear(1024, self.num_classes + 170)


        
        
    def forward(self, image_and_data):
        image = image_and_data[0]
        
        predictions = self.model(image)
        image_predictions = predictions[:, 0:self.num_classes]
        ehr_predictions = predictions[:, self.num_classes:]
        return (image_predictions, ehr_predictions)
   


            
                

class CNN_autoencoder(nn.Module):
    def __init__(self, data_type, model, pretrain, device = [], params = [], pretrain_file = ""):
        super().__init__()
        self.config_str = data_type + "." + model
        num_classes = config(data_type + "." + model + '.num_classes')
        
        try:
            pretrain = config(self.config_str + ".pretrain")
        except:
            pretrain = False 
         
        if (pretrain_file == ""):
            try: 
                pretrain_file = config(self.config_str + '.pretrain_file')
            except:
                pretrain_file = False
        try:
            penalize_classifier = config(self.config_str + '.penalize_classifier')
        except:
            penalize_classifier = False 
        if (pretrain != "random"):
            print("using pretrained densenet:", pretrain)

            self.model = models.densenet121(pretrained=pretrain)
            self.model.old_weights = []


            self.model.conv1 = nn.ConvTranspose2d(1024, 16, 83, stride=3)  # b, 16, 5, 5
            self.relu = nn.ReLU(True)
            self.model.conv2 =  nn.ConvTranspose2d(16, 8, 2, stride=2, padding=0)  # b, 8, 15, 15
            self.model.conv3 = nn.ConvTranspose2d(8, 3, 2, stride=2, padding=0)  # b, 1, 28, 28

        
        
            if (pretrain_file):
                print("Pretrained densenet file:", pretrain_file)

                checkpoint = torch.load(pretrain_file,map_location=torch.device('cpu') )
                state_dict = checkpoint['state_dict']
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                pretrain_classes = config(self.config_str + '.pretrain_classes')

                self.model.classifier = nn.Linear(1024, pretrain_classes)
                
                for k, v in state_dict.items():
                    if 'module.model' in k:
                        k = k.replace('module.model.','')
                    new_state_dict[k]=v
                self.model.load_state_dict(new_state_dict)
                
                
        else:
            print("randomly initializing densenet")
            self.model = models.densenet121()
    

    
    def forward(self, x):
        x = self.model.features(x)
        
        x = self.model.conv1(x)
        
        x  = self.relu(x)
        x = self.model.conv2(x)
        
        x= self.relu(x)
        x = self.model.conv3(x)
        x = self.relu(x)
        return x
        
        
                                        
            
#     class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)


#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
                                        
#     def forward(self, image):
#         x = self.model(image)
        
#         X = self.fc1(F.relu(self.model(image))) 
#         return X 
 
                
                
                
# class CNN_PACS_classifier(nn.Module):
#     def __init__(self, data_type, model, pretrain):
#         super().__init__()
#         self.config_str = data_type + "." + model
#         num_classes = config(data_type + "." + model + '.num_classes')
        
#         print("using pretrained densenet:", pretrain)
#         self.model = models.densenet121(pretrained=pretrain)

#         try: 
#             pretrain_file = config(self.config_str + '.pretrain_file')
#             print("Pretrained densenet file:", pretrain_file)

#             checkpoint = torch.load(pretrain_file)
#             state_dict = checkpoint['state_dict']
#             from collections import OrderedDict
#             new_state_dict = OrderedDict()

#             self.model.classifier = nn.Linear(1024, 14)
#             for k, v in state_dict.items():
#                 if 'module' in k:
#                     k = k.replace('module.','')
#                 new_state_dict[k]=v

#             self.model.load_state_dict(new_state_dict)
#             print("loaded pretrained chexpert from:", pretrain_file)
#         except:
#             pass


#         self.fc1 = nn.Linear(14, num_classes)
        
#     def forward(self, image):
    
#         X = self.fc1(F.relu(self.model(image))) 
#         return X 
 
    
    