'''
Late Fuse for images + ehr data 

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision import models
from utils import config

class LateFuse(nn.Module):
    def __init__(self, config_str, ehr_feature_size, n_layers = 1):
        super(LateFuse, self).__init__()
        self.num_classes = config(config_str + '.num_classes')
        self.n_layers = n_layers
        print("Late fuse model being loaded with:", n_layers, "layers")
        use_pretrained = True
        self.config_str = config_str
        
        try: 
            pretrain_file = config(self.config_str + '.pretrain_file')
        except:
            pretrain_file = False
            
        
        try: 
            self.pretrain_ehr = config(self.config_str + '.pretrain_ehr')
        except:
            self.pretrain_ehr = False
            
            

        if (pretrain_file):
            self.pretrain_classes = config(config_str + '.pretrain_classes')
            print("loading densenet from file:", pretrain_file)
            if torch.cuda.is_available():
                checkpoint = torch.load(pretrain_file)
            else:
                print("loading onto CPU:")
                checkpoint = torch.load(pretrain_file, map_location=torch.device('cpu'))

        
            state_dict = checkpoint['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            self.densenet = models.densenet121(pretrained=True)
            self.densenet.classifier = nn.Linear(1024, self.pretrain_classes)
            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.','')
                new_state_dict[k]=v

            self.densenet.load_state_dict(new_state_dict)
            
        else:
            print("using ImageNet initialized model")
            self.densenet = models.densenet121(pretrained=True)
           
        print("Feature sizE:", ehr_feature_size)

        if (self.n_layers == 2):
            if ehr_feature_size < 1000:
                self.ehr_hidden = 100
            else:
                self.ehr_hidden = 1024
            self.fc1 = nn.Linear(ehr_feature_size, self.ehr_hidden)

            self.fc2 = nn.Linear(1024 + self.ehr_hidden, self.num_classes)
            if self.pretrain_ehr:

                print("loading in pretraining weight from densenet")
                old_weights_image = self.densenet.classifier.weight.data
                with torch.no_grad():
                    self.fc2.weight[:,0:1024].copy_(old_weights_image)
                print("loading in pretraining weight from ehr")
                print(config(config_str + ".ehr_model"))
                self.ehr_model = torch.load(config(config_str + ".ehr_model"))["state_dict"]

                with torch.no_grad():
                    old_weights_ehr = self.ehr_model["module.module.fc1.weight"]
                    print(old_weights_ehr.shape, self.fc1.weight.shape)
                    self.fc1.weight.copy_(old_weights_ehr)

                    old_weights_ehr = self.ehr_model["module.module.fc2.weight"]
                    print(old_weights_ehr.shape, self.fc2.weight[:,1024:].shape)
                    self.fc2.weight[:, 1024:].copy_(old_weights_ehr)
        else:
            self.fc1 = nn.Linear(1024 + ehr_feature_size, self.num_classes)

#         if self.pretrain_ehr:
#             print("Using image classifier weights")
#             old_weights = self.densenet.classifier.weight.data
#             with torch.no_grad():
#                 self.fc1.weight[:,0:1024].copy_(old_weights)
                
                
                
                
                
#                 print(self.fc1.weight[:,0:1024].shape, old_weights.shape)
#                 self.fc1.weight[:,0:1024] = old_weights
#     # or
#                 print(self.fc1.weight[:,0:1024].is_leaf) # True

#         if (self.pretrain_ehr):
#             self.hidden = 100
# #             load ehr model 
#             print("loading ehr from file:", self.pretrain_ehr)
#             checkpoint = torch.load(self.pretrain_ehr)
#             state_dict = checkpoint['state_dict']
#             new_state_dict = OrderedDict()
#             for k, v in state_dict.items():
#                 if 'module' in k and "fc1" in k:
#                     k = k.replace('module.module.fc1.','')
#                     new_state_dict[k]=v
            
#             self.fc1 = nn.Linear(ehr_feature_size, self.hidden)
#             print("new state dict:", new_state_dict.keys())
#             self.fc1.load_state_dict(new_state_dict)
            
#             new_state_dict = OrderedDict()
#             for k, v in state_dict.items():
#                 if 'module' in k and "fc2" in k:
#                     k = k.replace('module.module.fc2.','')
#                     new_state_dict[k]=v
            
#             temp_fc2 = nn.Linear(self.hidden, self.num_classes)
#             temp_fc2 = nn.Linear(self.hidden, self.num_classes)
#             temp_fc2.load_state_dict(new_state_dict)
            
#             with torch.no_grad:
#                 print("FC2 to classifier:")
#                 print(self.fc2.weight[:, 1024:].shape, temp_fc2.weight.data.shape)
#                 self.fc2.weight[:, 1024:].copy(temp_fc2.weight.data)
#             self.ReLu = nn.ReLU()

#             self.S = nn.Sequential(self.fc1, self.ReLu, self.fc2)

#             self.fc3 = nn.Linear(2, self.num_classes)
#         else:
            
            
    def forward(self, image_and_data):
        image = image_and_data[0]
        data = image_and_data[1]
        


        features = self.densenet.features(image)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        x1 = torch.flatten(out, 1)

        if (self.n_layers == 2):
            ehr_hidden = F.relu(self.fc1(torch.squeeze(data, dim = 1)))
            x = self.fc2(torch.cat((x1, ehr_hidden), dim = 1))
#             x = F.relu(self.fc1(torch.cat((x1, torch.squeeze(data, dim = 1)), dim = 1)))
#             x = self.fc2(x)
        elif (self.n_layers == 3):
            x = F.relu(self.fc1(torch.squeeze(data, dim = 1)))
            x = torch.cat((x1, x), dim = 1)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.fc1(torch.cat((x1, torch.squeeze(data, dim = 1)), dim = 1))

        return x
   
#         if (self.pretrain_ehr):
#             image_pred = torch.squeeze(self.densenet(image), dim =1)
            
#             ehr_pred = torch.squeeze(self.S(data))
#             x = torch.stack((image_pred,ehr_pred), dim = 0).T
#             x = self.fc3(x)
            
#         else: