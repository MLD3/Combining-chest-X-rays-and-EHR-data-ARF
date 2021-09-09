import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN_EHR(nn.Module):
    """
    Multilayer CNN with 1D convolutions
    """
    def __init__(
        self,
        in_channels,
        L_in,
        output_size,
        depth=2,
        filter_size=3, 
        n_filters=64, 
        n_neurons=64, 
        dropout=0.2,
        activation='relu',
    ):
        super().__init__()
        self.depth = depth
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        padding = int(np.floor(filter_size / 2))
        print("OUTPUT SIZE", output_size)
        if depth == 1:
            self.conv1 = nn.Conv1d(in_channels, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(int(L_in * (n_filters) / 2), n_neurons)
            self.fc1_drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(n_neurons, output_size)
    
        elif depth == 2:
            self.conv1 = nn.Conv1d(in_channels, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(64, n_neurons)
            self.fc1_drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(n_neurons, output_size)
            
        elif depth == 3:
            self.conv1 = nn.Conv1d(in_channels, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(64, n_neurons)
            self.fc1_drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(n_neurons, output_size)
    
    
    
    def forward(self, x):
        # x: tensor (batch_size, L_in, in_channels)
        x = x.transpose(1,2) # swap time and feature axes
        
        x = self.pool1(self.activation(self.conv1(x)))
        if self.depth == 2 or self.depth == 3:
            x = self.pool2(self.activation(self.conv2(x)))
        if self.depth == 3:
            x = self.pool3(self.activation(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # flatten
        x = self.activation(self.fc1_drop(self.fc1(x)))
        x = self.fc2(x)
#         x = torch.sigmoid(self.fc2(x))
        return x

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes, depth):
        super(LogisticRegression, self).__init__()
        self.depth = depth 
        self.ReLu = nn.ReLU()
        if input_size < 1000:
            self.hidden = 100
        else:
            self.hidden = 1024
        print("depth:",depth, input_size)
        if (depth == 1):
            print("going from to" , input_size, num_classes)
            self.fc1 = nn.Linear(input_size, num_classes)
            self.S = nn.Sequential(self.fc1)
            
#             print("loading in pretrained model")
#             pretrain = torch.load("./checkpoint/ehr/mimic_pretrain//seed_0/best_OrderedDict([('batch_size', 32), ('depth', 1), ('lr', 0.1), ('momentum', 0.8), ('weight_decay', 0.0001)])_checkpoint.pth.tar")["state_dict"]
            
                
#             with torch.no_grad():
#                 old_weights_ehr = pretrain["module.module.fc1.weight"]
#                 print(old_weights_ehr.shape, self.fc1.weight.shape)
#                 self.fc1.weight.copy_(old_weights_ehr)
#                 old_weights_ehr = pretrain["module.module.fc1.bias"]
#                 print(old_weights_ehr.shape, self.fc1.weight.shape)
#                 self.fc1.bias.copy_(old_weights_ehr)
                
                
        elif depth == 2:
            self.fc1 = nn.Linear(input_size, self.hidden)
            self.fc2 = nn.Linear(self.hidden, num_classes)
            self.S = nn.Sequential(self.fc1, self.ReLu, self.fc2)
            
#             print("loading in pretrained model")
#             pretrain = torch.load("./checkpoint/ehr/mimic_pretrain//seed_1/best_OrderedDict([('batch_size', 32), ('depth', 2), ('lr', 0.01), ('momentum', 0.9), ('weight_decay', 0.0001)])_checkpoint.pth.tar")["state_dict"]
#             with torch.no_grad():
#                 old_weights_ehr = pretrain["module.module.fc1.weight"]
#                 print(old_weights_ehr.shape, self.fc1.weight.shape)
#                 self.fc1.weight.copy_(old_weights_ehr)
#                 old_weights_ehr = pretrain["module.module.fc1.bias"]
#                 print(old_weights_ehr.shape, self.fc1.weight.shape)
#                 self.fc1.bias.copy_(old_weights_ehr)
                
#                 old_weights_ehr = pretrain["module.module.fc2.weight"]
#                 print(old_weights_ehr.shape, self.fc2.weight.shape)
#                 self.fc2.weight.copy_(old_weights_ehr)
#                 old_weights_ehr = pretrain["module.module.fc2.bias"]
#                 print(old_weights_ehr.shape, self.fc2.weight.shape)
#                 self.fc2.bias.copy_(old_weights_ehr)
                
                
            
        elif depth == 3:
            self.fc1 = nn.Linear(input_size, self.hidden)
            self.fc2 = nn.Linear(self.hidden, self.hidden)
            self.fc3 = nn.Linear(self.hidden, num_classes)
            self.S = nn.Sequential(self.fc1, self.ReLu, self.fc2, self.ReLu, self.fc3)
        elif depth == 4: 
            self.fc1 = nn.Linear(input_size, self.hidden)
            self.fc2 = nn.Linear(self.hidden, self.hidden)
            self.fc3 = nn.Linear(self.hidden, self.hidden)
            self.fc4 = nn.Linear(self.hidden, num_classes)
            self.S = nn.Sequential(self.fc1, self.ReLu, self.fc2, self.ReLu, self.fc3, self.ReLu, self.fc4)
        
    def forward(self, x):
        out = self.S(x)
        return out
    
