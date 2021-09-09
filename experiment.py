from trainer import Trainer, config
import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterSampler
from dataset.image_dataset import get_train_val_test_loaders as get_image_loaders
from dataset.ehr_dataset import get_train_val_test_loaders as get_ehr_loaders
from model.ehr_nn import LogisticRegression
from model.image_cnn import CNN_PACS, CNN_aux, CNN_aux_end, CNN_autoencoder
import os
from tqdm import tqdm
from dataset.fused_dataset import get_fused_loader
from model.fused import LateFuse
import torch.nn as nn
import collections 

class Experiment(object):
    def __init__(self, optimizer, device, config_str, model_type, model_name, param_grid, save_best_num, savename, budget=1, repeat=1, name='tmp', save_every = None, eval_train = True, mixed = True, metadata = "", pretrain_file = "", seed = 0):
        self.eval_train = eval_train
        self.model_name = model_name
        self.model_type = model_type 
        self.budget = budget
        self.repeat = repeat # number of restarts with different random seeds
        self.param_grid = param_grid
        self.param_sampler = ParameterSampler(param_grid, n_iter=self.budget, random_state=0)
        

        self.config_str = config_str
        print("device", device)
        self.device = device 
        self.save_every = save_every 
        self.optimizer = optimizer 
        self.mixed = mixed
        self.save_best_num = save_best_num
        self.savename = savename
        self.metadata = metadata
        self.pretrain_file = pretrain_file
        self.va_loader = []
        self.tr_loader = []
        self.te_loader = []
        self.bias_te_loader = []
        self.loader_names = config(config_str + ".loader_names")
        self.get_biased_loader = config(config_str + ".bias_te")
        self.save_best_num  = save_best_num
        self.seed = seed
    def run(self):
        if (os.path.exists('{}/log/df_search.csv'.format(self.savename)) and self.repeat > 1):
            df_search = pd.read_csv('{}/log/df_search.csv'.format(self.savename))
        if (os.path.exists('{}/seed_{}/df_search.csv'.format(self.savename, self.seed)) and self.repeat == 1):
            df_search = pd.read_csv('{}/seed_{}/df_search.csv'.format(self.savename, self.seed))
        else:
            df_search = pd.DataFrame(columns=['best_score', 'best_iter', 'seed', 'savename'] + list(self.param_grid.keys()))
                  
        start_time = time.time()
        
        iterator = 0
        for run, params in tqdm(enumerate(self.param_sampler), desc='params' + str(iterator), leave = False):
            print(self.config_str, '\t', 'Run:', run, '/', self.budget)
            print("length:", len(df_search))        
#             print(params)
            for i in range(self.repeat):
#                 print("REOEAT", self.repeat, self.seed)
                if (self.repeat == 1):
                    seed = self.seed
                else:
                    seed = i 
                if not os.path.exists('{}/seed_{}/'.format(self.savename, seed)):
                    os.makedirs('{}/seed_{}/'.format(self.savename, seed))
            
                params_ordered = collections.OrderedDict()
                for k in sorted (params.keys()):
                    params_ordered[k] = params[k]
                savename = '{}/seed_{}/{}_checkpoint.pth.tar'.format(self.savename, seed, params_ordered)
#                 print("SAVENAME:", savename)
#                 if (not sum(df_search['savename']== savename)): 
                print("Savename", np.where((df_search['savename']== savename).values))
#                 if (np.where((df_search['savename']== savename).values)[0] != run):
#                     print(savename)
                if savename == "./checkpoint/image/michigan_ground_truth_classifier//seed_0/OrderedDict([('augmentation', ['rotate']), ('batch_size', 32), ('lr', 0.01), ('momentum', 0.8), ('weight_decay', 0.0001)])_checkpoint.pth.tar":
                    print(run)
                if (not sum(df_search['savename']== savename)): 
                    results = self._run_trial(seed, params)
                    if (self.save_best_num > 1):
                        for result in results:
                            df_search = df_search.append(result, ignore_index=True)
                    else:
                        df_search = df_search.append(results, ignore_index = True)
                else:
                    df_search = df_search.drop_duplicates(subset = ["savename"], keep = "first")

#                     print("Already ran this, moving onto next")
#                 print("savename:", self.savename)
#                 print("Saving at:", self.savename[0:self.savename.rfind("seed")])
                if self.repeat == 1:
                    df_search.to_csv('{}/seed_{}/df_search.csv'.format(self.savename[0:self.savename.rfind("seed")], self.seed), index=False)
                else:
                    df_search.to_csv('{}/log/df_search.csv'.format(self.savename[0:self.savename.rfind("seed")]), index=False)
    
            iterator += 1
        print('Took:', time.time() - start_time)
        return df_search
    
    def _run_trial(self, seed, params_unordered):
        print("Running trial:", seed)
        
        params = collections.OrderedDict()
        for i in sorted (params_unordered.keys()):
            params[i] = params_unordered[i]
                
        savename = '{}/seed_{}/{}_checkpoint.pth.tar'.format(self.savename, seed, params)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        print("Getting model")
        self.model, criterion, optimizer = self._get_model(seed, params)
        
        self.model = nn.DataParallel(self.model)
        
        
        print("Getting data")
        self._get_data_loaders(seed, params, "train")
        

        print("Initializing trainer")
        self.trainer = Trainer(seed,self.model, criterion, optimizer, params, self.loaders, self.loader_names, self.device, savename, self.config_str, save_best_num = self.save_best_num, save_every = self.save_every, eval_train = self.eval_train, mixed = self.mixed)
        
        print("Training")
        self.trainer.fit()
        
        print(self.trainer.best_iter, self.trainer.best_score)
        
        if (self.save_best_num > 1):
            print("best checkpoint to score", self.trainer.best_checkpoint_to_score)
            all_scores = [ {
            'best_score': self.trainer.best_checkpoint_to_score[checkpoint], 'best_iter': 0, 
            'savename': checkpoint, 'seed': seed,
            **params,
                } for checkpoint in self.trainer.best_checkpoint_to_score]
        else:
            best_score = self.trainer.best_score
            _best_iter = self.trainer.best_iter
            all_scores = self.trainer.scores
            all_losses = self.trainer.losses
            best_lower = self.trainer.best_lower
            best_upper = self.trainer.best_upper
            best_test = self.trainer.best_test            
            best_loss = self.trainer.best_loss
            all_data = collections.OrderedDict()

            for loader_name in self.loader_names:
                all_data[loader_name + "_score"] = best_test[loader_name]
                all_data[loader_name + "_loss"] = best_loss[loader_name]
                if (loader_name != "train"):
                    all_data[loader_name + "_best_lower"] = best_lower[loader_name]
                    all_data[loader_name + "_best_upper"] = best_upper[loader_name]
                
            print("all_data", all_data)
            print("params", params)
            print("Best score:", best_score, "best iter:", _best_iter)
            all_scores = {
            'best_score': best_score, 'best_iter': _best_iter, 
            'savename': savename, 'seed': seed, 
            **params, 
            **all_data,
        }


        print("deleting")
        del self.loaders
        print("done deleting")
        return all_scores
    
    def _get_data_loaders(self, seed_no, params, split = "train", labels = []):

        model_name = self.model_name
        model_type = self.model_type

        if (model_type == "image"):
            self.loaders,  _ = get_image_loaders(seed_no, self.config_str,params["batch_size"], params["augmentation"], config(self.config_str + '.num_classes'), self.get_biased_loader, data = self.metadata)
        elif (model_type == "ehr"):

            self.tr_loader, self.va_loader, self.te_loader = get_ehr_loaders(seed_no, self.config_str,params["batch_size"], config(self.config_str + ".num_classes"), True, 0, split == "train")
            self.loaders = [self.tr_loader, self.va_loader, self.te_loader]
        else:
            print("Getting data for fused model")
            
            print("getting ehr loaders")
            tr_loader_ehr, va_loader_ehr, te_loader_ehr = get_ehr_loaders(seed_no, self.config_str,params["batch_size"], config(self.config_str + ".num_classes"), True, 0, split == "train")

            augmentation = params['augmentation']
            print("getting image loaders")
            self.loaders, _ = get_image_loaders(seed_no, self.config_str,params["batch_size"], params["augmentation"],config(self.config_str + '.num_classes'), split == "train")

            if (split == "train"):
                print("getting all loaders")
                self.tr_loader = get_fused_loader(seed_no, self.loaders[0], tr_loader_ehr, params["batch_size"], "train")
                self.va_loader = get_fused_loader(seed_no, self.loaders[1], va_loader_ehr,params["batch_size"], "valid")
                self.te_loader = get_fused_loader(seed_no, self.loaders[2], te_loader_ehr,params["batch_size"], "test")

            else:
                self.tr_loader = []
                self.va_loader = []
                print("only getting test loader")
                self.te_loader = get_fused_loader(seed_no, self.loaders[2], te_loader_ehr,params["batch_size"], "test")
            
            self.loaders = [self.tr_loader, self.va_loader, self.te_loader]

        return self.loaders
    
    def _get_model(self, seed_no, params): 
        model_type = self.model_type 
        model_name = self.model_name

        if (model_type == "image"): 
                
            try: 
                tune_classifier = config(self.config_str + ".tune_classifier")
            except:
                tune_classifier = False 

            model = CNN_PACS(model_type, model_name, config(self.config_str + ".pretrain"), self.device, params, pretrain_file = self.pretrain_file).model.to(self.device)


            criterion = torch.nn.BCEWithLogitsLoss()
#             print("model", model)
            
            try:
                freeze_all_layers = config(self.config_str + ".freeze_all")
                
            except:
                freeze_all_layers = False 
            

            if (self.model_name == "bias_pretrain_ehr"):
                print("only tuning classifier")
                parameters = model._modules.get('module').model.classifier.parameters()    
            elif (freeze_all_layers):
                print("Freezing all layers")
                for param in model.features.parameters():
                    
                    param.requires_grad = False
                
                parameters = model.classifier.parameters()
#             elif freeze_block:
#                 num_blocks = config(self.config_str + ".num_blocks")
#                 if (num_blocks == 1 ):
#                     print("Tuning denseblock 4")
#                     parameters = list(model.features.denseblock4.parameters()) + list(model.features.norm5.parameters()) + list(model.classifier.parameters()) 
#                 elif num_blocks == 2:
#                     print("Tuning denseblock 4 and 3")
#                     parameters = list(model.features.denseblock4.parameters()) + list(model.features.norm5.parameters()) + list(model.classifier.parameters()) + list(model.features.denseblock3.parameters()) + list(model.features.transition3.parameters()) 
#                 elif num_blocks == 3:
#                     print("Tuning denseblock 4, 3, and 2")
#                     parameters = list(model.features.denseblock4.parameters()) + list(model.features.norm5.parameters()) + list(model.classifier.parameters()) + list(model.features.denseblock3.parameters()) + list(model.features.transition3.parameters()) + list(model.features.denseblock2.parameters()) + list(model.features.transition2.parameters())

            elif (freeze_beginning_layers):
                print("Freezing first denseblock of densenet")
                parameters = list(model._modules.get('module').features.transition2.parameters()) + list(model._modules.get('module').features.transition3.parameters()) + list(model._modules.get('module').features.denseblock2.parameters()) + list(model._modules.get('module').features.denseblock3.parameters()) + list(model._modules.get('module').features.denseblock4.parameters()) + list(model._modules.get('module').features.norm5.parameters()) + list(model._modules.get('module').classifier.parameters()) 
            elif (tune_last_denseblock):
                print("Tuning last denseblock")
                parameters = list(model._modules.get('module').features.denseblock4.parameters()) + list(model._modules.get('module').features.norm5.parameters()) + list(model._modules.get('module').classifier.parameters()) 
#             elif (tune_classifier):
#                 print("optimizing second classifier weights")
#                 parameters = list(model._modules.get('module').fc1.parameters())
            else:
                print("All parameters are being updated")
                parameters = model.parameters()
                parameters = list(model.features.parameters()) + list(model.classifier.parameters()) 
                parameters = list(model.features.denseblock4.parameters()) + list(model.features.norm5.parameters()) + list(model.classifier.parameters()) + list(model.features.denseblock3.parameters()) + list(model.features.transition3.parameters()) + list(model.features.denseblock2.parameters()) + list(model.features.transition2.parameters())+ list(model.features.denseblock1.parameters()) + list(model.features.transition1.parameters())+ list(model.features.conv0.parameters())+ list(model.features.norm0.parameters())
                    
            if (self.optimizer == "sgd"):
                optimizer = torch.optim.SGD(parameters,
                        lr=params["lr"], momentum=params["momentum"], weight_decay = params["weight_decay"])
                    
            else:
                print("using adam")
                optimizer = torch.optim.Adam(model.parameters(), lr = params["lr"])
#                 print(optimizer)
                
                
                
        elif (model_type == "ehr"):
            num_classes = config(self.config_str + ".num_classes")

            tr_loader_ehr, _,_ = get_ehr_loaders(seed_no, self.config_str,params["batch_size"], num_classes, True, 0, "train")
            in_channels = tr_loader_ehr.dataset[0][0].shape[1]
            print("in channels:", in_channels)
            print("params", params)
            model = nn.DataParallel(LogisticRegression(in_channels, num_classes, params["depth"])).to(self.device)


            if (self.mixed):
                model = model.half()
            
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], momentum = params["momentum"])



        else:
            tr_loader_ehr, _,_ = get_ehr_loaders(seed_no, self.config_str,int(params["batch_size"]), config(self.config_str + ".num_classes"), True, 0)

            in_channels_ehr = tr_loader_ehr.dataset[0][0].shape[1]

            
            del tr_loader_ehr # don't need this anymore, saves up memory 
            
            n_layers = params["depth"]
            print("Getting fused model")
            
            model = LateFuse(self.config_str, in_channels_ehr, n_layers = n_layers).to(self.device)     
#             model = model.cpu().to(self.device)
        
            print("Defining loss and optimizer")
            
#             check if model is penalized for changing too much from previous model 
            criterion = torch.nn.BCEWithLogitsLoss() 
    
            lr = params["lr"]
            weight_decay = params["weight_decay"]
            momentum = params["momentum"]
            
            if (model_name == "mimic_rad_image_ehr_init"):
                print("using initialized image and ehr models")
                print("freezing densenet")
                optimizer = torch.optim.SGD([{'params':model.densenet.classifier.parameters()}, 
                                             {'params' : model.fc1.parameters(), 'weight_decay' : weight_decay},
                                            {'params' : model.fc2.parameters(), 'weight_decay' : weight_decay}, 
                                            {'params' : model.fc3.parameters(), 'weight_decay' : weight_decay}], 
                                            lr = lr, momentum = momentum)                 
            else:
                try:
                    freeze_beginning_layers = config(self.config_str + ".freeze_denseblock")
                except:
                    freeze_beginning_layers = False

                try:
                    freeze_all_layers = config(self.config_str + ".freeze_all")
                except:
                    freeze_all_layers = False 

                try: 
                    tune_last_denseblock = config(self.config_str + ".tune_last_denseblock")
                except:
                    tune_last_denseblock = False     
                    
                if (freeze_all_layers):
                    print("Freezing all layers")
                    for param in model.densenet.features.parameters():
                        param.requires_grad = False
                    
                    if n_layers ==1:
                        print("Tuning fc1")
                        
                
                        parameters = model.fc1.parameters()
                    elif (n_layers == 2):
                        print("Tuning 2 layers: fc1, fc2")
                        parameters = list(model.fc1.parameters()) + list(model.fc2.parameters())
                    optimizer = torch.optim.SGD([{"params" : parameters, "weight_decay" : weight_decay}], lr = lr, momentum = momentum)
                else:
                    print("Tuning all Densenet parameters")
                    parameters = model.densenet.parameters()

                    if n_layers ==1:
                        print("Tuning 1 FC layer + densenet parameters")
                        optimizer = torch.optim.SGD([{'params': parameters}, 
                                                     {'params' : model.fc1.parameters(), 'weight_decay' : weight_decay}], 
                                                    lr = lr, momentum = momentum) 

                    elif (n_layers == 2):
                        print("Tuning 2 layers: fc1, fc2, and any chosen densenet parameters")
                        optimizer = torch.optim.SGD([{'params':parameters}, 
                                                     {'params' : model.fc1.parameters(), 'weight_decay' : weight_decay},
                                                    {'params' : model.fc2.parameters(), 'weight_decay' : weight_decay}], 
                                                    lr = lr, momentum = momentum) 


        return model, criterion, optimizer
    
    