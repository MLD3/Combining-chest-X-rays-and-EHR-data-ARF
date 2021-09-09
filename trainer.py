# from utils import config
import itertools
import os
import torch
import numpy as np 
import utils 
from sklearn import metrics
import random
import pickle 
from tqdm import tqdm 
from matplotlib import pyplot as plt
import torch.nn as nn
import shutil 
from sklearn.utils import resample  

# from apex import amp 

class Trainer(object):
    def __init__(self, seed, model, criterion, optimizer, params, loaders, loader_names,  device, checkpoint, config_str, batch_size=None,save_best_num = 1,save_every=None,cuda=True, verbose = True, eval_train = False, mixed = True, savename = ""):
        self.num_best_saved = 0

        self.best_checkpoint_to_score = {}
        self.save_best_num = save_best_num
        self.mixed = mixed
        self.eval_train = True
        self.verbose = verbose
        self.seed = seed
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.n_classes = config(config_str + ".num_classes")
        self.loaders = loaders
        self.loader_names = loader_names
        self.params = params
        self.n_iters = 0
        self.epoch = 0
        self.best_iter = 0
        self.config_str = config_str
        self.save_every = save_every 
        self.checkpoint = checkpoint
        split = self.checkpoint.split('/')
        split[-1] = 'best_' + split[-1]
        self.best_checkpoint = '/'.join(split)
        try: 
            self.predict_ehr = config(self.config_str + ".predict_ehr")
            self.loss_coef = self.params['loss']
        except:
            self.predict_ehr = False
            
        
#         try:
#             self.penalize_classifier = config(self.config_str + ".penalize_classifier")
#         except:
#             self.penalize_classifiexr = False 
            
        try:
            self.autoencoder = config(self.config_str + ".autoencoder")
        except:
            self.autoencoder = False 
            
            
        
                
                
#         self.training_plot_axes = self._init_training_plot()
        self.stop = False
        
        
        self.training_plot_file = '{}_training_plot.png'.format(self.checkpoint[0:self.checkpoint.rfind("_")])
                                                                               
        
        self._iter = 0
        
        self.cuda = cuda and torch.cuda.is_available()

        self.num_train = 500 
        self.eval_all = True

        
        self.loader_pts = {}
        
        for loader_name, loader in list(zip(self.loader_names, self.loaders)):
            self.loader_pts[loader_name] = [] if loader_name != "train" else False 
            pos = []
            if (loader_name != "train"):
                for X, y, pt_id in loader:
                    for idx,p in enumerate(pt_id.numpy()):
                        self.loader_pts[loader_name].append(p)
        print("Finding checkpint:", self.checkpoint)
        if (os.path.exists(self.checkpoint)):
            print("loading latest checkpoint")
            self._load_best()

                
                
        else:
            print("resetting logs")
            self.reset_logs()
        
        
        print('Number of float-valued parameters:', count_parameters(self.model))

        

    def _eval_all(self):
        for loader_name, loader in list(zip(self.loader_names, self.loaders)):
#             if (loader_name == "valid" or (loader_name == "train" and self.eval_train)):
            if (loader_name == "valid"):
                self.latest_losses[loader_name], self.latest_scores[loader_name] = self._eval(loader, self.loader_pts[loader_name])
                self.losses[loader_name].append(self.latest_losses[loader_name])
                self.scores[loader_name].append(self.latest_scores[loader_name])

            else: 
                self.losses[loader_name].append(0)
                self.scores[loader_name].append(0) 

    def _print_status(self, print_all = True):
        if (print_all):
            for loader_name in self.loader_names:
                print(loader_name, ", loss: ", self.latest_losses[loader_name], ", score: ", self.latest_scores[loader_name])
        else:
                print("valid", ", loss: ", self.latest_losses["valid"], ", score: ", self.latest_scores["valid"])

    def reset_logs(self):
        
        self.best_iter = 0

        self.losses = {}
        self.scores = {}
        
        self.latest_losses = {}
        self.latest_scores = {}
        for loader_name in self.loader_names:
            self.losses[loader_name] = []
            self.scores[loader_name] = []
            self.latest_losses[loader_name] = []
            self.latest_scores[loader_name] = []
              
        self._eval_all()
        
        self.best_loss, self.best_score = self._eval(self.loaders[1], self.loader_pts["valid"])
        print("best score:, reset logs", self.best_score)
        self._save(self.best_score)
        

    def fit(self):
        self.stop = self._check_early_stopping() 
        if (self.stop):                 
            print("Saving final!")
            self._save_final()
            return 
            
        while not self.stop:
            if (self.verbose):
                print("Epoch:", self.epoch)
            for i, (X, y, pt_id) in tqdm(enumerate(self.loaders[0]), desc='epoch '+str(self.epoch) + ', iter ' + str(self._iter), leave=False):
                loss, pred = self._train_batch(X, y)
                if self.save_every and self._iter % self.save_every == 0:
                    print("Iteration:", self._iter, "saving")
                    self._eval_all()
                    if (self.autoencoder):
                        self._save(self.latest_losses["valid"])
                    else:
                        self._save(self.latest_scores["valid"])
                    self.stop = self._check_early_stopping()
                self._iter += 1
               
            self._eval_all()
            self.epoch += 1 
            self._print_status()
    
            if (self.autoencoder):
                self._save(self.latest_losses["valid"])
            else:
                self._save(self.latest_scores["valid"])

            self._plot_losses()
            self.stop = self._check_early_stopping()
            
            if (self.stop):
                self._save_final()
                
                break
            
                

    def _train_batch(self, X, y):
        self.model.train()
        if (type(X) is list):
            for i in range(len(X)):
                X[i] = X[i].to(self.device)
                if (torch.isnan(X[i]).any()):
                    print(X[i])
                y = y.to(self.device)
        else:  
            X,y = X.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(X)

            
        if (config(self.config_str + ".mask")):
            output[y<0] = 0
            y[y<0] = 0
        
        loss = self.criterion(torch.squeeze(output), torch.squeeze(y).float())

#         if (self.penalize_classifier):
#             loss += self.params['lambda'] *  torch.norm(self.model._modules.get('module').classifier.weight - self.model._modules.get('module').old_weights)

        loss.backward()
        self.optimizer.step()
        
        if (self.predict_ehr):
            return loss, (output[0].data.detach().cpu(), y.data.detach().cpu())
        
        return loss, (output.data.detach().cpu(), y.data.detach().cpu())
  
    def get_roc_CI(self, y_true, y_score):
#         pickle.dump(y_true, open("y_true.p", "wb"))
#         pickle.dump(y_score, open("y_score.p", "wb"))
        lower = []
        upper = []
        if (self.n_classes > 1):
            for i in range(self.n_classes):
                true = y_true[:,i]
                score = y_score[:,i]
                roc_curves, auc_scores, aupr_scores = [], [], []
                if (config(self.config_str + ".mask")):
                    score = score[true > -1]
                    true = true[true > -1]
                try:
                    for j in range(1000):
                        yte_true_b, yte_pred_b = resample(true, score, replace=True, random_state=j)


                        roc_curve = metrics.roc_curve(yte_true_b, yte_pred_b)
                        auc_score = metrics.roc_auc_score(yte_true_b, yte_pred_b)
                        aupr_score = metrics.auc(*metrics.precision_recall_curve(yte_true_b, yte_pred_b)[1::-1])

                        roc_curves.append(roc_curve)
                        auc_scores.append(auc_score)
                        aupr_scores.append(aupr_score)
                    lower.append(np.percentile(auc_scores, 2.5))
                    upper.append(np.percentile(auc_scores, 97.5))
                
                except:
                    print("only 1 class present")
                    lower.append(0.5)
                    upper.append(0.5)
           
                
        else:
            roc_curves, auc_scores, aupr_scores = [], [], []
            if (config(self.config_str + ".mask")):
                y_score = y_score[y_true > -1]
                y_true = y_true[y_true > -1]
            for j in range(1000):
                yte_true_b, yte_pred_b = resample(y_true, y_score, replace=True, random_state=j)

                try:
#                     roc_curve = metrics.roc_curve(yte_true_b, yte_pred_b)
                    auc_score = metrics.roc_auc_score(yte_true_b, yte_pred_b)
                    aupr_score = metrics.auc(*metrics.precision_recall_curve(yte_true_b, yte_pred_b)[1::-1])

                    auc_scores.append(auc_score)
                    aupr_scores.append(aupr_score)
                except:
                    auc_scores.append(0.5)
                    aupr_scores.append(0.5)
                    continue  
#             roc_curves.append(roc_curve)
            
            lower = np.percentile(auc_scores, 2.5)
            upper = np.percentile(auc_scores, 97.5)
        
        return lower, upper 

    def _eval(self, data_loader, pts, train = False, get_CI = False):
        self.model.eval()
        running_loss = []
        running_pred = []
        num_processed = 0
        with torch.no_grad():
            for X, y, pt_id in data_loader:
                if (train and num_processed > self.num_train and not self.eval_all):
                    break
                if (self.mixed):
                    if (type(X) is list):
                        for i in range(len(X)):
                            X[i] = X[i].contiguous().to(self.device).half()
                            y = y.contiguous().to(self.device).half()
                    else:  
                        X,y = X.to(self.device).half(), y.to(self.device).half()
                    
                    output = self.model(X)

                else:
                    if (type(X) is list):
                        for i in range(len(X)) :
                            X[i] = X[i].contiguous().to(self.device)
                            y = y.contiguous().to(self.device)
                    else:  
                        X,y = X.to(self.device), y.to(self.device)
                
                    output = self.model(X).float()

            
                
                if (self.predict_ehr):
                    predicted = predictions(output[0].data)
                else:
                    predicted = predictions(output.data)
                running_pred.append((predicted.data.detach().cpu().numpy(), y.data.detach().cpu().numpy(), pt_id.data.detach().cpu().numpy()))
                
                 #         mask loss 
                if (config(self.config_str + ".mask")):
                    output[y<0] = 0
                    y[y<0] = 0
                    
                if (self.autoencoder):

                    loss = self.criterion(torch.squeeze(output), X)
                else:
                    loss = self.criterion(torch.squeeze(output), torch.squeeze(y).float())
#                     if (self.penalize_classifier):
#                         loss += self.params['lambda'] *  torch.norm(self.model._modules.get('module').classifier.weight - self.model._modules.get('module').old_weights)
                        
                        
                running_loss.append(loss.data.detach().cpu())
                
                num_processed += len(y)

#         print("Num processed:", num_processed)
#         print("pts:", len(pts))
        if self.autoencoder:
            print("not getting auroc")
            return np.mean(running_loss), 0
#         print("running prediction", running_pred)
        return np.mean(running_loss), self._get_score(running_pred, pts, get_CI)

    def _check_early_stopping(self):
        #     check for early stopping 
        print("cheking early stopping epoch:", self.epoch)
        if (not config(self.config_str + ".early_stop")):
            return self.epoch > 3
        stop = False 
        if(len(self.losses['valid']) > 10):
            min_epoch = np.argmin(self.losses['valid'])
            if  min_epoch < len(self.losses['valid']) - 5 or np.abs(self.losses['valid'][-1] - self.losses['valid'][-2]) < 0.001:
                stop = True  
            
        return stop 
   
    def _get_score(self, running_pred, pt_ids, get_CI = False):
        y_pred, y_true, pt_ids = zip(*running_pred)
        

        
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        pt_ids = np.concatenate(pt_ids)

        assert(len(y_pred) == len(y_true) == len(pt_ids))
#         print(list(zip(y_true, pt_ids)))
        
        unique_pt_ids = np.unique(pt_ids)
        unique_predictions = []
        unique_truth_values = []
        for pt_id in unique_pt_ids:
            indices = np.where(pt_ids == pt_id)[0]
            if(len(y_pred[indices]) == 1):
                unique_predictions.append(y_pred[indices][0])
                unique_truth_values.append(y_true[indices][0])
            else:
                unique_predictions.append(np.average(y_pred[indices], axis = 0))
#                 print(np.average(y_true[indices], axis = 0))
#                 print((np.average(y_true[indices], axis = 0) == 0.5).any())
                if ((np.average(y_true[indices], axis = 0) == -0.5).any()):
                    print(pt_id, y_true[indices])
                unique_truth_values.append(np.average(y_true[indices], axis = 0))
            
        unique_predictions = np.squeeze(np.array(unique_predictions))

        unique_truth_values = np.squeeze(np.array(unique_truth_values))

        
#         mask loss 
        
        

        score = []
        for n in range(self.n_classes):
            if (self.n_classes == 1):
                unique_truth = unique_truth_values
                unique_pred = unique_predictions 
        
            else:
                unique_truth = unique_truth_values[:,n]
                unique_pred = unique_predictions[:,n] 
            try:
                if (config(self.config_str + ".mask")):
                    predictions = unique_pred[unique_truth > -1]
                    truths = unique_truth[unique_truth > -1]
#                     for idx, i in truths:
#                         if (i != 0 and i != 1):
#                             print("here", i, "pt_id", unique_pt_ids[idx])
                    score.append(metrics.roc_auc_score(truths, predictions))
                else:
#                     print(unique_pred)
#                     print(unique_truth)
                    score.append(metrics.roc_auc_score(unique_truth, unique_pred))
            except:
                print("1 class psresent in class", n)
                score.append(0.5)
        print("Calculated ROC for:", len(unique_predictions), "patients")

        assert(len(unique_predictions) == len(unique_pt_ids))
        
        
        if (get_CI):
            lower, upper = self.get_roc_CI(unique_truth_values, unique_predictions)
            return score, lower, upper 
        return score
    
    def _save_final(self):
        
#         load best checkpoint 
        print("finished training, loading best checkpoint for test set")
        if len(self.best_lower.keys()):
            print("already tested, returning")
            return 
        self._load_best(best = True)
        


#             get test set ROC_CI 
        self.best_lower = {}
        self.best_upper = {}
        self.best_test = {}
        self.best_loss = {}
        for loader_name in self.loader_names:
            if loader_name != "train" or (loader_name == "train" and self.eval_train):
                res = self._eval(self.loaders[self.loader_names.index(loader_name)], self.loader_pts[loader_name], train = loader_name == "train", get_CI = True)
                self.best_loss[loader_name], (self.best_test[loader_name], self.best_lower[loader_name], self.best_upper[loader_name]) = res 

#         print("best loss:", self.best_loss["valid"])
#         print("losses:", self.losses["valid"])
#         print("best iter:", self.best_iter)
#         print(self.losses["valid"][self.best_iter])
#         make sure valid score and loss is same as recorded in array 
#         print("AUROC:", self.scores["valid"])
#         print("loss", self.losses["valid"])
#         print(self.best_iter)
#         print(self.best_test["valid"], self.scores["valid"][self.best_iter])
#         print(self.best_loss["valid"], self.losses["valid"][self.best_iter])

#         if (self.save_best_num == 1):
#             assert(self.best_test["valid"] == self.scores["valid"][self.best_iter])
#             assert(self.best_loss["valid"] == self.losses["valid"][self.best_iter])

    
    def _save(self, new_score):
        if (self.save_best_num > 1):
            highest_best_score = 0
#             check if we have 10 anyways 
            if (len(self.best_checkpoint_to_score) == self.save_best_num):
                sorted_checkpoints = sorted(self.best_checkpoint_to_score.items(), key = 
                             lambda kv:(np.mean(kv[1]), kv[0]))
    
                if (self.autoencoder):
                    lowest_best_score = sorted_checkpoints[-1][1]
                    lowest_best_checkpoint = sorted_checkpoints[-1][0]
                    highest_best_score = sorted_checkpoints[0][1]
                    highest_best_checkpoint = sorted_checkpoints[0][0]
                else:
                    lowest_best_score = sorted_checkpoints[0][1]
                    lowest_best_checkpoint = sorted_checkpoints[0][0]
                    highest_best_score = sorted_checkpoints[-1][1]
                    highest_best_checkpoint = sorted_checkpoints[-1][0]
                if not self.autoencoder:
                    is_best = bool(np.mean(new_score) >= np.mean(lowest_best_score))
                else:
                    is_best = bool(np.mean(new_score) <= np.mean(lowest_best_score))

            else:
                is_best = True 
            print("Set of top best before:")

            for key in self.best_checkpoint_to_score:
                print(key, '->', np.mean(self.best_checkpoint_to_score[key]))
        else:
            if (self.autoencoder):
                is_best = bool(np.mean(new_score) <= np.mean(self.best_score))
            else:
                is_best = bool(np.mean(new_score) >= np.mean(self.best_score))
            print(new_score, self.best_score, is_best)
        if is_best:
            self.best_score = new_score
            
#             avoids duplicate maxes 
            if self.epoch and (self.save_best_num == 1):
                occurences = np.where(np.array(self.scores['valid']) == max(self.scores['valid']))[0][-1]
                self.best_iter = occurences
            else:
                self.best_iter = self._iter
            split = self.best_checkpoint.find("checkpoint.")
            new_best_checkpoint = self.best_checkpoint[0:split] + "_" + str(self.num_best_saved) + "_" + self.best_checkpoint[split:]
            
            
        
#         if is_best:
#             self.best_score = new_score
            
# #             avoids duplicate maxes 
#             occurences = np.where(np.array(self.scores['valid']) == max(self.scores['valid']))[0][-1]
#             assert(self.epoch == occurences)
#             self.best_iter = occurences

#         print("Epoch:", self.epoch)
#         print("iteR:", self._iter)
 
        self.best_lower = {}
        self.best_upper = {}
        self.best_test = {}
        state = {
            'best_iter' : self.best_iter,
            'best_score' : self.best_score, 
            'best_loss' : self.best_loss, 
            '_epoch': self.epoch,
            '_iter' : self._iter,
            'batch_size': self.batch_size,
            'state_dict': self.model.state_dict(),
            'arch': str(type(self.model)),
            'optimizer': self.optimizer.state_dict(),
            'losses' : self.losses, 
            'scores' : self.scores, 
            'latest_losses' : self.latest_losses, 
            'latest_scores' : self.latest_scores, 
            'params' : self.params,
            'stop' : self.stop, 
            'best_lower': self.best_lower, 
            'best_upper': self.best_upper, 
            'best_test': self.best_test,
            'best_checkpoint_to_score' : self.best_checkpoint_to_score
        }
        torch.save(state, self.checkpoint)
        if is_best: 
            if (self.save_best_num == 1):
                shutil.copyfile(self.checkpoint, self.best_checkpoint)
            else:
                self.best_checkpoint_to_score[new_best_checkpoint]= new_score
                if (self.autoencoder):
                    if (np.mean(new_score) <= np.mean(highest_best_score)):
                        self.best_checkpoint = new_best_checkpoint
                else:
                    if (np.mean(new_score) >= np.mean(highest_best_score)):
                        self.best_checkpoint = new_best_checkpoint
                if (len(self.best_checkpoint_to_score) > self.save_best_num):
                    try:
                        os.remove(lowest_best_checkpoint)

                        del self.best_checkpoint_to_score[lowest_best_checkpoint]
                    except: 
                        pass
                    print("Set of top best after:")
                    for key in self.best_checkpoint_to_score:
                        print(key, '->', np.mean(self.best_checkpoint_to_score[key]))
                
                torch.save(state, self.checkpoint)
                print("Saving new best at: ", new_best_checkpoint)
                print("With score:", new_score)
                self.num_best_saved += 1
                shutil.copyfile(self.checkpoint, new_best_checkpoint)
        torch.save(state, self.checkpoint)
                         
                        
#         torch.save(state, self.checkpoint)
#         if is_best:
#             shutil.copyfile(self.checkpoint, self.best_checkpoint)

    def _load_best(self, best = False):
        if (best):
            if (self.save_best_num > 1):

                    
                sorted_checkpoints = sorted(self.best_checkpoint_to_score.items(), key = 
                             lambda kv:(np.mean(kv[1]), kv[0]))
                print("best checkpoint to score before:", self.best_checkpoint_to_score)
                highest_best_score = sorted_checkpoints[-1][1]
                highest_best_checkpoint = sorted_checkpoints[-1][0]
                print("best checkpoint loaded:", highest_best_checkpoint)
                checkpoint = torch.load(highest_best_checkpoint)
                
                
            else:
                print("loading", self.best_checkpoint)
                if torch.cuda.is_available():
                    checkpoint = torch.load(self.best_checkpoint)
#             need full list of best checkpoints 
                    self.best_checkpoint_to_score = torch.load(self.checkpoint)["best_checkpoint_to_score"]
                else:
                    print("loading best checkpoint on cpu")
                    checkpoint = torch.load(self.best_checkpoint, map_location = "cpu")
                
#             need full list of best checkpoints 
                    self.best_checkpoint_to_score = torch.load(self.checkpoint, map_location = "cpu")["best_checkpoint_to_score"]

#             print( checkpoint['scores'])
#             print(checkpoint['losses'])
#             print(checkpoint['best_iter'])
        else:
            if torch.cuda.is_available():
                checkpoint = torch.load(self.checkpoint)
            else:
                print("loading best checkpoint on cpu")
                checkpoint = torch.load(self.checkpoint, map_location = "cpu")
            self.best_checkpoint_to_score = checkpoint["best_checkpoint_to_score"]
        
        self.epoch = checkpoint["_epoch"]
        self._iter = checkpoint["_iter"]
        self.batch_size = checkpoint["batch_size"]
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.latest_losses = checkpoint['latest_losses']
        self.losses = checkpoint['losses']
        
        self.latest_scores = checkpoint['latest_scores']
        self.scores = checkpoint['scores']
        
        self.best_lower = checkpoint['best_lower']
        self.best_upper = checkpoint['best_upper']
        self.best_test = checkpoint['best_test']

        self.params = checkpoint['params']
        self.stop = checkpoint['stop']
        
            
        if (not len(self.scores['valid'])):
            self._eval_all()

        self.best_iter = checkpoint['best_iter']
        self.best_loss = checkpoint['best_loss']
        self.best_score = checkpoint['best_score']
#         print("new best, load bes:", self.best_score)
#         print(len(self.scores["valid"]), self.epoch)
        if (self.save_best_num == 1):
            assert(self.epoch + 1 == len(self.scores["valid"]))
#     def _load(self):
#         checkpoint = torch.load(self.checkpoint)
#         self.epoch = checkpoint["_epoch"]
#         self._iter = checkpoint["_iter"]
#         self.batch_size = checkpoint["batch_size"]
#         self.model.load_state_dict(checkpoint['state_dict'])
#         self.optimizer.load_state_dict(checkpoint["optimizer"])
        
#         self.latest_losses = checkpoint['latest_losses']
#         self.losses = checkpoint['losses']
        
#         self.latest_scores = checkpoint['latest_scores']
#         self.scores = checkpoint['scores']

#         self.params = checkpoint['params']
#         self.stop = checkpoint['stop']
        
        
    def _init_training_plot(self):
        """
        Runs the setup for an interactive matplotlib graph that logs the loss and
        accuracy
        """
        fig, axes = plt.subplots(1,1, figsize=(10,5))
        plt.suptitle(config(self.config_str+ ".plot_title") + ' Training')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.legend(['Validation', 'Train'])
#         plt.text(.5, .05, param_str, ha='center')
        return axes
    
    def _plot_losses(self):
        
        self.training_plot_axes = self._init_training_plot()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        for idx, loader_name in enumerate(self.loader_names):
            self.training_plot_axes.plot(range(self.epoch - len(self.losses[loader_name]) + 1, self.epoch + 1), 
                                        self.losses[loader_name], linestyle = '--', marker = 'o', color = colors[idx])
        plt.savefig(self.training_plot_file, dpi=200)
        plt.close()
        
    

def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predictions(logits):
    """
    Given the network output, determines the predicted class index

    Returns:
        the predicted class output as a PyTorch Tensor
    """
    sig = torch.nn.Sigmoid()
    return sig(logits)
    #

    
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
