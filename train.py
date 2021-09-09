import torch
import numpy as np
import random
from model.image_cnn import CNN_PACS
from trainer import * 
# from utils import config
import utils
from sklearn import metrics
import torch.nn as nn
import os
from matplotlib import pyplot as plt
from IPython.display import clear_output
import importlib
from sklearn.model_selection import ParameterGrid
import collections
import yaml 
from experiment import Experiment 
# from apex import amp, fp16_utils
import re 
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_type', type=str, required=False, default = "image")
parser.add_argument('--gpu', type=str, required=False, default = "1,2,3,4,6,7")
parser.add_argument('--budget', type=int, required=False, default = 50)
parser.add_argument('--repeats', type=int, required=False, default = 3)
parser.add_argument('--save_every', type=int, required=False, default = None)
parser.add_argument('--save_best_num', type=int, required=False, default = 1)
parser.add_argument('--optimizer', type=str, required=False, default = "sgd")
parser.add_argument('--eval_train', type=int, required=False, default  = True)
parser.add_argument('--mixed', type=int, required=False, default = False)
parser.add_argument('--csv', type=str, required = False, default = "")
parser.add_argument('--pretrain_file', type=str, required = False, default = "")
parser.add_argument('--start', type=int, required = False, default = 0)
parser.add_argument('--stop', type=int, required = False, default = 0)
parser.add_argument('--seed', type=int, required = False, default = 0)

args = parser.parse_args()
pretrain_files = args.pretrain_file 
model_name = args.model_name
model_type = args.model_type
budget = args.budget
repeats = args.repeats
save_every = args.save_every 
optimizer = args.optimizer.lower()
eval_train = args.eval_train
mixed = args.mixed 
save_best_num = args.save_best_num 
csv = args.csv 
start = args.start
stop = args.stop
seed = args.seed

print("pretrain_file", pretrain_files)
weights = None
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu



def main():
    global pretrain_files

    print("model name:", model_name)
    config_str = model_type + "." + model_name
    with open('hyperparameters.yaml') as f:
        hyperparameters = yaml.load(f)    
    
    try:
        
        hyper = config(config_str + ".hyper")
        param_grid = hyperparameters[hyper]
    except:
        try:
            param_grid = hyperparameters[model_name]
        except:
            print("hyperparameters loaded for model type")
            param_grid = hyperparameters[model_type]
            print(param_grid)
            

    print("hyperparameters:", param_grid)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

    if (csv != ""):
        if (".csv" in csv):
            print("Using data from csv file:", csv)
            csv_files = [csv]
            stop_csv = 1
        else:
            print("Running multiple experiments, opening:", csv)
            csv_files = pickle.load(open(csv, "rb"))
            if (stop == 0):
                stop_csv = len(csv_files)
            else:
                stop_csv = stop 
        for csv_file in csv_files[start:stop_csv]:
            print("CSV:", csv_file)
            savename = config(config_str + ".checkpoint") + "/" + csv_file.replace(".csv","/")
            metadata = csv[0:csv.rfind("/")] + "/" + csv_file 
            print("metadata:", metadata)
            if not os.path.exists(savename):
                os.makedirs(savename)
            if not os.path.exists('{}/log/'.format(savename)):
                os.makedirs('{}/log/'.format(savename))
            
            exp = Experiment(optimizer,device, config_str, model_type, model_name, 
                             param_grid, save_best_num,savename, budget, repeats, 
                             eval_train = eval_train, save_every = save_every, mixed = mixed, metadata = metadata)

            print('EXPERIMENT:', savename)

            df_search = exp.run()
            df_search.to_csv('{}/log/df_search.csv'.format(savename), index=False)
    elif (pretrain_files != ""):
        pretrain_files = pickle.load(open(pretrain_files, "rb"))
        print("pretrain_files", type(pretrain_files))
        str_1 = pretrain_files['str_1']
        str_2 = pretrain_files["str_2"]
        print()
        pretrain_files = pretrain_files["pretrain_files"][0]
        print("running multiple experiments with different initializations")
        if (stop == 0):
            stop_pretrain_files = len(pretrain_files)
        else:
            stop_pretrain_files = stop 
        for pretrain_file in pretrain_files:
            print(pretrain_file, str_1, str_2)
            m = re.search(str_1 + '(.+?)' + str_2, pretrain_file)
            print(m)
            if m:
                found = m.group(1)
            print("found", found)
            savename = config(config_str + ".checkpoint") + "/pretrained_on_" + found + "/"
            if not os.path.exists(savename):
                os.makedirs(savename)
            if not os.path.exists('{}/log/'.format(savename)):
                os.makedirs('{}/log/'.format(savename))
            exp = Experiment(optimizer,device, config_str, model_type, model_name, 
                             param_grid, save_best_num,savename, budget, repeats, 
                             eval_train = eval_train, save_every = save_every, mixed = mixed, pretrain_file = pretrain_file, seed = seed)

            print('EXPERIMENT:', savename)

            df_search = exp.run()
            if repeats == 1:
                df_search.to_csv('{}/log/seed_{}/df_search.csv'.format(savename, seed), index=False)
            else:
                df_search.to_csv('{}/log/df_search.csv'.format(savename), index=False)
            
    else:
        savename = config(config_str + ".checkpoint")
  
        if not os.path.exists(savename):
            os.makedirs(savename)
        if not os.path.exists('{}/log/'.format(savename)):
            os.makedirs('{}/log/'.format(savename))
        exp = Experiment(optimizer,
            device, config_str, model_type, model_name, param_grid, save_best_num,savename, budget, repeats, eval_train = eval_train, save_every = save_every, mixed = mixed, seed = seed)

        print('EXPERIMENT:', config(config_str + ".checkpoint"))

        df_search = exp.run()
        print("length", len(df_search))
        if repeats == 1:
            df_search.to_csv('{}/seed_{}/df_search.csv'.format(savename, seed), index=False)
        else:
            df_search.to_csv('{}/log/df_search.csv'.format(savename), index=False)

#         df_search.to_csv('{}/log/df_search.csv'.format(savename), index=False)
if __name__ == '__main__':
    main()