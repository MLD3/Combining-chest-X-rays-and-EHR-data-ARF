# Overview
This is the code repository for the manuscript ["Combining chest X-rays and EHR data using machine learning to diagnose acute respiratory failure"](https://arxiv.org/pdf/2108.12530.pdf). 

## Directory structures 

- dataset/ contains data loaders

- model/ contains model loaders

- checkpoint/ is where model checkpoints are saved

## Download and Process the Data 
Follow directions to download the [MIMIC-IV](https://physionet.org/content/mimiciv/0.4/), [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/), and [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) datasets

Note: code to extract an AHRF cohort from MIMIC will be available soon. 

## Running the code

**Config file**

Pre-specified arguments can be set in config.json: 

Required arguments: 

- **csv_file**: Path to metadata file.  
- **checkpoint**: Path to file location where model checkpoints will be saved. 
- **labels**: column name in the metadata files of the classes. These can be separated by "|" (e.g., CHF|Pneumonia|COPD)
- **rotate degrees**: degrees of rotation to use for random rotation in image augmentation. 
- **disk**: if disk = 1, all images will be loaded into memory before training. Otherwise during training images will be fetched from disk. 
- **mask** : mask = 1 if masked loss will be used (i.e., if there are missing labels). All missing labels in the metadata file should be set to -1. 
- **early_stop**: early_stop = 1 if early stopping criteria will be used. Otherwise model will train to 3 epochs. 
- **pretrain**: Whether or not to use an initialization. If pretrain is "yes", then ImageNet initialization will be used unless a pretrain file is specified. Otherwise, pretrain should be "random" 
- **pretrain_file**: file path to pretrained model (i.e., pretrained model on MIMIC-CXR and CheXpert)
- **pretrain_classes**: number of labels pretrain model had 
- **freeze_all**: 1 or 0: whether or not to freeze all the layers but the classifier in the DenseNet
- **loader_names** : list of split names (i.e., \["train", "valid", "test"\]). You do not have to include "test". 


**Training a model** 

The following exmple code will train a model using train.py. Each run requires that a model_name and model_type be specificied. There are pre-specified in the config file along with other parameters (described in further detail below). Models will be saved in the directory chexpoint/model_type/model_name. 

Other non-required arguments are: 

## Arguments
- **gpu**: specify the gpu numbers to train on, default is 0 and 1. 
- **budget**: number of hyperparameter combinations to try. Default is 50. 
- **repeats**: number of seed initializations to try. Default is 3. 
- **save_every**: for pretraining on MIMIC-CXR and CheXpert. Number of iterations to complete before saving a checkpoint. Default is None and will save after every epoch. 
- **save_best_num**, for pretraining on MIMIC-CXR and CheXpert. Number of top checkpoints to save (based on best AUROC performance on the validation set). Default is 1. 
- **optimizer**: optimzier to use. Default is "sgd", but can also choose "adam" for pretraining on MIMIC-CXR and CheXpert. 

```bash
python train.py --model_type example_model_type --model_name example_model_name 
```
***Pretraining***

To train a model on MIMIC-CXR and CheXpert, you'll want to use the **save_every**, **save_best_num**, and **optimizer** arguments. This will train on an ImageNet initialized model: 

```bash
python train.py --model_type example_model_type --model_name example_model_name --save_every 4800 --save_best_num 10 --optimizer adam  
```
***Fine-tuning a model on the AHRF cohort:***

To train a DenseNet model after pretraining on either MIMIC-CXR/CheXpert, you'll need to specify the file location of the pretrained model in the config file, as well as the number of classes in the pretrained model. 
