import SimpleITK as sitk
import os
import sys
import numpy as np
import pandas as pd
import random

import torch
import matplotlib.pyplot as plt
import csv
import glob 
import datetime

from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

########## GENERAL UTILS ############

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def path2npa(nii_path):
    #path to numpy array
    im_sitk= sitk.ReadImage(nii_path)    
    im_npa = sitk.GetArrayFromImage(im_sitk)
    return im_npa
   

################ SAVING FUNCTIONS ################    
##################################################

##Parser stuff
    
def get_default_parser():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, help="Config file path")
    return parser


def setup_config(path=None):
    
    if path==None:
        path = "config.yaml"
        print("loaded vanilla config")
    config_path = path
    config = OmegaConf.load(config_path)

    return config

##Logger name stuff

def create_profile(args):

    modelname = args.ModelName +'_'+datetime.datetime.now().strftime('%m_%d') 
    
    return modelname

def get_progressbar():

    progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
        metrics_text_delimiter="\n",
        metrics_format=".3e",
    )
)
    return progress_bar

def get_class_weights(opt):

    pd_label =  pd.read_excel('/store/Datasets/ENE_HNC_2024_volatile/SwinUnetrClassification/EneGrade.xlsx',header=0)
    Len0 =len(pd_label[pd_label[f'{opt.case}'] == 0])
    Len1 =len(pd_label[pd_label[f'{opt.case}'] == 1])
    ratio = Len0/Len1

    return torch.Tensor([ratio])


def summary_cv(args):
    
    lines = pd.read_csv(f'/store/Datasets/ENE_HNC_2024_volatile/SwinUnetrClassification/results/{args.case}.csv',header=0)
    lines = lines[['ACC','PRECISION','RECALL','SPECIFICITY','AUC']].tail(5)

    mean=[]
    std = []
    for key in lines.keys():
        column = lines[key].to_list()
        typecheck = [float(element) for element in column]
        mean.append(np.round(np.mean(typecheck),3))
        std.append(np.round(np.std(typecheck),3))
    
    f = open(f'/store/Datasets/ENE_HNC_2024_volatile/SwinUnetrClassification/results/{args.case}.csv', 'a')
    writer = csv.writer(f)

    row_scores=['Average','----','----','----']
    row_scores+= mean
    writer.writerow(row_scores) 

    row_scores=['Std','----','----','----']
    row_scores+= std
    writer.writerow(row_scores) 

    f.close()