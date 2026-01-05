
import numpy as np
import re 
import glob
import os
#Imports

import os
import matplotlib.pyplot as plt
import torch
import argparse

from Modules.utils_general import *
from Modules.transforms import get_transforms

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
    PersistentDataset
)


def split_train_val_test(opt):

    """Split the dataset after having removed cases where the CT/CBCT registration didn't work."""
    file = pd.read_csv(f'{opt.splitdir}/Split{opt.case}.csv',header=0)
    #select right fold
    file = file[file["Fold"]==opt.fold]

    #get each patient list 
    train_list = file[file['Partition']=='train']['Number'].to_list()
    val_list = file[file['Partition']=='val']['Number'].to_list()
    test_list = file[file['Partition']=='test']['Number'].to_list()
    
    return train_list,val_list,test_list


def get_data(opt,train_list,val_list,test_list):

    
    df = pd.read_excel('/store/Datasets/ENE_HNC_2024_volatile/SwinUnetrClassification/EneGrade.xlsx',header=0,index_col=0)[f'{opt.case}']
    #Get list of pairs // Scan - Label
    train_images = []
    train_masks = []
    train_labels = []

    for pnb in train_list :

        ctpath = opt.data_dir+'imagesTr/'+f"{int(pnb):05d}" + "_0000.nii.gz"
        maskpath = opt.data_dir+'labelsTr/'+f"{int(pnb):05d}"+".nii.gz"
        label = torch.Tensor([df.loc[int(pnb)]]).float()

        train_images.append(ctpath)
        train_masks.append(maskpath)
        train_labels.append(label)

    train_files = [{"image": image_name, "label": label_name,"mask": mask} for image_name, label_name,mask in zip(train_images, train_labels,train_masks)]


    #Get list of pairs // Scan - Label
    val_images = []
    val_masks = []
    val_labels = []

    for pnb in val_list :

        ctpath = opt.data_dir+'imagesTr/'+f"{int(pnb):05d}" + "_0000.nii.gz"
        maskpath = opt.data_dir+'labelsTr/'+f"{int(pnb):05d}"+".nii.gz"
        label = torch.Tensor([df.loc[int(pnb)]]).float()

        val_images.append(ctpath)
        val_masks.append(maskpath)
        val_labels.append(label)

    val_files = [{"image": image_name, "label": label_name,"mask": mask} for image_name, label_name,mask in zip(val_images, val_labels,val_masks)]


    #Get list of pairs // Scan - Label
    test_images = []
    test_masks = []
    test_labels = []

    for pnb in test_list :

        ctpath = opt.data_dir+'imagesTr/'+f"{int(pnb):05d}" + "_0000.nii.gz"
        maskpath = opt.data_dir+'labelsTr/'+f"{int(pnb):05d}"+".nii.gz"
        label = torch.Tensor([df.loc[int(pnb)]]).float()

        test_images.append(ctpath)
        test_masks.append(maskpath)
        test_labels.append(label)

    test_files = [{"image": image_name, "label": label_name,"mask": mask} for image_name, label_name,mask in zip(test_images, test_labels,test_masks)]

    return train_files,val_files,test_files

def get_persistent_loaders(train_files,val_files,test_files,opt,persistent_cache):

     #Construct CacheDataset (More efficient datasets, check monai documentation)
    persistent_train_ds = PersistentDataset(data=train_files, 
                                    transform=get_transforms(opt,traintest='train'),
                                    cache_dir=persistent_cache
    )

    persistent_val_ds = PersistentDataset(
            data=val_files,
            transform=get_transforms(opt,traintest='infer'),
            cache_dir=persistent_cache
        )
    
    persistent_test_ds = PersistentDataset(
        data=test_files,
        transform=get_transforms(opt,traintest='infer'),
        cache_dir=persistent_cache
    )

    #Construct Dataloaders
    train_loader = DataLoader(
        persistent_train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=16,
    )

    val_loader = DataLoader(persistent_val_ds, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=4, 
                            pin_memory=True)
    
    test_loader = DataLoader(persistent_test_ds, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=4, 
                        pin_memory=True)
    
    
    
    return train_loader,val_loader,test_loader