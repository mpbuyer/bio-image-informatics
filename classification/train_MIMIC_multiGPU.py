import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import torch.utils.data as data
import torchvision.transforms as transforms
from utils_MIMIC_multiGPU import *
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
        

def main(num_epochs, batch_size, pretrained):
    if num_epochs <= 0: # if someone mistypes...
        return

    print('==> Preparing data...')
    # data transforms
    data_transforms = transforms.Compose(
            [transforms.Resize((512,512), interpolation=transforms.InterpolationMode.BILINEAR), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
   
    current_dir = "/scratch/mpbuyer/MIMIC_JPG/physionet.org/files/mimic-cxr-jpg/2.0.0"
    # current_dir = "."

    train_dataset = MimicDataset(current_dir, data_transforms, "train")
    val_dataset = MimicDataset(current_dir, data_transforms, "validate")
    test_dataset = MimicDataset(current_dir, data_transforms, "test")

    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=7)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=7)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=7)

    print('==> Building and training model...')
    torch.set_float32_matmul_precision("high")
    
    model = SwinTransformer(pretrained=pretrained,num_epochs=num_epochs)
    checkpoint_callback = ModelCheckpoint(monitor='val_auc',dirpath='checkpoints/',
                                          filename='CheSS-{epoch:02d}-{val_auc:.3f}',
                                          save_top_k=2,mode='max')
    earlyStop_callback = EarlyStopping(monitor='val_auc',mode="max",patience=7)

    callbacks = [checkpoint_callback,earlyStop_callback]


    trainer = Trainer(min_epochs=1,max_epochs=num_epochs,callbacks=callbacks,logger=True)
    
    trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders=val_loader)
    trainer.test(model=model,dataloaders=test_loader,verbose=True,ckpt_path='best') 




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Swin Transformer on MIMIC dataset')

    parser.add_argument('--num_epochs',
                        default=10,
                        help='number of epochs of training',
                        type=int)
    parser.add_argument('--batch_size',
                        default=16,
                        type=int)
    parser.add_argument('--pretrained',
                        help='Use ImageNet weights',
                        action="store_true")


    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    pretrained = args.pretrained
    
    main(num_epochs, batch_size, pretrained)
