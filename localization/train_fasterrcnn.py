import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import torch.utils.data as data
import torchvision.transforms as transforms
from faster_rcnn import *
from node21_dataset import *
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
        

def main(num_epochs, batch_size, pretrained):
    if num_epochs <= 0: # if someone mistypes...
        return

    print('==> Preparing data...')
    # data transforms
    # if pretrained:
    #     data_transforms = transforms.Compose( 
    #         [transforms.ToTensor(),
    #          transforms.Normalize(mean=[.456], std=[.226])])
    # else:
    #     data_transforms = transforms.Compose( 
    #         [transforms.ToTensor()])

    root_dir = "processed_data"
   

    # root_dir = "/scratch/mpbuyer/Node21/processed_data"
    data_module = Node21Module(data_dir=root_dir, batch_size=batch_size)

    print('==> Building and training model...')
    
    torch.set_float32_matmul_precision("high")
    
    model = Fasterrcnn(pretrained=pretrained,num_epochs=num_epochs)


    checkpoint_callback = ModelCheckpoint(monitor='val_AP_0.5',dirpath='checkpoints/',
                                          filename='Node21-{epoch:03d}-{val_AP:.5f}',
                                          save_top_k=2,mode='max')
    earlyStop_callback = EarlyStopping(monitor='val_AP_0.5',mode="max",patience=11)

    callbacks = [checkpoint_callback,earlyStop_callback]

    # device = torch.device("mps")

    logger = TensorBoardLogger("Node21_logs", name = "fasterrcnn")

    trainer = Trainer(min_epochs=1,max_epochs=num_epochs, logger=logger,
                      callbacks=callbacks)
    
    trainer.fit(model=model,datamodule=data_module)
    trainer.validate(model=model,datamodule=data_module,verbose=True,ckpt_path="best")






if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Faster r-cnn on Node21 dataset')

    parser.add_argument('--num_epochs',
                        default=10,
                        help='number of epochs of training',
                        type=int)
    parser.add_argument('--batch_size',
                        default=4,
                        type=int)
    parser.add_argument('--pretrained',
                        help='Use ImageNet weights',
                        action="store_true")



    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    pretrained = args.pretrained
    
    main(num_epochs, batch_size, pretrained)
