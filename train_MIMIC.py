import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from utils_MIMIC import *

def main(num_epochs, gpu_ids, batch_size, pretrained):

    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]


    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    # device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
    device = torch.device("mps") 

    print('==> Preparing data...')
    # data transforms
    data_transforms = transforms.Compose(
            [transforms.Resize((2048, 2048), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
   
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir = "/scratch/mpbuyer/MIMIC-CXR-JPG-SAMPLE-1K"
    # print(current_dir)
    # print(os.getcwd())
    
    # current_dir = "MIMIC-CXR-JPG-SAMPLE-1K"
    train_dataset = MimicDataset(current_dir, data_transforms, "train")
    val_dataset = MimicDataset(current_dir, data_transforms, "validate")
    test_dataset = MimicDataset(current_dir, data_transforms, "test")

    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    print('==> Building and training model...')
    
    n_classes = 14*4 # 14 diseases, 4 categories

    if pretrained:
        model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT)
    else:
        model = models.swin_v2_s()
    num_features = model.head.in_features
    model.fc = nn.Linear(num_features, n_classes)
    model = model.to(device)

    # which loss?
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()

    if num_epochs == 0:
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model.to("cpu"))

    global iteration
    iteration = 0
    best_auc = 0
    
    for epoch in range(num_epochs):
        print("Epoch: ", epoch + 1, "-----------")        
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Train Loss: {train_loss:.4f}')
        
        scheduler.step()

        val_loss, val_acc, val_auc = test(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation ACC: {val_acc:.4f}')
        print(f'Validation AUC: {val_auc:.4f}')
            
        if val_auc > best_auc:
            best_epoch = epoch
            best_auc = val_auc
            best_model = deepcopy(model.to("cpu"))
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)
        
        print("--------------------------------")

    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(current_dir, 'best_model.pth')
    torch.save(state, path)

    test_loss, test_acc, test_auc = test(best_model, val_loader, criterion, optimizer, device)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test ACC: {test_acc:.4f}')
    print(f'Test AUC: {test_auc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Swin on MIMIC (or ChestXray-14)')

    parser.add_argument('--num_epochs',
                        default=10,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=16,
                        type=int)
    parser.add_argument('--pretrained',
                        help='Pretrain the model on ImageNet',
                        action="store_true")


    args = parser.parse_args()
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    pretrained = args.pretrained
    
    main(num_epochs, gpu_ids, batch_size, pretrained)