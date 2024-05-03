

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import torch.nn as nn
#import opencxr
import numpy as np
import SimpleITK as sitk
import lightning as L

import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# define lightning data module
class Node21Module(L.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.csv_file = "metadata_training.csv"
        self.transforms = transforms.Compose( 
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.456], std=[.226])])

    def setup(self, stage=None):
        # split dataset into train, validation, (and test?) sets
        self.train_dataset = Node21Dataset(root=self.data_dir,csv_file=self.csv_file,
                                           transform=get_transform(train=True),split="train")
        self.val_dataset = Node21Dataset(root=self.data_dir,csv_file=self.csv_file,
                                         transform=get_transform(train=False),split="validation")
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=7,
                                collate_fn=Node21Dataset.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=7,
                                collate_fn=Node21Dataset.collate_fn)
    
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



# define custom dataset
class Node21Dataset(Dataset):
    def __init__(self, root, csv_file, transform, split):
        self.root = root
        self.transform = transform
        self.data = pd.read_csv(os.path.join(root,csv_file))
        self.data = self.data[self.data["split"] == split] # train or val
        self.imgs = list(sorted(os.listdir(os.path.join(root,"images"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root,"images",str(self.imgs[idx]))
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = img.astype(np.float32)/np.max(img)
        
        img = Image.fromarray(img)

        nodule_data = self.data[self.data['img_name']==str(self.imgs[idx])]

        num_objs = len(nodule_data)
        boxes = []

        if nodule_data['label'].any()==1:
            for i in range(num_objs):
                x_min = nodule_data.iloc[i]['x']
                y_min = nodule_data.iloc[i]['y']
                x_max = x_min + nodule_data.iloc[i]['width']
                y_max = y_min + nodule_data.iloc[i]['height']
                boxes.append([x_min, y_min, x_max, y_max])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:,3] - boxes[:,1])*(boxes[:,2] - boxes[:,0])
            labels = torch.ones((num_objs,),dtype=torch.int64)
            
        else:
            boxes = [1, 1, 1024, 1024]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).unsqueeze(0)
            area = (boxes[:,3] - boxes[:,1])*(boxes[:,2] - boxes[:,0])
            labels = torch.zeros((1,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area

        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img,target
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = list(images)
        targets = list(targets)
        return images, targets
