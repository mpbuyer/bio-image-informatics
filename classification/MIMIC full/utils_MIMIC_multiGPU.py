

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import torch.nn as nn
import numpy as np

import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import lightning as L
import torchmetrics

class SwinTransformer(L.LightningModule):
    def __init__(self, pretrained, num_epochs):
        super().__init__()
        # Multilabel multiclass classification: 14 labels, 4 classes
        # do I use ImageNet weights or not
        if pretrained:
            self.model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT) 
        else:
            self.model = models.swin_v2_b()
        # change classification head
        num_features = self.model.head.in_features
        self.model.head = nn.Linear(num_features, 56)

        # self.model = create_CheSS_model(56, "CheSS/chess.pth.tar")

        self.milestones = [0.5 * num_epochs, 0.75 * num_epochs] # for scheduler
        self.gamma = 0.1 # for scheduler
        self.criterion = nn.BCEWithLogitsLoss()
        self.auc_metric = torchmetrics.AUROC(task= 'multiclass', num_classes = 4, average = None)
        self.best_auc = 0 # validation auc will decide what model to save
        self.diseases = ['Atelectasis','Cardiomegaly','Consolidation','Edema',
                         'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
                         'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
                         'Pneumothorax','Support Devices']

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= self.milestones, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs = outputs.view(-1, 14, 4)
        targets = targets.float()
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss
    
    def on_validation_epoch_start(self):
        self.logits_list = []
        self.targets_list = []
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs = outputs.view(-1, 14, 4)
        targets = targets.float()
        loss = self.criterion(outputs, targets)
        self.log('validation_loss', loss, sync_dist=True)

        self.logits_list.append(outputs)
        self.targets_list.append(targets)

    def on_validation_epoch_end(self):
        # Concatenate the predictions and targets for the entire validation set
        logits = torch.cat(self.logits_list, dim=0)
        targets = torch.cat(self.targets_list, dim=0)

        logits = logits.view(logits.size(0)*logits.size(1), -1)
        targets = targets.view(targets.size(0)*targets.size(1), -1)

        targets = torch.argmax(targets,dim=-1)

        # Calculate AUC for the validation set
        auc = self.auc_metric(logits, targets)
        avg_auc = torch.mean(auc)
        
        # Log the AUC metric
        for i in range(4):
            self.log(f'val_auc_class{i}', auc[i], sync_dist=True)

        self.log('val_auc', avg_auc, sync_dist=True)

        self.logits_list = []
        self.targets_list = []

    def on_test_start(self):
        self.logits_list = []
        self.targets_list = []

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        outputs = outputs.view(-1, 14, 4)
        targets = targets.float()
        loss = self.criterion(outputs, targets)
        self.log('test_loss', loss)

        self.logits_list.append(outputs)
        self.targets_list.append(targets)

    def on_test_epoch_end(self):
        # Concatenate the predictions and targets for the entire test set
        logits = torch.cat(self.logits_list, dim=0)
        targets = torch.cat(self.targets_list, dim=0)

        logits = logits.view(logits.size(0)*logits.size(1), -1)
        targets = targets.view(targets.size(0)*targets.size(1), -1)

        targets = torch.argmax(targets,dim=-1)

        # Calculate AUC for each class
        auc = self.auc_metric(logits, targets)
        avg_auc = torch.mean(auc)
        # print(f"overall AUC: {avg_auc}")

        for i, disease in enumerate(self.diseases):
            indexes = torch.arange(logits.shape[0]) % 14 == i
            # Get AUC for each label
            logits_by_label = logits[indexes,:]
            targets_by_label = targets[indexes]
            label_auc = self.auc_metric(logits_by_label, targets_by_label)[0]
            # label_auc = torch.mean(label_auc)
            self.log(f'test_auc_{disease}', label_auc, sync_dist=True)
            # print(f"{disease} AUC: {label_auc}")
        
        # Log the AUC metric
        for i in range(4):
            self.log(f'test_auc_class{i}', auc[i], sync_dist=True)
        self.log('test_auc', avg_auc, sync_dist=True)





# define custom dataset
class MimicDataset(Dataset):
    def __init__(self, csv_path, transform, mode):
        self.transform = transform
        self.path = csv_path
        self.data = pd.read_csv(csv_path + "/mimic-cxr-2.0.0-all.csv")
        self.data = self.data[self.data["split"] == mode]
        self.diseases = ['Atelectasis','Cardiomegaly','Consolidation','Edema',
                         'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
                         'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
                         'Pneumothorax','Support Devices']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # find image_path
        subject_id = str(row['subject_id'])
        subject_folder = "p" + subject_id[0:2] + "/p" + subject_id
        study_id = str(row['study_id'])
        study_folder = "s" + study_id
        dicom_id = str(row['dicom_id'])
        image_file = dicom_id + ".jpg"

        image_path = os.path.join(self.path, "files",subject_folder,study_folder,image_file)
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(image_path).convert('L')

        target = []
        for disease in self.diseases:
            if row[disease] == 1.0:
                one_hot_vector = [1,0,0,0]
            elif row[disease] == 0.0:
                one_hot_vector = [0,1,0,0]
            elif row[disease] == -1.0:
                one_hot_vector = [0,0,1,0]
            else:
                one_hot_vector = [0,0,0,1]
            target.append(one_hot_vector)

        image = self.transform(image)
        target = torch.as_tensor(target)

        return image,target
    


# --------------------------------------------------------------------------------------------------
# From https://github.com/mi2rl/CheSS
# from CheSS.upstream.moco.resnet_ori import resnet50
# def create_CheSS_model(num_classes, weightsPath):
#     model = resnet50(num_classes=1000)

#     pretrained_model = weightsPath
#     if pretrained_model is not None:
#         if os.path.isfile(pretrained_model):
#             print("=> loading checkpoint '{}'".format(pretrained_model))
#             checkpoint = torch.load(pretrained_model, map_location="cpu")

#             # rename moco pre-trained keys
#             state_dict = checkpoint['state_dict']
#             for k in list(state_dict.keys()):
#                 # retain only encoder_q up to before the embedding layer
#                 if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
#                     # remove prefix
#                     state_dict[k[len("module.encoder_q."):]] = state_dict[k]
#                 # delete renamed or unused k
#                 del state_dict[k]

#             msg = model.load_state_dict(state_dict, strict=False)
#             assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

#             print("=> loaded pre-trained model '{}'".format(pretrained_model))
#         else:
#             print("=> no checkpoint found at '{}'".format(pretrained_model))

#         ##freeze all layers but the last fc
#         for name, param in model.named_parameters():
#             if name not in ['fc.weight', 'fc.bias']:
#                 param.requires_grad = False
            
#         model.fc = nn.Linear(2048, num_classes)
#     return model
