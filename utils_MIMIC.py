

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score


# Make distributed processes


# define custom dataset
class MimicDataset(Dataset):
    def __init__(self, csv_path, transform, mode):
        self.transform = transform
        self.path = csv_path
        self.data = pd.read_csv(csv_path + "/mimic_small.csv")
        self.data = self.data[self.data["split"] == mode]
        self.diseases = ['Atelectasis','Cardiomegaly','Consolidation','Edema',
                         'Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity',
                         'No Finding','Pleural Effusion','Pleural Other','Pneumonia',
                         'Pneumothorax','Support Devices']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = str(row['file_path'])
        # print(self.path)
        # print(image_path)
        # print(os.path.join(self.path, image_path))
        image = Image.open(self.path + image_path).convert('RGB')
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


def train(model, train_loader, criterion, optimizer, device):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = targets.to(torch.float32).to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        iteration += 1

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, data_loader, criterion, device):

    model.eval()
    
    total_loss = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
            total_loss.append(loss.item())

            batch_size,_,_ = outputs.size()

            outputs = torch.reshape(outputs, (batch_size*14,4))
            targets = torch.reshape(targets, (batch_size*14,4))

            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
        
        test_loss = sum(total_loss) / len(total_loss)

        auc = roc_auc_score(targets, outputs)
        acc = accuracy_score(targets, outputs)

        return [test_loss, auc, acc]