

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
import numpy as np


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

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(inputs.to(device))
        outputs = outputs.view(-1, 14, 4)

        targets = targets.to(torch.float32).to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, data_loader, criterion, device):
    model.eval()
    
    total_loss = []
    truths = np.empty((0,4))
    predictions = np.empty((0,4))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))
            outputs = outputs.view(-1, 14, 4) 
            # the model wants 56 targets
            # the loss function wants (14,4)
            # scikit-learn wants them back to 56
            
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
            total_loss.append(loss.item())

            m = nn.Softmax(dim=-1)
            outputs = m(outputs)
            
            outputs = outputs.view(outputs.size(0)*outputs.size(1), -1).cpu().numpy()
            targets = targets.view(targets.size(0)*targets.size(1), -1).cpu().numpy()

            truths = np.concatenate((truths, targets),0)
            predictions = np.concatenate((predictions, outputs),0)
        
        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, computeAUC(truths,predictions), computeAccuracy(truths, predictions)]
    

def computeAUC(targets, outputs):
    pr_auc_scores = []
    for label_index in range(targets.shape[1]):
        true_labels = targets[:, label_index]
        predicted_scores = outputs[:, label_index]
        precision, recall, _ = precision_recall_curve(true_labels, predicted_scores)
        auc_score = auc(recall, precision)
        pr_auc_scores.append(auc_score)
    return np.mean(pr_auc_scores)

def computeAccuracy(targets, outputs):
    predicted_classes = np.argmax(outputs, axis=-1)
    predicted_binarized = (np.arange(4) == predicted_classes[..., None]).astype(int)
    return accuracy_score(targets.flatten(), predicted_binarized.flatten())
