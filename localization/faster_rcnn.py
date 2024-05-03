

import torch
import torchvision.models.detection as models
import lightning as L
from torchmetrics.detection import MeanAveragePrecision
import numpy as np



class Fasterrcnn(L.LightningModule):
    def __init__(self, pretrained, num_epochs):
        super().__init__()
        if pretrained:
            self.model = models.fasterrcnn_resnet50_fpn(weights = models.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        else:
            self.model = models.fasterrcnn_resnet50_fpn()

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)

        self.lr = 1e-4
        self.milestones = [0.75 * num_epochs] # for scheduler
        self.gamma = 0.1 # for scheduler
        self.iou_thresholds = [i for i in np.arange(0.5,1,0.05)]
        self.metric = MeanAveragePrecision(iou_thresholds=self.iou_thresholds)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        #print(f"Batch {batch_idx}: targets={targets}")
        outputs = self.model(inputs, targets)
        losses = {k: v.mean() for k,v in outputs.items()}
        #for k,v in losses.items():
         #   print(f"{k}: {v}")
        loss = sum(l for l in losses.values())
        self.log('train_loss', loss)
        return loss
    
    def on_validation_start(self):
        self.predictions = []
        self.targets = []
        self.total_images = 0
    
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        #print(f"Batch {batch_idx}: inputs shape={[img.shape for img in inputs]}, targets={targets}")
        outputs = self.model(inputs, targets)
        # print(outputs)
        self.metric.update(outputs, targets)

        for dict in outputs:
            for i in range(dict["boxes"].shape[0]):
                score = dict["scores"][i]
                x1,y1,x2,y2 = dict["boxes"][i]
                self.predictions.append((score,x1,y1,x2,y2))

        for dict in targets:
            for i in range(dict["boxes"].shape[0]):
                x1,y1,x2,y2 = dict["boxes"][i]
                self.targets.append((x1,y1,x2,y2))

        self.total_images += len(inputs)


    def on_validation_epoch_end(self):

        # Calculate AP for the validation set
        mAP = self.metric.compute()

        # for threshold in self.iou_thresholds:
        #     self.log(f'val_AP_{threshold}', mAP[threshold], sync_dist=True)

        self.log(f'val_AP_0.5', mAP['map_50'], sync_dist=True)

        # get_FROC(self.predictions,self.targets,self.total_images)



        self.metric.reset()
        self.predictions = []
        self.targets = []
        self.total_images = 0



# FROC
    
import matplotlib.pyplot as plt

# Example lists of predictions and targets
# Each prediction is a tuple: (confidence_score, x1, y1, x2, y2)


# Each target is a tuple: (x1, y1, x2, y2)

# Function to calculate the Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Calculate the coordinates of the intersection rectangle
    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)
    
    # Calculate the area of intersection rectangle
    inter_area = max(0, x6 - x5 + 1) * max(0, y6 - y5 + 1)
    
    # Calculate the area of both bounding boxes
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    
    # Calculate the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    
    return iou

def get_FROC(predictions, targets, total_images):

    # Sort the predictions by confidence score in descending order
    sorted_predictions = sorted(predictions, key=lambda x: x[0], reverse=True)

    # Initialize the FROC curve data
    froc_data = []
    total_targets = len(targets)

    # Iterate through the sorted predictions
    tp, fp, fn = 0, 0, total_targets
    for conf_score, x1, y1, x2, y2 in sorted_predictions:
        # Check if the prediction matches any target
        matched = False
        for target in targets:
            iou = calculate_iou((x1, y1, x2, y2), target)
            if iou >= 0.5:  # Adjust the IoU threshold as needed
                matched = True
                tp += 1
                targets.remove(target)
                break
        
        if not matched:
            fp += 1
        
        fn = total_targets - tp
        if tp == 0:
            sensitivity = 0
            avg_fp_per_image = 0 if fp == 0 else fp / total_images
        else:
            sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
            avg_fp_per_image = fp / total_images

        froc_data.append((sensitivity, avg_fp_per_image))

    # Plot the FROC curve
    sensitivities, avg_fps = zip(*froc_data)
    plt.plot(avg_fps, sensitivities)
    plt.xlabel('Average False Positives per Image')
    plt.ylabel('Sensitivity')
    plt.title('FROC Curve')
    plt.show()

    # Save the plot as an image file
    plt.savefig('froc_curve.png', dpi=300, bbox_inches='tight')

