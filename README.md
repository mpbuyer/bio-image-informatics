# Bio-Image-Informatics
This repository is meant to showcase classifying, detecting, and segmenting diseases in the thorax area based on chest x-ray data. The python files are the "scars" from setting up experiments, and the majority of work is from other repositories linked below. Experiments are ran on a supercomputing cluster, using 1-4 GPUs at a time. I primarily used the Lightning framework if I was not cloning repositories, and PyTorch otherwise. 

## Classification
The MIMIC (Medical Information Mart for Intensive Care) dataset https://physionet.org/content/mimic-cxr/2.0.0/ is for classification. I first tested supervised learning with Swin Transformer https://github.com/microsoft/Swin-Transformer. I could improve my AUC of 0.7446 by improving hyperparameters. CheXzero https://github.com/rajpurkarlab/CheXzero is CLIP-like. I did zero-shot classification and got a subpar AUC of 0.646. CheSS https://github.com/mi2rl/CheSS is self-supervised contrastive learning. I fine-tuned it for classification and got an AUC of 0.808. Finally, I continued pretraining I-JEPA https://github.com/facebookresearch/ijepa and didn't get classification results (yet).

## Localization
Localization involved detecting nodules given this challenge dataset: https://node21.grand-challenge.org. I tried Faster R-CNN from Torchvision. Future work would try a detection transformer since my mean average precision was merely 0.1.

## Segmentation
The dataset is a small one but it has 14 labels with boxes besides mask contours: https://github.com/Deepwise-AILab/ChestX-Det-Dataset. I fine-tuned a self-supervised method called PEAC https://github.com/jlianglab/PEAC for segmentation. I got a dice score of 0.909, but with binary segmentation instead (disease or no disease).

