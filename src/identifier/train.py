from time import time
import numpy as np
from tqdm import tqdm
from network import FeatureExtractor

import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2
from pytorch_metric_learning.losses import SubCenterArcFaceLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from pathlib import Path


img_extensions = (*IMG_EXTENSIONS, ".JPG")

training_transforms = v2.Compose([
    v2.RandomAffine(0.0, (0.05, 0.05), (0.98, 1.02)),
    v2.ColorJitter(0.10, 0.1, 0.1),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.485, 0.406], [0.229, 0.224, 0.225])
])


val_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Normalize([0.485, 0.485, 0.406], [0.229, 0.224, 0.225])
])

checkpoints_dir = Path("./checkpoints")
checkpoints_dir.mkdir(exist_ok=True)

training_data = ImageFolder("/home/erb/ufsc/ine5448/extracted-rois/train", training_transforms)
val_data = ImageFolder("/home/erb/ufsc/ine5448/extracted-rois/val", val_transforms)

training_data_idx_to_class = {v: k for k, v in training_data.class_to_idx.items()}
val_data_idx_to_class = {v: k for k, v in val_data.class_to_idx.items()}

training_data_labels = [training_data_idx_to_class[x[1]] for x in training_data.imgs]
val_data_labels = [val_data_idx_to_class[x[1]] for x in val_data.imgs]

batch_size = 32
num_workers = 0
epochs = 150

start_lr = 2.31e-4
end_lr = 6.58e-3

threshold = 0.85

train_dataloader = DataLoader(training_data, batch_size, sampler=MPerClassSampler(training_data_labels, 4, batch_size), pin_memory=True, num_workers=num_workers)
val_dataloader = DataLoader(val_data, batch_size, sampler=MPerClassSampler(val_data_labels, 4, batch_size),  pin_memory=True, num_workers=num_workers)

num_train_classes = len(training_data.classes)
num_val_classes = len(val_data.classes)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = FeatureExtractor().to(device)
arcface = SubCenterArcFaceLoss(num_classes=num_train_classes + num_val_classes, embedding_size=512)

optimizer = torch.optim.AdamW([{"params": model.parameters()}, {"params": arcface.parameters()}], start_lr, weight_decay=1e-2)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, end_lr, steps_per_epoch=len(train_dataloader), epochs=epochs)

for i in range(epochs):
    model.train()
    
    loop = tqdm(train_dataloader, desc=f"Epoch {i+1}/{epochs}", leave=False)
    
    total_loss = 0
    avg_loss = 0
    for j, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)
        
        embedds = model(images)
        loss = arcface(embedds, labels)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss/(j+1)
        loop.set_postfix(loss=loss.item(), avg_loss=avg_loss)

    model.eval()
    with torch.no_grad():
        genuine_pairs_step = 48
        false_pairs_step = 448

        genuine_pairs_total = 0
        false_pairs_total = 0
        
        genuine_pairs_rejected = 0
        false_pairs_accepted = 0
        
        non_identity_matrix = ~(torch.eye(batch_size, batch_size).to(device).bool())
        for j, (images, labels) in enumerate(tqdm(val_dataloader, desc=f"Validation", leave=False)):
            images = images.to(device)
            labels = labels.to(device)

            embbedings = model(images)
            normalized_embbedings = F.normalize(embbedings, 2)
            
            distance_matrix = torch.matmul(normalized_embbedings, normalized_embbedings.T)
            rejected_matrix = distance_matrix < threshold

            labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

            false_rejects_matrix = rejected_matrix & labels_matrix & non_identity_matrix
            false_accepts_matrix = (~rejected_matrix) & (~labels_matrix)
            
            false_rejects = torch.sum(false_rejects_matrix).item()//2
            false_accepts = torch.sum(false_accepts_matrix).item()//2
            
            genuine_pairs_rejected += false_rejects
            false_pairs_accepted += false_accepts

            genuine_pairs_total += genuine_pairs_step
            false_pairs_total += false_pairs_step

        print(f"Epoch: {i+1}, Avg. Loss: {avg_loss}, FAR: {false_pairs_accepted/false_pairs_total}, FRR: {genuine_pairs_rejected/genuine_pairs_total}")

        if i % 10 == 0:
            print("Saving Model.")
            torch.save(model.state_dict(), str(checkpoints_dir / f"{j}-Model.pth"))
            with (checkpoints_dir / f"{j}-Model.txt").open("w") as f:
                f.write(f"Epoch: {i+1}, Avg. Loss: {avg_loss}, FAR: {false_pairs_accepted/false_pairs_total}, FRR: {genuine_pairs_rejected/genuine_pairs_total}")
    
            
            
