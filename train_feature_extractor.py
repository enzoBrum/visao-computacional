from pathlib import Path
from time import time

import numpy as np
from pytorch_metric_learning.losses import SubCenterArcFaceLoss
from pytorch_metric_learning.samplers import MPerClassSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2
from tqdm.auto import tqdm

import timm


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnetv2_rw_s.ra2_in1k",
        num_features: int = 1792,
        embedding_size: int = 512,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone, pretrained=True, num_classes=0, global_pool=""
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(num_features, 512)
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Linear(512, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn(x)
        x = self.dropout(x)

        return self.embedding(x)


img_extensions = (*IMG_EXTENSIONS, ".JPG")

training_transforms = v2.Compose(
    [
        v2.RandomAffine(0.0, (0.1, 0.1), (0.95, 1.05)),
        v2.ColorJitter(0.10, 0.1, 0.1),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.485, 0.406], [0.229, 0.224, 0.225]),
    ]
)


val_transforms = v2.Compose(
    [v2.ToTensor(), v2.Normalize([0.485, 0.485, 0.406], [0.229, 0.224, 0.225])]
)

checkpoints_dir = Path("./checkpoints")
checkpoints_dir.mkdir(exist_ok=True)

training_data = ImageFolder("extracted-rois/train", training_transforms)
val_data = ImageFolder("extracted-rois/val", val_transforms)

training_data_idx_to_class = {v: k for k, v in training_data.class_to_idx.items()}
val_data_idx_to_class = {v: k for k, v in val_data.class_to_idx.items()}

training_data_labels = [training_data_idx_to_class[x[1]] for x in training_data.imgs]
val_data_labels = [val_data_idx_to_class[x[1]] for x in val_data.imgs]

batch_size = 32
num_workers = 4
epochs = 50

# start_lr = 2.31e-4
# end_lr = 6.58e-3

start_lr = 1e-4

thresholds = [0.1, 0.3, 0.5, 0.65, 0.85]

train_dataloader = DataLoader(
    training_data,
    batch_size,
    sampler=MPerClassSampler(training_data_labels, 4, batch_size),
    pin_memory=True,
    num_workers=num_workers,
)
val_dataloader = DataLoader(
    val_data,
    batch_size,
    sampler=MPerClassSampler(val_data_labels, 4, batch_size),
    pin_memory=True,
    num_workers=num_workers,
)

num_train_classes = len(training_data.classes)
num_val_classes = len(val_data.classes)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:0")
# device = torch.device("cpu")

model = FeatureExtractor().to(device)
arcface = SubCenterArcFaceLoss(
    num_classes=num_train_classes + num_val_classes, embedding_size=512
)

backbone_params = model.backbone.parameters()
head_params = [p for n, p in model.named_parameters() if "backbone" not in n]


optimizer = torch.optim.AdamW(
    [
        {"params": head_params},
        {"params": backbone_params, "lr": 1e-5},
        {"params": arcface.parameters()},
    ],
    start_lr,
    weight_decay=1e-4,
)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=5
)
main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[5]
)

# from torch_lr_finder import LRFinder

# lr_finder = LRFinder(model, optimizer, arcface, device="cuda:4")
# lr_finder.range_test(train_dataloader, end_lr=100, num_iter=100)
# lr_finder.plot() # to inspect the loss-learning rate graph

# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#    optimizer,
#    end_lr,
#    steps_per_epoch=len(train_dataloader),
#    epochs=epochs,
# )

best_far = 1e9
best_frr = 1e9

for i in range(epochs):
    model.train()

    loop = tqdm(train_dataloader, desc=f"Epoch {i+1}/{epochs}", leave=False)

    total_loss = 0
    avg_loss = 0

    for j, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embedds = model(images)
        loss = arcface(embedds, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (j + 1)
        loop.set_postfix(loss=loss.item(), avg_loss=avg_loss)

    model.eval()
    last_five = []
    with torch.no_grad():
        genuine_pairs_step = 48
        false_pairs_step = 448

        genuine_pairs_total = [0, 0, 0, 0, 0]
        false_pairs_total = [0, 0, 0, 0, 0]

        genuine_pairs_rejected = [0, 0, 0, 0, 0]
        false_pairs_accepted = [0, 0, 0, 0, 0]

        non_identity_matrix = ~(torch.eye(batch_size, batch_size).to(device).bool())
        for j, (images, labels) in enumerate(
            tqdm(val_dataloader, desc=f"Validation", leave=False)
        ):
            images = images.to(device)
            labels = labels.to(device)

            embbedings = model(images)
            normalized_embbedings = F.normalize(embbedings, 2)

            distance_matrix = torch.matmul(
                normalized_embbedings, normalized_embbedings.T
            )

            for k, threshold in enumerate(thresholds):
                rejected_matrix = distance_matrix < threshold

                labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

                false_rejects_matrix = (
                    rejected_matrix & labels_matrix & non_identity_matrix
                )
                false_accepts_matrix = (~rejected_matrix) & (~labels_matrix)

                false_rejects = torch.sum(false_rejects_matrix).item() // 2
                false_accepts = torch.sum(false_accepts_matrix).item() // 2

                genuine_pairs_rejected[k] += false_rejects
                false_pairs_accepted[k] += false_accepts

                genuine_pairs_total[k] += genuine_pairs_step
                false_pairs_total[k] += false_pairs_step

        for k in range(5):
            far = false_pairs_accepted[k] / false_pairs_total[k]
            frr = genuine_pairs_rejected[k] / genuine_pairs_total[k]
            print(
                f"Epoch: {i+1}, Avg. Loss: {avg_loss}, FAR({thresholds[k]}): {far}, FRR({thresholds[k]}): {frr}"
            )

        if (far := false_pairs_accepted[3] / false_pairs_total[3]) < best_far:
            print("Saving Model. New best FAR.")
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),  # <-- Add this
                    "loss": loss,
                },
                str(checkpoints_dir / f"BEST-FAR-Model.pth"),
            )
            best_far = far
        if (frr := genuine_pairs_rejected[3] / genuine_pairs_total[3]) < best_frr:
            print("Saving Model. New best FRR.")
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),  # <-- Add this
                    "loss": loss,
                },
                str(checkpoints_dir / f"BEST-FRR-Model.pth"),
            )
            best_frr = frr
        if (i < 10 and i % 3 == 0) or i % 5 == 0:
            print("Saving Model.")
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),  # <-- Add this
                    "loss": loss,
                },
                str(checkpoints_dir / f"{i+1}-Model.pth"),
            )

        scheduler.step()
