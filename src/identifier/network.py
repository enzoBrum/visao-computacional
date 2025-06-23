import timm
import torch.nn as nn
import torch

class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "efficientnetv2_rw_s.ra2_in1k", num_features: int = 1792, embedding_size: int = 512):
        super().__init__()
        
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_features, embedding_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.linear(x)
        
