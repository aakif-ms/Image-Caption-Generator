import torch 
import torch.nn as nn
from torchvision import models

class CNNEncoder(nn.Module):
    def __init__(self, embed_size, train=False):
        super().__init__()
        
        self.train = train
        self.embed_size = embed_size
        
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        self.linear = nn.Linear(
            self.resnet[-1].in_features if hasattr(self.resnet[-1], 'in_features') else 2048, embed_size
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        if not self.train:
            with torch.no_grad():
                features = self.resnet(x)
        else:
            features = self.resnet(x)
        
        features = features.reshape(features.size(0), -1)
        features = self.dropout(self.relu(self.linear(features)))
        return features