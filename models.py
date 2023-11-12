import torch
import torch.nn as nn

import torchvision

from transformers import ConvNextForImageClassification

class TransformerFinetune(nn.Module):
    """ Model using vit_b_16 as backbone with a linear classifier """
    def __init__(self, num_classes, pretrained = False, frozen = False, path = None):
        super().__init__()
        
        self.backbone = torchvision.models.vit_b_16()
        self.backbone.heads = nn.Identity()
        if pretrained:
            self.backbone.load_state_dict(torch.load(path)())
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class ResNet(nn.Module):
    def __init__(self,num_classes,pretrained = False, frozen = False, path = None):
        super().__init__()
        
        self.backbone = torchvision.models.resnet18()
        if pretrained:
            self.backbone.load_state_dict(torch.load(path)())
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class BasicTransformer(nn.module):
    """ Basic Transformer model"""
    def __init__(self, num_classes, d_model):
        super().__init__()

        self.d_model = d_model

        self.Linear1 = nn.Linear(64 * 64, d_model)
        self.Transformer = nn.Transformer(d_model = d_model)
        self.Linear2 = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.Transformer(x)
        return self.Linear2(x)

