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

class CNNNetwork(torch.nn.Module): # a class inheriting from nn.Module
"""Un CNN classique avec plusieurs couches"""
  def __init__(self):
    super().__init__()
    self.layer1 = torch.nn.Conv2d(1,profondeur,kernel, padding = (kernel-1)//2)
    self.MaxPool = torch.nn.MaxPool2d(2)
    self.layer3 = torch.nn.Conv2d(profondeur,2*profondeur,kernel, padding = (kernel-1)//2)
    self.layer5 = torch.nn.Conv2d(2*profondeur,4*profondeur,kernel, padding = (kernel-1)//2)
    self.layer6 = torch.nn.Linear(16*16*16,50)
    self.layer7 = torch.nn.Linear(50,2)
    self.layer8 = torch.nn.Linear(128,2)
    self.soft = torch.nn.LogSoftmax()
    
  def forward(self,x):
    #print(x.shape)
    result = self.layer1(x)
    #print(result.shape)
    result = self.MaxPool(result)
    #print(result.shape)
    result = self.layer3(result)
    #print(result.shape)
    result = self.MaxPool(result)
    result = self.layer5(result)
    #print(result.shape)
    result = result.view(result.shape[0], 16*16*16)
    result = self.layer6(result)
    result = self.layer7(result)
    result = self.soft(result)
    return result
