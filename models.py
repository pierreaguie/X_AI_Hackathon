import torch
import torch.nn as nn

import torchvision

from transformers import ConvNextForImageClassification

class TransformerFinetune(nn.Module):
    """ Model using vit_b_16 as backbone with a linear classifier """
    def __init__(self, num_classes, frozen = False):
        super().__init__()
        
        self.backbone = torchvision.models.vit_b_16("ViT_B_16_Weights.DEFAULT")
        self.backbone.heads = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class ConvNextFinetune(nn.Module):
    """ Model using ConvNext as backbone with a linear classifier """
    def __init__(self, num_classes, frozen = False):
        super().__init__()
        
        self.backbone = ConvNextForImageClassification.from_pretrained("facebook/convnext-large-224")
        self.backbone.classifier = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.classifier = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.backbone(x).logits
        x = self.classifier(x)
        return x
