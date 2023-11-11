import torch
import torch.nn as nn

import torchvision

from transformers import ConvNextForImageClassification

import clip

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

class CLIPFinetune(nn.Module):
    """ Model using CLIP as backbone with a linear classifier """
    def __init__(self, num_classes, class_tokens, device, finetune = False):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device = device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(class_tokens).float()
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        
        for param in self.model.parameters():
            param.requires_grad = False    

        self.classifier = nn.Linear(512, 512)
        if not finetune:
            self.classifier.weight = nn.Parameter(torch.eye(512))
            self.classifier.bias = nn.Parameter(torch.zeros(512))

    def forward(self, x):
        with torch.no_grad():
            x_features = self.model.encode_image(x).float()

        x_features /= x_features.norm(dim = -1, keepdim = True)
        return (self.classifier(x_features) @ self.text_features.T)