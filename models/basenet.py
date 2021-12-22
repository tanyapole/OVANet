from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn


class ResBase(nn.Module):
    def __init__(self):
        super(ResBase, self).__init__()
        self.dim = 2048
        model_ft = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*(list(model_ft.children())[:-1]))           

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        return x


class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05, norm=True):
        super(ResClassifier_MME, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))
    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)
