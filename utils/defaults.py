import torch
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
from apex import amp, optimizers
from data_loader.get_loader import get_loader, get_loader_label
from models.basenet import ResBase, ResClassifier_MME


def get_dataloaders(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    evaluation_data = kwargs["evaluation_data"]
    conf = kwargs["conf"]
    val_data = None
    if "val" in kwargs:
        val = kwargs["val"]
        if val:
            val_data = kwargs["val_data"]
    else:
        val = False

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader(source_data, target_data, evaluation_data,
                      data_transforms,
                      batch_size=conf.data.dataloader.batch_size,
                      return_id=True,
                      balanced=conf.data.dataloader.class_balance,
                      val=val, val_data=val_data)



def get_dataloaders_label(source_data, target_data, target_data_label, evaluation_data, conf):

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        evaluation_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader_label(source_data, target_data, target_data_label,
                            evaluation_data, data_transforms,
                            batch_size=conf.data.dataloader.batch_size,
                            return_id=True,
                            balanced=conf.data.dataloader.class_balance)

