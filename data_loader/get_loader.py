import numpy as np
from sklearn.utils import shuffle
from .mydataset import ImageFolder
from collections import Counter, OrderedDict
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from datetime import datetime as dt

def _get_label(s:str): return int(s.strip().split()[1])
def _get_labels(lines): return list(map(_get_label, lines))
def _get_unique_labels(lines): return np.unique(_get_labels(lines))
def _count(lines): return Counter(_get_labels(lines))
def _rel_count(lines):
    counter = _count(lines)
    d = OrderedDict()
    for k in _get_unique_labels(lines):
        d[k] = counter[k] / len(lines)
    return d

def _make_split(source_path):
    with open(source_path, 'r') as f:
        lines = f.readlines()
    trn, val = train_test_split(lines, test_size=200, stratify=_get_labels(lines), random_state=0)
    now = dt.now().strftime('%m%d_%H%M%S_%f')
    source_path = f'tmp/trn_{now}.txt'
    valid_path = f'tmp/val_{now}.txt'
    with open(source_path, 'w') as g: g.writelines(trn)
    with open(valid_path, 'w') as g: g.writelines(val)
    print('Split: ', len(trn), len(val))
    return source_path, valid_path


def get_loader(source_path, target_path, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False, val=False, val_data=None):

    source_path, valid_path = _make_split(source_path)

    source_folder = ImageFolder(os.path.join(source_path),
                                transforms["src"],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                  transform=transforms["tgt"],
                                  return_paths=False, return_id=return_id)
    if val:
        raise Exception
        source_val_train = ImageFolder(val_data, transforms[source_path], return_id=return_id)
        target_folder_train = torch.utils.data.ConcatDataset([target_folder_train, source_val_train])
        source_val_test = ImageFolder(val_data, transforms[evaluation_path], return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms["eval"],
                                   return_paths=True)
    valid_folder = ImageFolder(valid_path, transform=transforms["eval"] ,return_paths=True)

    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
    valid_loader = torch.utils.data.DataLoader(
        valid_folder,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    if val:
        test_loader_source = torch.utils.data.DataLoader(
            source_val_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4)
        return source_loader, target_loader, test_loader, test_loader_source

    return source_loader, target_loader, test_loader, valid_loader, target_folder_train


def get_loader_label(source_path, target_path, target_path_label, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    target_folder_label = ImageFolder(os.path.join(target_path_label),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    target_loader_label = torch.utils.data.DataLoader(
        target_folder_label,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, target_loader_label, test_loader, target_folder_train



