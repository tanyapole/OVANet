from __future__ import print_function
import numpy as np
import yaml
import easydict
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from apex import amp, optimizers
from utils.utils import log_set, save_model
from utils.loss import ova_loss, open_entropy
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_dataloaders, get_models
from eval import test
import argparse
import wandb
from pathlib import Path

parser = argparse.ArgumentParser(description='Pytorch OVANet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml',
                    help='/path/to/config/file')

parser.add_argument('--source_data', type=str,
                    default='./utils/source_list.txt',
                    help='path to source list')
parser.add_argument('--target_data', type=str,
                    default='./utils/target_list.txt',
                    help='path to target list')
parser.add_argument('--log-interval', type=int,
                    default=100,
                    help='how many batches before logging training status')
parser.add_argument('--exp_name', type=str,
                    default='office',
                    help='/path/to/config/file')
parser.add_argument('--network', type=str,
                    default='resnet50',
                    help='network name')
parser.add_argument("--gpu_devices", type=int, nargs='+',
                    default=None, help="")
parser.add_argument("--no_adapt",
                    default=False, action='store_true')
parser.add_argument("--save_model",
                    default=False, action='store_true')
parser.add_argument("--save_path", type=str,
                    default="record/ova_model",
                    help='/path/to/save/model')
parser.add_argument('--multi', type=float,
                    default=0.1,
                    help='weight factor for adaptation')
parser.add_argument('--steps', type=int, help='number of steps, overrides config train.min_step')
args = parser.parse_args()

config_file = args.config
conf = yaml.load(open(config_file))
if args.steps is not None:
    conf['train']['min_step'] = args.steps
conf = easydict.EasyDict(conf)
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
args.cuda = torch.cuda.is_available()

source_data = args.source_data
target_data = args.target_data
evaluation_data = args.target_data
network = args.network
use_gpu = torch.cuda.is_available()
n_share = conf.data.dataset.n_share
n_source_private = conf.data.dataset.n_source_private
n_total = conf.data.dataset.n_total
open = n_total - n_share - n_source_private > 0
num_class = n_share + n_source_private
script_name = os.path.basename(__file__)

inputs = vars(args)
inputs["evaluation_data"] = evaluation_data
inputs["conf"] = conf
inputs["script_name"] = script_name
inputs["num_class"] = num_class
inputs["config_file"] = config_file

source_loader, target_loader, \
test_loader, valid_loader, target_folder = get_dataloaders(inputs)

logname = log_set(inputs)

G, C1, C2, opt_g, opt_c, \
param_lr_g, param_lr_c = get_models(inputs)
ndata = target_folder.__len__()

# os.environ["WANDB_MODE"] = "offline"

def _get_domain_name(filepath):
    filepath = Path(filepath)
    parts = filepath.name[:-len(filepath.suffix)].split('_')
    assert len(parts) == 3, f"can't process filename {filepath}"
    return parts[1]


def _get_run_name(source, target):
    source, target = _get_domain_name(source), _get_domain_name(target)
    return f'{source} -> {target}'



def train():
    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    run = wandb.init(project='original-OVANet', 
                        name='with valid ' + _get_run_name(args.source_data, args.target_data), 
                        config={'source': args.source_data, 'target': args.target_data},
                        tags=['valid'])
    for step in range(conf.train.min_step + 1):
        G.train()
        C1.train()
        C2.train()
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_s = next(data_iter_s)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_c, opt_c, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        img_s = data_s[0]
        label_s = data_s[1]
        img_t = data_t[0]
        img_s, label_s = Variable(img_s.cuda()), \
                         Variable(label_s.cuda())
        img_t = Variable(img_t.cuda())
        opt_g.zero_grad()
        opt_c.zero_grad()
        C2.module.weight_norm()

        ## Source loss calculation
        feat = G(img_s)
        out_s = C1(feat)
        out_open = C2(feat)
        ## source classification loss
        loss_s = criterion(out_s, label_s)
        ## open set loss for source
        out_open = out_open.view(out_s.size(0), 2, -1)
        open_loss_pos, open_loss_neg = ova_loss(out_open, label_s)
        ## b x 2 x C
        loss_open = 0.5 * (open_loss_pos + open_loss_neg)
        ## open set loss for target
        all = loss_s + loss_open
        log_string = 'Train {}/{} \t ' \
                     'Loss Source: {:.4f} ' \
                     'Loss Open: {:.4f} ' \
                     'Loss Open Source Positive: {:.4f} ' \
                     'Loss Open Source Negative: {:.4f} '
        log_values = [step, conf.train.min_step,
                      loss_s.item(),  loss_open.item(),
                      open_loss_pos.item(), open_loss_neg.item()]
        if not args.no_adapt:
            feat_t = G(img_t)
            out_open_t = C2(feat_t)
            out_open_t = out_open_t.view(img_t.size(0), 2, -1)
            ent_open = open_entropy(out_open_t)
            all += args.multi * ent_open
            log_values.append(ent_open.item())
            log_string += "Loss Open Target: {:.6f}"
        with amp.scale_loss(all, [opt_g, opt_c]) as scaled_loss:
            scaled_loss.backward()
        opt_g.step()
        opt_c.step()
        opt_g.zero_grad()
        opt_c.zero_grad()
        if step % conf.train.log_interval == 0:
            print(log_string.format(*log_values))
        run.log({
            'step': step,
            'loss/src': loss_s.item(),
            'loss/open': loss_open.item(),
            'loss/open_src_pos': open_loss_pos.item(),
            'loss/open_src_neg': open_loss_neg.item(),
            'loss/open_tgt': ent_open.item()
        })
        if step > 0 and step % conf.test.test_interval == 0:
            val_loss, val_acc = validate(step, valid_loader, list(range(num_class)), G, C1, criterion)
            acc_o, h_score = test(step, test_loader, logname, n_share, G,
                                  [C1, C2], open=open)
            print("acc all %s h_score %s " % (acc_o, h_score))
            G.train()
            C1.train()
            if args.save_model:
                save_path = "%s_%s.pth"%(args.save_path, step)
                save_model(G, C1, C2, save_path)
            run.log({
                'step': step,
                'acc': acc_o,
                'h_score': h_score,
                'valid_loss': val_loss,
                'valid_acc': val_acc
            })
    run.finish()

def validate(step, valid_loader, src_classes, G, C1, criterion):
    G.eval()
    C1.eval()
    losses, labels, preds = [], [], []
    for batch in valid_loader:
        img, label = batch[0].cuda(), batch[1].cuda()
        with torch.no_grad():
            out = C1(G(img))
            loss = criterion(out, label)
        losses.append(loss)
        labels.append(label)
        preds.append(out.argmax(dim=1))
    loss = sum(losses) / len(losses)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)
    accs = []
    for c in src_classes:
        idxs = labels == c
        comparison = (labels[idxs] == preds[idxs])
        acc = comparison.sum() / len(comparison)
        accs.append(acc.item())
    acc = np.mean(accs)
    return loss.item(), acc


train()
