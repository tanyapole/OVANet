from __future__ import print_function
import yaml
import easydict
import os
import torch
from torch import nn
import torch.nn.functional as F
from apex import amp, optimizers
from utils.utils import log_set, save_model
from utils.loss import ova_loss, open_entropy
from utils.lr_schedule import inv_lr_scheduler
from utils.defaults import get_models, get_dataloaders
from data_handling import api
import numpy as np
# from eval import test
import argparse
import wandb
from pathlib import Path

class ValueClipper:
    def __init__(self, max_val): self.max_val = max_val
    def clip(self, sample): return sample[0], min(sample[1], self.max_val)

def _get_domain_name(filepath):
    filepath = Path(filepath)
    parts = filepath.name[:-len(filepath.suffix)].split('_')
    assert len(parts) == 3, f"can't process filename {filepath}"
    return parts[1]

def _get_run_name(source, target):
    source, target = _get_domain_name(source), _get_domain_name(target)
    return f'{source} -> {target}'

def next_safe(itr, dl):
    try:
        batch = next(itr)
    except:
        itr = iter(dl)
        batch = next(itr)
    return batch, itr

def test(step, dataset_test, n_share, G, Cs, open_class = None, is_open=False, entropy=False, thr=None):
    G.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0].cuda(), data[1].cuda()
            feat = G(img_t).squeeze(-1).squeeze(-1)
            out_t = Cs[0](feat)
            if batch_idx == 0:
                open_class = int(out_t.size(1))
                class_list.append(open_class)
            pred = out_t.data.max(1)[1]
            correct_close += pred.eq(label_t.data).cpu().sum()
            out_t = F.softmax(out_t)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                out_open = Cs[1](feat)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                tmp_range = torch.range(0, out_t.size(0)-1).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > 0.5)[0]
            pred[ind_unk] = open_class
            correct += pred.eq(label_t.data).cpu().sum()
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if is_open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. *float(correct_close) / close_count
    known_acc = per_class_acc[:open_class - 1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    return acc_all, h_score


def train(args):
    config_file = args.config
    with open(config_file, 'r') as f:
        conf = yaml.load(f)
    if args.steps is not None:
        conf['train']['min_step'] = args.steps
    conf = easydict.EasyDict(conf)
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    # os.environ["WANDB_MODE"] = "offline"
    args.cuda = torch.cuda.is_available()

    source_data = args.source_data
    target_data = args.target_data
    evaluation_data = args.target_data
    network = args.network
    use_gpu = torch.cuda.is_available()
    n_share = conf.data.dataset.n_share
    n_source_private = conf.data.dataset.n_source_private
    n_total = conf.data.dataset.n_total
    is_open = n_total - n_share - n_source_private > 0
    num_class = n_share + n_source_private
    script_name = os.path.basename(__file__)

    inputs = vars(args)
    inputs["evaluation_data"] = evaluation_data
    inputs["conf"] = conf
    inputs["script_name"] = script_name
    inputs["num_class"] = num_class
    inputs["config_file"] = config_file

    source_loader, target_loader, \
    test_loader, target_folder = get_dataloaders(inputs)

    logname = log_set(inputs)

    G, C1, C2, opt_g, opt_c, \
    param_lr_g, param_lr_c = get_models(inputs)
    ndata = target_folder.__len__()


    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    src_iter = iter(source_loader)
    tgt_iter = iter(target_loader)
    # len_train_source = len(source_loader)
    # len_train_target = len(target_loader)
    run = wandb.init(project='debug-ova', 
                        name='original ' + _get_run_name(args.source_data, args.target_data), 
                        config={'source': args.source_data, 'target': args.target_data},
                        tags=['move_code']
                        )

    for step in range(conf.train.min_step + 1):
        G.train()
        C1.train()
        C2.train()
        data_t, tgt_iter = next_safe(tgt_iter, target_loader)
        data_s, src_iter = next_safe(src_iter, source_loader)
        inv_lr_scheduler(param_lr_g, opt_g, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        inv_lr_scheduler(param_lr_c, opt_c, step,
                         init_lr=conf.train.lr,
                         max_iter=conf.train.min_step)
        img_s = data_s[0].cuda()
        label_s = data_s[1].cuda()
        img_t = data_t[0].cuda()
        # img_s, label_s = Variable(img_s.cuda()), \
        #                  Variable(label_s.cuda())
        # img_t = Variable(img_t.cuda())
        opt_g.zero_grad()
        opt_c.zero_grad()
        C2.weight_norm()

        ## Source
        feat = G(img_s)
        out_s = C1(feat)
        out_open = C2(feat)
        loss_s = criterion(out_s, label_s)
        out_open = out_open.view(out_s.size(0), 2, -1)
        open_loss_pos, open_loss_neg = ova_loss(out_open, label_s)
        loss_open = 0.5 * (open_loss_pos + open_loss_neg)
        all = loss_s + loss_open
        # target
        feat_t = G(img_t)
        out_open_t = C2(feat_t)
        out_open_t = out_open_t.view(img_t.size(0), 2, -1)
        ent_open = open_entropy(out_open_t)
        all += args.multi * ent_open
        # optimization
        with amp.scale_loss(all, [opt_g, opt_c]) as scaled_loss:
            scaled_loss.backward()
        opt_g.step()
        opt_c.step()
        opt_g.zero_grad()
        opt_c.zero_grad()
        run.log({
            'step': step,
            'loss/src': loss_s.item(),
            'loss/open': loss_open.item(),
            'loss/open_src_pos': open_loss_pos.item(),
            'loss/open_src_neg': open_loss_neg.item(),
            'loss/open_tgt': ent_open.item()
        })
        if step > 0 and step % conf.test.test_interval == 0:
            acc_o, h_score = test(step, test_loader, n_share, G, [C1, C2], is_open=is_open)
            print("acc all %s h_score %s " % (acc_o, h_score))
            # G.train()
            # C1.train()
            if args.save_model:
                save_path = "%s_%s.pth"%(args.save_path, step)
                save_model(G, C1, C2, save_path)
            run.log({
                'step': step,
                'acc': acc_o,
                'h_score': h_score
            })
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch OVANet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='/path/to/config/file')

    parser.add_argument('--source_data', type=str, help='path to source list')
    parser.add_argument('--target_data', type=str, help='path to target list')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches before logging training status')
    parser.add_argument('--exp_name', type=str, default='office', help='/path/to/config/file')
    parser.add_argument('--network', type=str, default='resnet50', help='network name')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    parser.add_argument("--no_adapt", default=False, action='store_true')
    parser.add_argument("--save_model", default=False, action='store_true')
    parser.add_argument("--save_path", type=str, default="record/ova_model", help='/path/to/save/model')
    parser.add_argument('--multi', type=float, default=0.1, help='weight factor for adaptation')
    parser.add_argument('--steps', type=int, help='number of steps, overrides config train.min_step')
    args = parser.parse_args()
    train(args)
