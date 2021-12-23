import yaml
import easydict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from apex import amp#, optimizers
# from utils.loss import ova_loss, open_entropy
# from utils.defaults import get_models, get_dataloaders
from data_handling import api
import torchvision.transforms as TF
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np
# from eval import test
import argparse
import wandb


class ValueClipper:
    def __init__(self, max_val): self.max_val = max_val
    def clip(self, sample): return sample[0], min(sample[1], self.max_val)

def next_safe(itr, dl):
    try:
        batch = next(itr)
    except:
        itr = iter(dl)
        batch = next(itr)
    return batch, itr

def get_domain(domain_name):
    for el in api.Domain:
        if el.value == domain_name:
            return el
    raise Exception(f'Failed to create domain {domain_name}')

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

def create_dataloaders(source:api.Domain, target:api.Domain, batch_size:int):
    source = get_domain(source)
    target = get_domain(target)

    src_filelist = api.get_UDA_filelist(source)
    src_filelist = list(filter(lambda x: x[1] < 15, src_filelist))
    tgt_filelist = api.get_UDA_filelist(target)
    tgt_filelist = list(filter(lambda x: ((x[1] < 10) or (x[1] >= 15)), tgt_filelist))
    value_clipper = ValueClipper(15)
    tgt_filelist = list(map(value_clipper.clip, tgt_filelist))

    trn_tfm = TF.Compose([TF.Resize((256,256)), TF.RandomHorizontalFlip(), TF.RandomCrop(224)])
    eval_tfm = TF.Compose([TF.Resize((256,256)), TF.CenterCrop(224)])
    src_ds = api.FilelistDataset(src_filelist, mode=api.ImageMode.Tensor, tfm=trn_tfm, prefix=api._prefix)
    tgt_ds = api.FilelistDataset(tgt_filelist, mode=api.ImageMode.Tensor, tfm=trn_tfm, prefix=api._prefix)
    eval_ds = api.FilelistDataset(tgt_filelist, mode=api.ImageMode.Tensor, tfm=eval_tfm, prefix=api._prefix)

    src_labels = [_[1] for _ in src_ds.samples]
    freq2 = Counter(src_labels)

    class_weight = {x: 1.0 / freq2[x] for x in freq2}
    source_weights = [class_weight[x] for x in src_labels]
    sampler = WeightedRandomSampler(source_weights, len(src_labels))
    print("use balanced loader")
    src_dl = DataLoader(src_ds, batch_size=batch_size, sampler=sampler, drop_last=True)

    tgt_dl = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    return src_dl, tgt_dl, eval_dl

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

def save_model(model_g, model_c1, model_c2, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c1_state_dict': model_c1.state_dict(),
        'c2_state_dict': model_c2.state_dict(),
    }
    torch.save(save_dic, save_path)

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=10,
                     power=0.75, init_lr=0.001,weight_decay=0.0005,
                     max_iter=10000):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    gamma = 10.0
    lr = init_lr * (1 + gamma * min(1.0, iter_num / max_iter)) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i+=1
    return lr


def ova_loss(out_open, label):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    out_open = F.softmax(out_open, 1)
    label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
    label_range = torch.range(0, out_open.size(0) - 1).long()
    label_p[label_range, label] = 1
    label_n = 1 - label_p
    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                    + 1e-8) * label_p, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
    return open_loss_pos, open_loss_neg

def open_entropy(out_open):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1)
    ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
    return ent_open

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

    n_share = conf.data.dataset.n_share
    n_source_private = conf.data.dataset.n_source_private
    n_total = conf.data.dataset.n_total
    is_open = n_total - n_share - n_source_private > 0
    num_class = n_share + n_source_private

    # data
    source_loader, target_loader, test_loader = create_dataloaders(args.source, args.target, conf.data.dataloader.batch_size)

    # models
    G = ResBase().cuda().eval()
    C1 = ResClassifier_MME(num_classes=num_class, norm=False, input_size=2048).cuda().eval()
    C2 = ResClassifier_MME(num_classes=2 * num_class, norm=False, input_size=2048).cuda().eval()

    params = [{'params': [p], 'lr': conf.train.multi, 'weight_decay': conf.train.weight_decay} for (n,p) in G.named_parameters()]
    opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum, weight_decay=0.0005, nesterov=True)
    opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0, momentum=conf.train.sgd_momentum, weight_decay=0.0005, nesterov=True)
    [G, C1, C2], [opt_g, opt_c] = amp.initialize([G, C1, C2], [opt_g, opt_c], opt_level="O1")

    param_lr_g = [param_group["lr"] for param_group in opt_g.param_groups]
    param_lr_c = [param_group["lr"] for param_group in opt_c.param_groups]

    # training
    criterion = nn.CrossEntropyLoss().cuda()
    print('train start!')
    src_iter = iter(source_loader)
    tgt_iter = iter(target_loader)
    run = wandb.init(project='debug-ova', 
                        name=f'original {args.source}->{args.target}', 
                        config={'source': args.source, 'target': args.target}, 
                        tags=['more_changes'])
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
            print(f"acc all {acc_o} h_score {h_score}")
            if args.save_model:
                save_path = f"{args.save_path}_{step}.pth"
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

    parser.add_argument('--source', choices=['art', 'clipart', 'product', 'real_world'], help='source domain')
    parser.add_argument('--target', choices=['art', 'clipart', 'product', 'real_world'], help='source domain')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches before logging training status')
    parser.add_argument('--exp_name', type=str, default='office', help='/path/to/config/file')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
    parser.add_argument("--no_adapt", default=False, action='store_true')
    parser.add_argument("--save_model", default=False, action='store_true')
    parser.add_argument("--save_path", type=str, default="record/ova_model", help='/path/to/save/model')
    parser.add_argument('--multi', type=float, default=0.1, help='weight factor for adaptation')
    parser.add_argument('--steps', type=int, help='number of steps, overrides config train.min_step')
    args = parser.parse_args()
    train(args)
