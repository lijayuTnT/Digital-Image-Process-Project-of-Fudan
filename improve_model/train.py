# coding: utf-8
import argparse
import os
import math
import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from models.model import DM2FNet
from models.CR import *
from models.metrics import psnr
from tools.config import TRAIN_ITS_ROOT, TEST_SOTS_ROOT
from datasets import ItsDataset, SotsDataset
from tools.utils import AvgMeter, check_mkdir



def parse_args():
    parser = argparse.ArgumentParser(description='Train a DM2FNet')
    parser.add_argument(
        '--gpus', type=str, default='0', help='gpus to use ')
    parser.add_argument('--ckpt-path', default='./ckpt_path', help='checkpoint path')
    parser.add_argument(
        '--exp-name',
        default='RESIDE_ITS_G',
        help='experiment name.')
    args = parser.parse_args()

    return args


cfgs = {
    'use_physical': True,
    'iter_num': 20000,
    'train_batch_size': 1,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'w_loss_l1':1,
    'w_loss_vgg7' : 0.,
    'cl_lambda' :0.25,
    'snapshot': '',
    'val_freq': 1000,
    'crop_size': 256,
    'blocks' :19,
}


def main():
    net = DM2FNet(cfgs['blocks']).cuda().train()
    # net = nn.DataParallel(net)

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * cfgs['lr']},
        {'params': [param for name, param in net.named_parameters()
                    if name[-4:] != 'bias' and param.requires_grad],
         'lr': cfgs['lr'], 'weight_decay': cfgs['weight_decay']}
    ])

    if len(cfgs['snapshot']) > 0:
        print('training resumes from \'%s\'' % cfgs['snapshot'])
        net.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                    args.exp_name, cfgs['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path,
                                                          args.exp_name, cfgs['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * cfgs['lr']
        optimizer.param_groups[1]['lr'] = cfgs['lr']

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    open(log_path, 'w').write(str(cfgs) + '\n\n')

    train(net, optimizer)

def train(net, optimizer):
    curr_iter = cfgs['last_iter']
    w_loss_vgg = cfgs['w_loss_vgg7']
    while curr_iter <= cfgs['iter_num']:
        train_loss_record = AvgMeter()
        loss_x_jf_record, loss_x_j0_record = AvgMeter(), AvgMeter()
        loss_x_j1_record, loss_x_j2_record = AvgMeter(), AvgMeter()
        loss_x_j3_record, loss_x_j4_record = AvgMeter(), AvgMeter()
        loss_t_record, loss_a_record = AvgMeter(), AvgMeter()

        for data in train_loader:
            optimizer.param_groups[0]['lr'] = 2 * cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']
            optimizer.param_groups[1]['lr'] = cfgs['lr'] * (1 - float(curr_iter) / cfgs['iter_num']) \
                                              ** cfgs['lr_decay']

            haze, gt_trans_map, gt_ato, gt, _ = data

            batch_size = haze.size(0)

            haze = haze.cuda()
            gt_trans_map = gt_trans_map.cuda()
            gt_ato = gt_ato.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()

            x_jf, x_j0, x_j1, x_j2, x_j3, x_j4,t, a = net(haze)

            loss_l1 =0
            loss_x_jf = criterion[0](x_jf, gt)
            loss_x_j0 = criterion[0](x_j0, gt)
            loss_x_j1 = criterion[0](x_j1, gt)
            loss_x_j2 = criterion[0](x_j2, gt)
            loss_x_j3 = criterion[0](x_j3, gt)
            loss_x_j4 = criterion[0](x_j4, gt)
            loss_t = criterion[0](t, gt_trans_map)
            loss_a = criterion[0](a, gt_ato)
            if cfgs['w_loss_l1']>0 :
                loss_l1 = loss_x_jf + loss_x_j0 + loss_x_j1 + loss_x_j2 + loss_x_j3 + loss_x_j4\
                    + 10 * loss_t + loss_a
            if cfgs['w_loss_vgg7']>0:
                loss_vgg = criterion[1](x_jf, gt ,haze)
            else:
                loss_vgg = 0.
                # w_loss_vgg = 0.5 * (1 + math.cos(curr_iter * math.pi / cfgs['iter_num'])) * init_w_vgg
            loss = cfgs['w_loss_l1'] * loss_l1 + w_loss_vgg * loss_vgg

            loss.backward()

            optimizer.step()

            # update recorder
            train_loss_record.update(loss.item(), batch_size)

            loss_x_jf_record.update(loss_x_jf.item(), batch_size)
            loss_x_j0_record.update(loss_x_j0.item(), batch_size)
            loss_x_j1_record.update(loss_x_j1.item(), batch_size)
            loss_x_j2_record.update(loss_x_j2.item(), batch_size)
            loss_x_j3_record.update(loss_x_j3.item(), batch_size)
            loss_x_j4_record.update(loss_x_j4.item(), batch_size)
            loss_t_record.update(loss_t.item(), batch_size)
            loss_a_record.update(loss_a.item(), batch_size)

            curr_iter += 1

            if curr_iter % 20 == 0:
                log = '[iter %d], [train loss %.5f], [loss_x_fusion %.5f], [loss_x_phy %.5f], [loss_x_j1 %.5f], ' \
                    '[loss_x_j2 %.5f], [loss_x_j3 %.5f], [loss_x_j4 %.5f],[loss_t %.5f], [loss_a %.5f], ' \
                    '[lr %.13f]' % \
                    (curr_iter, train_loss_record.avg, loss_x_jf_record.avg, loss_x_j0_record.avg,
                    loss_x_j1_record.avg, loss_x_j2_record.avg, loss_x_j3_record.avg, loss_x_j4_record.avg,
                    loss_t_record.avg, loss_a_record.avg, optimizer.param_groups[1]['lr'])
                open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % cfgs['val_freq'] == 0:
                validate(net, curr_iter, optimizer)

            if curr_iter > cfgs['iter_num']:
                break


def validate(net, curr_iter, optimizer):
    print('validating...')
    net.eval()

    loss_record = AvgMeter()
    psnrs = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            haze, gt, _ = data

            haze = haze.cuda()
            gt = gt.cuda()

            dehaze = net(haze)

            loss = criterion[0](dehaze, gt)
            loss_record.update(loss.item(), haze.size(0))
            psnr_= psnr(dehaze,gt)
            psnrs.append(psnr_)
    
    snapshot_name = 'iter_%d_loss_%.5f_lr_%.6f_psnr_%.5f' % (curr_iter + 1, loss_record.avg, optimizer.param_groups[1]['lr'],np.mean(psnrs))
    print('[validate]: [iter %d], [loss %.5f], [psnr %.5f] ' % (curr_iter + 1, loss_record.avg , np.mean(psnrs)))
    torch.save(net.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(),
               os.path.join(args.ckpt_path, args.exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True
    torch.cuda.set_device(int(args.gpus))

    train_dataset = ItsDataset(TRAIN_ITS_ROOT, True, cfgs['crop_size'])
    train_loader = DataLoader(train_dataset, batch_size=cfgs['train_batch_size'], num_workers=4,
                              shuffle=True, drop_last=True)

    val_dataset = SotsDataset(TEST_SOTS_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=2)
    
    criterion = []
    criterion.append(nn.L1Loss().cuda())
    criterion.append(ContrastLoss().cuda())
    log_path = os.path.join(args.ckpt_path, args.exp_name, str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + '.txt')
    main()