import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from utils.dataloader import get_loader1, test_dataset1
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
from backbone.U_Net import Unet
from backbone.MIA_Net import backbone
from backbone.Polyp_PVT import PolypPVT
from config import config

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def diceCoeff(pred, gt, eps=1e-5, activation='none'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)
    # print(activation_fn)
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + eps) / (unionset + eps)
    # print("Batch Dice:", loss)
    return loss.sum() / N


class SoftDiceLoss2c(nn.Module):
    __name__ = 'dice_loss'

    # 二分类,好像没有传背景
    def __init__(self, activation=None):
        super(SoftDiceLoss2c, self).__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeff(y_pr, y_gt, activation=self.activation)


def test(model, path, dataset, size):
    dice_list = []
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset1(image_root, gt_root, size)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        if config['out'] == 1:
            res = model(image)
        elif config['out'] == 2:
            res, _ = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        dice_list.append(dice)
        DSC = DSC + dice
    print(dice_list)
    return DSC / num1


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_P2_record = AvgMeter()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print("lr", lr)
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            if config['out'] == 1:
                P2 = model(images)
                loss_P2 = structure_loss(P2, gts)
                loss = loss_P2
            elif config['out'] == 2:
                P1, P2 = model(images)
                loss_P1 = structure_loss(P1, gts)
                loss_P2 = structure_loss(P2, gts)
                loss = loss_P1 + loss_P2
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    global dict_plot

    testpath = './dataset/TestDataset/'
    avg_dice = 0
    if (epoch + 1) % 1 == 0:
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice = test(model, testpath, dataset, 352)
            avg_dice += dataset_dice
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)
        if avg_dice > best:
            best = avg_dice
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{} {}'.format(best, best / 5))
            torch.save(model.state_dict(), save_path + 'best.pth')


if __name__ == '__main__':
    dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    model_name = config['model']
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=100, help='epoch number')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation', default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int, default=4, help='training batch size') # 显存不够，调为4

    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')

    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str, default='./dataset/TrainDataset/', help='path to train dataset')

    parser.add_argument('--test_path', type=str, default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str, default='./model_pth/' + model_name + '/')

    parser.add_argument('--log_save', type=str, default='./log/')
    opt = parser.parse_args()
    if not os.path.exists(opt.log_save):
        os.makedirs(opt.log_save)
    logging.basicConfig(filename=opt.log_save + str(model_name) + '.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    if model_name == 'UNet':
        model = Unet(3, 1).cuda()
    elif model_name == 'MIA-Net':
        model = backbone(isLarge=True).cuda()
    elif model_name == 'Polyp_PVT':
        model = PolypPVT().cuda()
    best = 0
    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.Adam(params, lr=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader1(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)

