import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from utils.dataloader import test_dataset1
import cv2
import numpy as np
import cv2 as cv
from backbone.U_Net import Unet
import matplotlib.pyplot as plt
from config import config

def blur_demo(image):
    # 均值模糊
    blur = cv.blur(image, (5, 5))
    # cv.imshow("lena_blur", blur)
    return blur

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    opt = parser.parse_args()
    if config['model'] == 'UNet':
        model = Unet(3,1)
    pth_path = './model_pth/{}/best.pth'.format(config['model'])
    model.load_state_dict(torch.load(pth_path))
    model.cuda()
    model.eval()
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        d = []
        DSC = 0
        data_path = './dataset/TestDataset/{}'.format(_data_name)
        save_path = './result_map/UNet/{}/'.format(_data_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset1(image_root, gt_root, 352)
        dice_list = []

        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            if config['out'] == 1:
                P1 = model(image)
            elif config['out'] == 2:
                P1, _ = model(image)
            res = F.upsample(P1 , size=gt.shape, mode='bilinear', align_corners=False)
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
            cv2.imwrite(save_path+name, res*255)
        print(np.mean(dice_list))
        print(dice_list)
        print(_data_name, 'Finish!')
