import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
# from lib.pvt import PolypPVT
from backbone.MIA_Net import backbone
from backbone.U_Net import Unet
from utils.dataloader import test_dataset1
import cv2
import numpy as np
import cv2 as cv
from config import config
import matplotlib.pyplot as plt

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
        model = Unet(3, 1)
    pth_path = '../model_pth/{}/best.pth'.format(config['model'])
    model.load_state_dict(torch.load(pth_path))
    model.cuda()
    model.eval()
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        data_path = '../dataset/TestDataset/{}'.format(_data_name)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset1(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            if config['out'] == 1:
                _, hotmap = model(image, isHotMap=True)
            elif config['out'] == 2:
                _, _, hotmap = model(image, isHotMap=True)
            x_visualize = hotmap.cpu().detach().numpy()  # 用Numpy处理返回的[1,256,513,513]特征图
            x_visualize = np.max(x_visualize, axis=1).reshape(352, 352)  # shape为[513,513]，二维
            x_visualize = (((x_visualize - np.min(x_visualize)) / (np.max(x_visualize) - np.min(x_visualize))) * 255).astype(np.uint8)  # 归一化并映射到0-255的整数，方便伪彩色化
            x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_JET)  # 伪彩色处理
            path = os.path.join('./UNet', _data_name)
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(os.path.join(path, name + '.jpg'), x_visualize)  # 保存可视化


