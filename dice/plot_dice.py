import os
import csv
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

def dice(input, target):
    smooth = 0.001
    input_flat = np.reshape(input, (-1))
    target_flat = np.reshape(target, (-1))
    intersection = (input_flat * target_flat)
    dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
    dice = '{:.4f}'.format(dice)
    dice = float(dice)
    return dice

def model_dice(dir_label, dir_mask, Thresholds):
    dir_label = dir_label
    dir_mask = dir_mask
    list_mask = os.listdir(dir_mask)
    print()
    num = len(list_mask)
    dice_list100 = np.zeros((num, 10))
    #print(list_mask)
    for n, p in enumerate(list_mask):
        label = np.array(Image.open(os.path.join(dir_label, p)).convert('L'))
        label = label / 255
        mask = np.array(Image.open(os.path.join(dir_mask, p)))
        mask = mask / 255
        #print(n, p, mask.shape, label.shape)
        for i, th in enumerate(Thresholds):
            Bi_pred = np.zeros_like(mask)
            Bi_pred[mask > th] = 1
            Bi_pred[mask < th] = 0
            d = dice(Bi_pred, label)
            dice_list100[n][i] = d
    return list(np.mean(dice_list100, 0))

if __name__ == "__main__":
    Thresholds = np.linspace(0.99, 0, 10)[::-1]  # 分段计算阈值
    # 搞一个csv文件写一下
    dataset = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    model = ['MDCA_Net', 'MIA_Net','PolypPVT_old', 'segformer', 'EU-Net', 'SANet', 'ccba', 'acs', 'HarDMSEG', 'PraNet','Att_Unet', 'Unet_plus','segnet' ,'unet_my']
    all_list = []  #
    dir_label = r'E:\Polyp-PVT-main\dataset\TestDataset\ETIS-LaribPolypDB\masks'
    for m in model:
        path = os.path.join(r'D:\新建文件夹 (2)\Desktop\polyp\map', m, 'ETIS-LaribPolypDB')
        print(path)
        ours = model_dice(dir_label, path, Thresholds)
        print(ours)
        all_list.append(ours)
        # break
    print(all_list)
    plt.title('ETIS')
    # plt.plot(Thresholds, all_list[0], color='green', linestyle='--', label='MDCA-Net')
    plt.plot(Thresholds, all_list[1], color='red', linestyle='--', label='MIA-Net')
    plt.plot(Thresholds, all_list[2], color='black', label='Polyp_PVT')
    plt.plot(Thresholds, all_list[3], color='aqua', label='SegFormer')
    plt.plot(Thresholds, all_list[4], color='blue', label='SANet')
    plt.plot(Thresholds, all_list[5], color='bisque', label='CCBANet')
    plt.plot(Thresholds, all_list[6], color='burlywood', label='ACSNet')
    plt.plot(Thresholds, all_list[7], color='darkblue', label='HarDMSEG')
    plt.plot(Thresholds, all_list[8], color='darkgreen', label='PraNet')
    plt.plot(Thresholds, all_list[9], color='chocolate', label='Att_UNet')
    plt.plot(Thresholds, all_list[10], color='brown', label='U-Net++')
    plt.plot(Thresholds, all_list[11], color='coral', label='SegNet')
    plt.plot(Thresholds, all_list[12], color='antiquewhite', label='U-Net')
    plt.legend()  # 显示图例

    plt.xlabel('阈值')
    plt.ylabel('Dice')
    # plt.show()
    plt.savefig('zIdea1ETIS.jpg', dpi=750)
    with open("./Idea1ETIS.csv", "a", newline='', encoding='GBK') as f:
        writer = csv.writer(f, delimiter=',')
        # 对于每一行的，将这一行的每个元素分别写在对应的列中
        for i in all_list:
            writer.writerow(i)

