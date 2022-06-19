from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import metrics
from dataset import dualctdataset,RandomGenerator1
from skimage.metrics import peak_signal_noise_ratio as psnr
import pydicom
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np
from PIL import Image
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import dualctdataset,RandomGenerator
from torchvision import transforms
import argparse
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
from skimage.metrics import structural_similarity as ssim
import torch

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model import define_D, GANLoss, get_scheduler, update_learning_rate
from matplotlib import pyplot as plt
import ex_transforms


if __name__ == '__main__':
    device = torch.device("cuda:0")

    def denormalise_scan(nrml_scan, mean, std, min_val, max_val):
        denormalised = denormalise_tanh(nrml_scan, min_val, max_val)
        return denormalise_zero_mean_unit_var(denormalised, mean, std)

    def denormalise_zero_mean_unit_var(arr, mean, std):
        return arr * std + mean

    def normalise_tanh(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        return 2 * ((arr - np.min(arr)) / (np.max(arr) - np.min(arr))) - 1, min_val, max_val

    def denormalise_tanh(arr, min_val, max_val):
        return ((arr + 1) / 2) * (max_val - min_val) + min_val

    def normalizationminmax1(data):
        print('min/max data: {}/{}'.format(np.min(data), np.max(data)))
        data = np.float32(data)
        max = np.max(data)
        min = np.min(data)
        data = data - min
        newmax = np.max(data)
        data = (data - (newmax / 2)) / (newmax / 2.)
        # print('this is the minmax of normalization')
        # print(np.max(data))
        # print(np.min(data))
        return data


    def nonormalization(data):
        mean = data.mean()
        std = data.std()
        return (data - mean) / std, mean, std

    def denormalise_zero_mean_unit_var(arr, mean, std):
        return arr * std + mean

    def normalise_tanh(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        return 2 * ((arr - np.min(arr)) / (np.max(arr) - np.min(arr))) - 1, min_val, max_val

    # center data and normalise to -1 and 1
    def normalise_scan(scan):
        # set data to zero mean and unit variance
        normalised_scan, mean, std = nonormalization(scan)

        # normalise to -1 and 1
        # min_val and max_val are the min and max values of normalised numpy array
        normalised_scan, min_val, max_val = normalise_tanh(normalised_scan)
        return normalised_scan, mean, std, min_val, max_val

    # TODO********************************************************************************
    print('===> Loading model...')
    model_path = "F:\Work\\NPC\\train_results\\proposed\\re_0\\netG_model.pth"
    net_g = torch.load(model_path).to(device)
    print('===> Loading datasets...')
    test_folder = r'F:\Work\NPC\dataset\HECKTOR\test_set_5'
    patients = os.listdir(test_folder)

    avg_dice = 0
    for patient in patients:
        print(patient)
        # patient = "CHUS078"
        file = os.path.join(test_folder, patient)
        # all_paths = [os.path.join(file, s) for s in natsorted(os.listdir(file))]
        all_paths = [os.path.join(test_folder, s) for s in patients]
        testA = all_paths

        val_transform = ex_transforms.Compose([
            ex_transforms.NormalizeIntensity(),
            ex_transforms.ToTensor()
        ])

        test_data = dualctdataset(testA, transform=val_transform)
        test_data_loader = DataLoader(dataset=test_data, num_workers=4, batch_size=1, shuffle=False)

        net_g.eval()
        with torch.no_grad():
            num = 0
            dice1 = 0
            pred_gt = np.zeros((144, 144, 144))
            gt = np.zeros((144, 144, 144))
            for sample in test_data_loader:
                input, target = sample['input'].to(device), sample['target'].to(device)
                id = sample['id']
                # print(id)
                ct = input.cpu().numpy()[0, 0, :, :]
                suv = input.cpu().numpy()[0, 1, :, :]

                prediction = net_g(input)
                pred = prediction.detach().cpu().numpy()
                tar = target.detach().cpu().numpy()
                pred_th = ex_transforms.ProbsToLabels()(pred)

                prediction = torch.from_numpy(pred_th).to(device)
                dice = metrics.dice(prediction, target)
                print("Slice:{}: Dice: {:.4f}".format(num, dice.item()))
                dice1 = dice+dice1
            print(dice1 / len(test_data_loader))
            print("********")

                # plt.figure()
                # plt.subplot(121)
                # plt.imshow(pred_th[0, 0, :, :])
                # plt.title('pred')
                # plt.subplot(122)
                # plt.imshow(tar[0, 0, :, :])
                # plt.title('gt')
                # plt.show()

                # pred_gt[:, :, num] = pred_th
                # gt[:, :, num] = tar
                # num = num+1

            pred_gt = torch.from_numpy(pred_gt).to(device)
            pred_gt = pred_gt.unsqueeze(0).unsqueeze(0)
            gt = torch.from_numpy(gt).to(device)
            gt = gt.unsqueeze(0).unsqueeze(0)
            dice_pred = metrics.dice_1(pred_gt, gt)
            print('*'*50)
            print("Name:{}: Dice: {:.4f}".format(patient, dice_pred.item()))

            avg_dice += dice_pred.item()
            # print(avg_dice)

    avg_dice = (avg_dice / len(patients))
    print("*"*100)
    print("average DSC: ", avg_dice)


        # 保存预测结果
        # file = os.path.join(pred_save_folder, str(num))
        # if (num % 5) == 0:
        #     np.savez(file, ct=ct, suv=suv, pred=pred_th, target=tar, id=id)

        # testB1 = r'J:\new32/'
        # ds = pydicom.dcmread(testB1 + path1[idx])
        # ds.PixelData = denormalise_scan2.tobytes()
        # ds.Rows, ds.Columns = denormalise_scan2.shape
        # save_folder_path = 'E:/'
        # ds.save_as(os.path.join(save_folder_path, str(idx)) + '.dcm')

        #   fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
        #       # 设置子图占满整个画布
        #   ax = plt.Axes(fig, [0., 0., 1., 1.])
        #       # 关掉x和y轴的显示
        #   ax.set_axis_off()
        #   fig.add_axes(ax)
        #   ax.imshow(out_img, cmap='gray')
        #   '''
        #   注意，这里加上plt.show()后，保存的图片就为空白了，因为plt.show()之后就会关掉画布，
        #   所以如果要保存加显示图片的话一定要将plt.show()放在plt.savefig(save_path)之后
        #   '''
        # #   save_path = './'
        # #  plt.show()
        # #   plt.savefig(save_path)
        # #   plt.imshow(input, cmap=plt.cm.gray)
        # #   plt.show()
        #   save_folder_path1 = './predict/'
        #   plt.savefig(os.path.join(save_folder_path1, str(idx)) + '.png')

    # print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    # print("===> Avg. SSIM: {:.4f} ".format(avg_ssim / len(testing_data_loader)))
