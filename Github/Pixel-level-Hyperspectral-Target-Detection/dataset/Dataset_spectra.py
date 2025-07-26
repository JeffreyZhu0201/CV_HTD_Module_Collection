import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as spo
import cv2
from dataset.comparison_methods import classic_detectors
import copy
from .comparison_methods import *

class HTD_dataset(Dataset):
    def __init__(self, img_dir, img_name, img_refer=[], eo=False, prior_transform=['dialation'], divide=1):
        # 初始化，设置先验变换方式、参考图像等
        self.prior_transform = prior_transform
        self.img_refer = img_refer
        self.refer_spectra = []
        # 解析场景，获取行列数
        row, col = self.parse_inscene(img_dir, img_name, divide)
        # 对高光谱数据按像素归一化
        self.img = self.img / np.linalg.norm(self.img, 2, axis=1, keepdims=True)
        # 将高光谱数据重塑为三维（行，列，波段）
        self.test_img = self.img.reshape(row, col, -1, order='F')
        # 如果需要，计算经典检测器的结果
        if eo:
            classic_results = classic_detectors(copy.deepcopy(self.img), copy.deepcopy(self.prior), row, col)
            self.classic_results = classic_results
        # 背景端元，保存归一化后的高光谱数据
        self.background_end = self.img

    def parse_inscene(self, img_dir, img_name, proportion):
        # 解析场景，读取.mat文件
        img_path = os.path.join(img_dir, img_name + '.mat')
        data = spo.loadmat(img_path)
        # 读取并转置高光谱数据
        self.img = data['img'].T
        # 对每个像素进行归一化
        self.img = self.img / np.linalg.norm(self.img, 2, axis=1)[:, np.newaxis]    # [n,c,1]
        # 读取地面真值
        self.groundtruth = data['groundtruth']
        row, col = self.groundtruth.shape[0], self.groundtruth.shape[1]
        # 根据先验变换方式选择不同的先验生成方式
        if 'part' in self.prior_transform:
            print('mode 1: Select proportion:1/{} of target pixels.'.format(proportion))
            self.parse_refer(img_dir, img_name, proportion=proportion)
            self.prior = np.stack(self.refer_spectra).mean(axis=0)
        elif 'MT' in self.prior_transform:
            print('mode 2: Use prior from different HSI for current HSI. Reference HSIs: {}'.format(self.img_refer))
            for img_ in self.img_refer:
                self.parse_refer(img_dir, img_, proportion=proportion)
            self.prior = np.stack(self.refer_spectra).mean(axis=0)
        else:
            print('mode 3: Averge all the pixels for prior.')
            # 默认：所有目标像素的平均光谱作为先验
            self.prior = self.img[self.groundtruth.reshape(-1, order='F') > 0].mean(axis=0)
        # 对先验光谱归一化
        self.prior /= np.linalg.norm(self.prior, 2).clip(1e-10,None)
        return row, col
        
    def parse_refer(self, img_dir, img_name, proportion=1):
        # 解析参考图像，生成参考光谱
        img_path = os.path.join(img_dir, img_name + '.mat')
        data = spo.loadmat(img_path)
        img = data['img'].T
        groundtruth = data['groundtruth']
        # 对每个像素进行归一化
        img = img / np.linalg.norm(img, 2, axis=1, keepdims=True)

        # 连通域分析，获取目标区域质心
        num, label,_, centroid = cv2.connectedComponentsWithStats(groundtruth)
        if len(centroid) > 1:
            centroid = centroid[::proportion]
        # 将groundtruth裁剪为0（防止异常）
        groundtruth = groundtruth.clip(0,0)
        # 将label每个元素变成一行
        label = [i[np.newaxis] for i in label]
        gt = np.concatenate(label)
        # 在质心位置赋值
        for i, c in enumerate(centroid):
            gt[int(c[1]), int(c[0])] = 2 * i
            groundtruth[int(c[1]), int(c[0])] = 1

        # 计算参考光谱（目标像素的平均光谱）
        prior_spectrum = (img)[groundtruth.reshape(-1, order='F')>0].mean(axis=0)
        self.refer_spectra.append(prior_spectrum)

    def __len__(self):
        # 返回数据集长度（像素数）
        return self.background_end.shape[0]
    
    def __getitem__(self, idx):
        # 获取单个像素的光谱及伪目标光谱
        spectra = self.background_end[idx]
        spectra = spectra / np.linalg.norm(spectra, 2)      
        norm = np.linalg.norm(spectra, 2)
        # 生成伪目标光谱
        pseudo_target = norm * 1 * self.prior / np.linalg.norm(self.prior, 2)
        pseudo_target = pseudo_target / np.linalg.norm(pseudo_target, 2)
        return torch.Tensor(pseudo_target).cuda(), torch.Tensor(spectra).cuda()