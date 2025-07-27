'''
根据生成的目标数据集，使用PCA进行降维处理 (PyTorch 版本)
'''
import torch
import numpy as np
from scipy.io import savemat
import scipy.io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main():
    # 加载 .mat 文件
    data = scipy.io.loadmat('./gen_target_AV.mat')['gen_target_AV']
    # 调用 Apply_PCA 函数
    transformed_data, num_components = Apply_PCA(data, fraction=15, choice=2)
    # 打印转换后的数据和保留的主成分数量
    print("Transformed Data:", transformed_data)
    print("Number of Components:", num_components)

def Apply_PCA(data, fraction, choice):
    """
    choice=1
    fraction：保留特征百分比
    choice=2
    fraction：降维后的个数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)

    if choice == 1:
        # 计算协方差矩阵
        cov_matrix = torch.matmul(data_tensor.T, data_tensor) / data_tensor.shape[0]
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        # 按特征值从大到小排序
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        # 保留指定百分比的特征值
        total_variance = torch.sum(eigenvalues)
        var_threshold = fraction * total_variance
        var_sum = 0.0
        num_components = 0
        for val in eigenvalues:
            var_sum += val
            num_components += 1
            if var_sum >= var_threshold:
                break
        # 选择前 num_components 个特征向量
        selected_eigenvectors = eigenvectors[:, :num_components]
        # 将数据转换到主成分空间
        img_pc = torch.matmul(data_tensor, selected_eigenvectors)
        img_pc = img_pc.cpu().numpy()
        print("PCA_DIM", num_components)
        return img_pc, num_components

    if choice == 2:
        # 计算协方差矩阵
        cov_matrix = torch.matmul(data_tensor.T, data_tensor)
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        # 按特征值从大到小排序
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, sorted_indices]
        # 选择前 fraction 个特征向量
        selected_eigenvectors = eigenvectors[:, :fraction]
        # 将数据转换到主成分空间
        img_pc = torch.matmul(data_tensor, selected_eigenvectors)
        img_pc = img_pc.cpu().numpy()
        print("PCA_DIM", fraction)
        print(img_pc.shape)
        savemat('./data/gen_target_15.mat', {'target_15': img_pc})
        return img_pc, fraction

if __name__ == "__main__":
    main()
