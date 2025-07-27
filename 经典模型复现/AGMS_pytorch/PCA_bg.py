import torch
import numpy as np
import scipy.io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 加载数据
data = scipy.io.loadmat('./data/AVIRIS.mat')
# 提取数据
prior_target = data['S']
prior_target = np.transpose(prior_target, (1, 0))
min_value = np.min(prior_target)
max_value = np.max(prior_target)
prior_target = (prior_target - min_value) / (max_value - min_value)
image = data['X']
[row, col, bands] = image.shape
image = np.reshape(image, [row * col, bands])

# 转换为PyTorch张量
prior_target = torch.tensor(prior_target, dtype=torch.float64)
image = torch.tensor(image, dtype=torch.float64)

# 计算 prior_target 的平均值
prior_target_mean = torch.mean(prior_target, dim=0, keepdim=True)
# 计算余弦相似度矩阵
similarity = torch.matmul(prior_target_mean, image.t()).squeeze(0)
# 取相似度最高的三个样本的索引
indices = torch.argsort(similarity, descending=True)[:3]

# 获取这三个样本
gathered_samples = image[indices]
# 合并为一个张量
back_image = torch.stack([gathered_samples[i] for i in range(3)], dim=0).numpy()

# 保存结果
scipy.io.savemat('./data/AVIRIS_bg.mat', {'similar_images': back_image})

def main():
    # 加载 .mat 文件
    data = scipy.io.loadmat('./data/AVIRIS_bg.mat')['similar_images']

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
    if choice == 1:
        data_tensor = torch.tensor(data, dtype=torch.float32)
        cov_matrix = torch.matmul(data_tensor.t(), data_tensor) / data_tensor.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        total_variance = torch.sum(eigenvalues)
        var_threshold = fraction * total_variance
        var_sum = 0.0
        num_components = 0
        for eigenvalue in reversed(eigenvalues):
            var_sum += eigenvalue
            num_components += 1
            if var_sum >= var_threshold:
                break
        selected_eigenvectors = eigenvectors[:, -num_components:]
        img_pc = torch.matmul(data_tensor, selected_eigenvectors).numpy()
        print("PCA_DIM", num_components)
        return img_pc, num_components

    if choice == 2:
        data_tensor = torch.tensor(data, dtype=torch.float32)
        covariance_matrix = torch.matmul(data_tensor.t(), data_tensor)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        selected_eigenvectors = eigenvectors[:, -fraction:]
        img_pc = torch.matmul(data_tensor, selected_eigenvectors).numpy()
        print("PCA_DIM", fraction)
        print(img_pc.shape)
        scipy.io.savemat('./data/AVIRIS_bg.mat', {'backnew': img_pc})
        return img_pc, fraction

if __name__ == "__main__":
    main()
