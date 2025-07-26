import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


 
mnist = datasets.MNIST(
    root='./others/',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
)
 
dataloader = DataLoader(
    dataset=mnist,
    batch_size=64,
    shuffle=True
)
 
def gen_img_plot(model, epoch, text_input):
    prediction = np.squeeze(model(text_input).detach().cpu().numpy()[:16])
    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow((prediction[i] + 1) / 2)
        plt.axis('off')
    plt.show()
 
# 生成器定义
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
 
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers
 
        self.mean = nn.Sequential(
            *block(100, 256, normalize=False),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )
 
    def forward(self, x):
        imgs = self.mean(x)
        imgs = imgs.view(-1, 1, 28, 28)
        return imgs
 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.mean = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        img = self.mean(x)  # 对 64条数据的每一条都进行模型运算
        return img
 
# 实例化
generator = Generator()
discriminator = Discriminator()
 
# 定义优化器
G_Apim = torch.optim.Adam(generator.parameters(), lr=0.0001)
D_Apim = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
 
# 损失函数
criterion = torch.nn.BCELoss()    # 对应 Sigmoid，计算二元交叉墒损失
 
 
epoch_num = 100
G_loss_save = []
D_loss_save = []
for epoch in range(epoch_num):  # 将 10000 条数据迭代了两遍
    G_epoch_loss = 0
    D_epoch_loss = 0
    count = len(dataloader)
    for i, (img, _) in enumerate(dataloader):   # 内层迭代次数为 10000 // 64 = 157次，每次 64个数据
        # 训练 Discriminator
        # 判断出假的
        size = img.size(0)  # 0 维有多少个数据
        fake_img = torch.randn(size, 100)
 
        output_fake = generator(fake_img)
        fake_socre = discriminator(output_fake.detach())    # .detach() 返回一个关闭梯度的 output_fake，这样前向传播不会修改 generater 的 grad
        D_fake_loss = criterion(fake_socre, torch.zeros_like(fake_socre))
        # 判断出真的
        real_socre = discriminator(img)
        D_real_loss = criterion(real_socre, torch.ones_like(real_socre))
 
        D_loss = D_fake_loss + D_real_loss
        D_Apim.zero_grad()
        D_loss.backward()
        D_Apim.step()
 
        # 训练 Generater
        # G_fake_img = torch.randn(size, 100)
        # G_output_fake = generator(G_fake_img)
        # fake_G_socre = discriminator(G_output_fake)
        fake_G_socre = discriminator(output_fake)
        G_fake_loss = criterion(fake_G_socre, torch.ones_like(fake_G_socre))
        G_Apim.zero_grad()
        G_fake_loss.backward()
        G_Apim.step()
 
        with torch.no_grad():   # 其中所有的 requires_grad 都被默认设置为 False
            G_epoch_loss += G_fake_loss
            D_epoch_loss += D_loss
 
    with torch.no_grad():
        G_epoch_loss /= count
        D_epoch_loss /= count
 
        G_loss_save.append(G_epoch_loss.item())
        D_loss_save.append(D_epoch_loss.item())
 
        print('Epoch: [%d/%d] | G_loss: %.3f | D_loss: %.3f'
              % (epoch, epoch_num, G_epoch_loss, D_epoch_loss))
        text_input = torch.randn(64, 100)
        gen_img_plot(generator, epoch, text_input)
 
 
x = [epoch + 1 for epoch in range(epoch_num)]
plt.figure()
plt.plot(x, G_loss_save, 'r')
plt.plot(x, D_loss_save, 'b')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['G_loss','D_loss'])
plt.show()