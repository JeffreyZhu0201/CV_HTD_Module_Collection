import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from scipy.io import loadmat, savemat
from DropBlock import DropBlock1D

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

z_real = loadmat('./data/gen_target_15.mat')
hyp_img = loadmat('./data/AV_selected_back.mat')
data1 = loadmat('./data/AVIRIS.mat')
data = hyp_img['selected_back']
min_value = np.min(data)
max_value = np.max(data)
data = (data - min_value) / (max_value - min_value)
d = data1['d']
d = np.array(d)
d = 2*((d-d.min()) /(d.max()-d.min()))
[num, bands]=data.shape
z_dim = 15
batch_size = num
n_epochs = 100
learning_rate = 1e-3
learning_rate_discriminator = 1e-4
beta1 = 0.8
results_path = './resultAV_bg/'

# PyTorch Dataset
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

# Networks
class Encoder(nn.Module):
    def __init__(self, input_dim, n_l1=500, n_l2=500, z_dim=15):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_l1)
        self.drop1 = DropBlock1D()
        self.fc2 = nn.Linear(n_l1, n_l2)
        self.drop2 = DropBlock1D()
        self.fc3 = nn.Linear(n_l2, z_dim)
        self.bn1 = nn.BatchNorm1d(n_l1)
        self.bn2 = nn.BatchNorm1d(n_l2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.drop1(x.unsqueeze(1)).squeeze(1)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = self.drop2(x.unsqueeze(1)).squeeze(1)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, z_dim, n_l2=500, n_l1=500, output_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, n_l2)
        self.drop1 = DropBlock1D()
        self.fc2 = nn.Linear(n_l2, n_l1)
        self.drop2 = DropBlock1D()
        self.fc3 = nn.Linear(n_l1, output_dim)
        self.bn1 = nn.BatchNorm1d(n_l2)
        self.bn2 = nn.BatchNorm1d(n_l1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.drop1(x.unsqueeze(1)).squeeze(1)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = self.drop2(x.unsqueeze(1)).squeeze(1)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, z_dim, n_l1=500, n_l2=500):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, n_l1)
        self.bn1 = nn.BatchNorm1d(n_l1)
        self.fc2 = nn.Linear(n_l1, n_l2)
        self.bn2 = nn.BatchNorm1d(n_l2)
        self.fc3 = nn.Linear(n_l2, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

class Adversary(nn.Module):
    def __init__(self, input_dim, n_l1=500, n_l2=500):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n_l1)
        self.bn1 = nn.BatchNorm1d(n_l1)
        self.fc2 = nn.Linear(n_l1, n_l2)
        self.bn2 = nn.BatchNorm1d(n_l2)
        self.fc3 = nn.Linear(n_l2, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

def autoencoder_Loss(x, d_output, d):
    # x, d_output: [batch, bands], d: [bands, 1]
    # SAM loss
    d = torch.tensor(d, dtype=torch.float32, device=x.device)
    d = d.squeeze()
    A = torch.sum(d_output * d, dim=1)
    B = torch.norm(d_output, p=2, dim=1)
    C = torch.norm(d, p=2)
    defen = A / (B * C + 1e-5)
    # 取top20
    s, _ = torch.topk(defen, k=min(20, defen.shape[0]))
    sam_loss = torch.mean(s)
    mse_loss = torch.mean((d_output - x) ** 2, dim=1)
    encoder_loss = mse_loss + 0.1 * sam_loss
    return encoder_loss.mean()

def form_results():
    saved_model_path = results_path  + '/Saved_models/'
    encoder_path = results_path  + '/encoder/'
    decoder_path = results_path  + '/decoder/'
    for p in [results_path, saved_model_path, encoder_path, decoder_path]:
        if not os.path.exists(p):
            os.makedirs(p)
    return saved_model_path, encoder_path, decoder_path

def train(train_model=True):
    input_dim = bands
    n_l1 = 500
    n_l2 = 500
    z_dim = 15
    encoder = Encoder(input_dim, n_l1, n_l2, z_dim).to(device)
    decoder = Decoder(z_dim, n_l2, n_l1, input_dim).to(device)
    discriminator = Discriminator(z_dim, n_l1, n_l2).to(device)
    adversary = Adversary(input_dim, n_l1, n_l2).to(device)

    optimizer_auto = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_discriminator, betas=(beta1, 0.999))
    optimizer_gen = torch.optim.Adam(encoder.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_adv = torch.optim.Adam(adversary.parameters(), lr=learning_rate_discriminator, betas=(beta1, 0.999))
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    dataset = SimpleDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    z_real_dist = z_real['target_15']
    min_value = np.min(z_real_dist)
    max_value = np.max(z_real_dist)
    z_real_dist = (z_real_dist - min_value) / (max_value - min_value)
    z_real_dist = torch.tensor(z_real_dist, dtype=torch.float32, device=device)

    d_vec = d
    saved_model_path, encoder_path, decoder_path = form_results()

    for epoch in range(n_epochs+1):
        print(f"------------------Epoch {epoch}/{n_epochs}------------------")
        for i, batch_x in enumerate(dataloader):
            batch_x = batch_x.to(device)
            # Forward
            z = encoder(batch_x)
            x_recon = decoder(z)
            d_real = discriminator(z_real_dist)
            d_fake = discriminator(z.detach())
            x_real = adversary(batch_x)
            x_fake = adversary(x_recon.detach())

            # Losses
            auto_loss = autoencoder_Loss(batch_x, x_recon, d_vec)
            dc_loss_real = F.binary_cross_entropy(d_real, torch.ones_like(d_real))
            dc_loss_fake = F.binary_cross_entropy(d_fake, torch.zeros_like(d_fake))
            dc_loss = 0.5 * (dc_loss_fake + dc_loss_real)
            gen_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))
            adv_loss_real = F.binary_cross_entropy(x_real, torch.ones_like(x_real))
            adv_loss_fake = F.binary_cross_entropy(x_fake, torch.zeros_like(x_fake))
            adv_loss = 0.5 * (adv_loss_fake + adv_loss_real)
            dec_loss = F.binary_cross_entropy(adversary(x_recon), torch.ones_like(adversary(x_recon)))

            # Backward
            optimizer_auto.zero_grad()
            auto_loss.backward(retain_graph=True)
            optimizer_auto.step()

            optimizer_disc.zero_grad()
            dc_loss.backward(retain_graph=True)
            optimizer_disc.step()

            optimizer_gen.zero_grad()
            gen_loss.backward(retain_graph=True)
            optimizer_gen.step()

            optimizer_adv.zero_grad()
            adv_loss.backward(retain_graph=True)
            optimizer_adv.step()

            optimizer_dec.zero_grad()
            dec_loss.backward()
            optimizer_dec.step()

            if i % 1 == 0:
                print(f"Epoch: {epoch}, iteration: {i}")
                print(f"Encoder_Loss: {auto_loss.item()}")
                print(f"D1_Loss: {dc_loss.item()}")
                print(f"G2_Loss: {gen_loss.item()}")
                print(f"D2_Loss: {adv_loss.item()}")
                print(f"Decoder_Loss: {dec_loss.item()}")

            if epoch % 100 == 0:
                # 保存mat文件
                e_output = z.detach().cpu().numpy()
                d_output = x_recon.detach().cpu().numpy()
                savemat(encoder_path + f'x_encoder{epoch}.mat', {'x_encoder': e_output})
                savemat(decoder_path + f'x_decoder{epoch}.mat', {'x_decoder': d_output})

if __name__ == '__main__':
    train(train_model=True)