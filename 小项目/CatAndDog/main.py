import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler
import random
x_train = []
y_train = []
x_test = []
y_test = []
batch_size = 100
torch.cuda.empty_cache()
for index, j in enumerate(['cat', 'dog']):
    for i in range(4000):
        
        current_img = np.array(cv2.resize(cv2.imread('./dataset/training_set/training_set/{}s/{}.{}.jpg'.format(j, j, i+1)),(256,256)))
        # (h,w,3)
        current_img = np.moveaxis(current_img, -1, 0)
        # (3,h,w)
        # train_img.append([current_img,1])
        # x_train = 
        x_train.append(current_img)
        y_train.append(index)

for index, j in enumerate(['cat', 'dog']):
    for i in range(1000):
        
        current_img = np.array(cv2.resize(cv2.imread('./dataset/test_set/test_set/{}s/{}.{}.jpg'.format(j, j, i+4001)),(256,256)))
        # (h,w,3)
        current_img = np.moveaxis(current_img, -1, 0)
        # (3,h,w)
        # train_img.append([current_img,1])
        # x_train = 
        x_test.append(current_img)
        y_test.append(index)

# x_train = torch.tensor(x_train,dtype=torch.float64)

# x_train = np.array(x_train)
# y_train = np.array(y_train)

# print(x_train.shape)
# print(y_train.shape)
# print(np.array(train_img).shape)

class AnimalDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        # Convert numpy array to float32 tensor and normalize to [0,1]
        x = torch.tensor(self.x[index], dtype=torch.float32) / 255.0
        y = torch.tensor(self.y[index], dtype=torch.long)  # Use long for class indices
        return x, y

train_dataset = AnimalDataset(x_train,y_train)
test_dataset = AnimalDataset(x_test,y_test)

train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

class CatAndDog(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 计算展平后的特征维度
        self.flattened_size = 64 * 64 * 64  # 你用的是256x256输入，池化两次，每次/2
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

model = CatAndDog(batch_size)
num_classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss = []

def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()
    loss = 0
    for batch,(X,y) in enumerate(dataloader):   # 循环每个batch=0.1.2.3...
        X,y = X.to(device),y.to(device)
        pred = model(X)     # 预测
        loss = loss_fn(pred, y)  # 计算损失
        optimizer.zero_grad()   # 梯度清零
        loss.backward()         # 反向传播
        optimizer.step()        # 更新参数

        if batch % 10 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{size:>5d}]")

    train_loss.append(loss.item())

test_loss = []
accuracy = []
def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)    
    num_batches = len(dataloader)
    model.eval()
    loss,correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X)
            loss += loss_fn(pred,y).item() # 计算损失 将一个只包含单个元素的 PyTorch 张量（tensor）转换为对应的 Python 数值（如 float 或 int）。
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # 计算正确预测的数量 argmax(1) 沿着第1维（即每一行）找到最大值的索引，也就是模型预测的类别。
    loss /= num_batches
    correct /= size

    accuracy.append(correct)
    test_loss.append(loss)
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")


epochs = 20

# 训练模型
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    
    print("Saving loss")
    # 保存test_loss和train_loss到同一个csv文件
    pd.DataFrame({
        "train_loss": train_loss,
        "test_loss": test_loss,
        "accuracy": accuracy
    }).to_csv("loss_history_saved.csv", index=False)

    print("Saving model...")
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    print("Saving model done!")
print("Done!")


