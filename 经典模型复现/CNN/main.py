
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
from model.CNN import CNN
from dataloader.SatelliteDataset import SatelliteDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# Path to the flattened dataset
flattenDataDirTrain = "CNN/overhead/train.csv"
flattenDataDirTest = "CNN/overhead/test.csv"

# 加载数据
train_dataset = pd.read_csv(flattenDataDirTrain).to_numpy()
test_dataset = pd.read_csv(flattenDataDirTest).to_numpy()
# print(train_dataset[0:10].to_numpy())
# print(train_dataset[:][1:])

# 将 n*724 的数据转为 n*28*28
train_images = train_dataset[:,1:].reshape(-1, 1,28, 28).astype(np.float32)
train_labels = train_dataset[:,0]
print(train_images.shape)
print(train_labels.shape)

test_images = test_dataset[:, 1:].reshape(-1,1,28,28).astype(np.float32)
test_labels = test_dataset[:,0]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

saved_data = {
    "trainLossPerEpoch":[],
    "testLossPerEpoch":[],
    "acc":[]
}

def main():
    loss_array = []
    accuracy_array = []
    satelliteDatasetTrain = SatelliteDataset(train_images, train_labels)
    satelliteDatasetTest = SatelliteDataset(test_images,test_labels)

    batch_size=32
    TrainLoader = DataLoader(satelliteDatasetTrain, batch_size=batch_size, shuffle=True)
    TestLoader = DataLoader(satelliteDatasetTest,batch_size=batch_size,shuffle=False)

    model = CNN(input_channels=1, num_classes=10,batch_size=batch_size)
    model.to(device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    for epoch in range(40):
        train_loss = 0
        train_total = 0
        model.train()
        for batch_index, batch in enumerate(tqdm(TrainLoader, desc=f"Epoch {epoch+1}/20")):
            images = batch[0].to(device)
            label = batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, label)
            loss.backward()
            train_total+= len(batch)
            train_loss += loss.item()*len(batch)
            optimizer.step()
        saved_data["trainLossPerEpoch"].append(train_loss/train_total)
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item():.4f}")
        # 验证集评估
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            # 使用tqdm可视化验证集进度
            for batch_index, bacth in enumerate(tqdm(TestLoader, desc=f"Validation {epoch+1}/20")):
                val_images = bacth[0].to(device)
                val_labels = bacth[1].to(device)
                val_outputs = model(val_images)
                v_loss = criterion(val_outputs, val_labels)
                batch_size_val = bacth[0].size(0)
                val_loss += v_loss.item() * batch_size_val
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += batch_size_val
                val_correct += (val_predicted == val_labels).sum().item()

        val_loss /= val_total
        saved_data["testLossPerEpoch"].append(val_loss)

        val_accuracy = val_correct / val_total
        saved_data["acc"].append(val_accuracy)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}")
        
        torch.save(model.state_dict(), "./CNN/cnn_model.pth")

    # model = torch.load(model.state_dict(),"cnn_model.pth")

    pd.DataFrame(saved_data).to_csv("./CNN/loss_history.csv",index=False)

if __name__ == "__main__":
    main()
    