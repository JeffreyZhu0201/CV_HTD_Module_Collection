import matplotlib.pyplot as plt
import pandas as pd

# 读取损失数据并用matplotlib显示
loss_history = pd.read_csv("./CNN/loss_history.csv")
plt.figure(figsize=(8, 5))
plt.plot(loss_history["trainLossPerEpoch"], label="Train Loss")
plt.plot(loss_history["testLossPerEpoch"], label="test Loss")
plt.plot(loss_history["acc"], label="accuraccy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss and Accuracy")
plt.legend()
plt.grid(True)
plt.show()
