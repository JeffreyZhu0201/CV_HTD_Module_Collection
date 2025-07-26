import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 假设 loss_history 文件为 CSV 格式
# 如果是 txt 或其他格式，请根据实际情况调整
df = pd.read_csv('./loss_history_saved.csv')

# 假设有 'epoch' 和 'loss' 两列
plt.figure(figsize=(8, 5))
# plt.plot(df['epoch'], df['loss'], label='Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
plt.plot(np.arange(len(df['train_loss'])), df['train_loss'], label='Train Loss', color='red', linestyle='-', linewidth=1)
plt.plot(np.arange(len(df['test_loss'])), df['test_loss'], label='Test Loss', color='orange', linestyle='-', linewidth=1)

plt.title('Loss History')
plt.legend()
plt.grid(True)
plt.show()

plt.twinx()
plt.plot(np.arange(len(df['accuracy'])), df['accuracy'], label='Accuracy', color='blue', linestyle='--', linewidth=1)
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()