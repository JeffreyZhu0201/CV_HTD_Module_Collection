
import os
import numpy as np
from torch.utils.data import Dataset

# class SatelliteDataset(Dataset):
#     def __init__(self, image_dir, label_file, transform=None):
#         self.image_dir = image_dir
#         self.label_file = label_file
#         self.transform = transform
#         self.images = os.listdir(image_dir)
#         self.labels = np.loadtxt(label_file)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.images[idx])
#         image = np.load(image_path)
#         label = self.labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, label

class SatelliteDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        assert len(images) == len(labels), "Images and labels must have the same length"
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label
