import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch

class BuildingDataset(Dataset):
    def __init__(self,root:str,train:bool,transforms=None):
        super(BuildingDataset,self).__init__()
        self.transforms = transforms
        self.img = []
        self.mask = []
        
        if train == True:
            train_data = []
            train_label = []
                    # 读取图片数据 (RGB, 3通道)# root = /massachusetts-buildings-dataset/png/
            for dirname, _, filenames in os.walk(f'{root}/train'):
                for filename in filenames:
                    img = Image.open(os.path.join(dirname, filename)).convert('RGB')
                    # arr = np.transpose(np.array(img), (2, 0, 1))  # (3, H, W)
                    train_data.append(img)

            # 读取标签 (灰度, 1通道)
            for dirname, _, filenames in os.walk(f'{root}/train_labels'):
                for filename in filenames:
                    mask = Image.open(os.path.join(dirname, filename)).convert('L')  # 灰度
                    
                    # arr = np.array(mask)[np.newaxis, ...]  # (1, H, W)
                    train_label.append(mask)
            self.img = train_data
            self.mask = train_label
        else:
            val_data = []
            val_label = []
            for dirname, _, filenames in os.walk(f'{root}/val'):
                for filename in filenames:
                    img = Image.open(os.path.join(dirname, filename)).convert('RGB')
                    # arr = np.transpose(, (2, 0, 1))
                    val_data.append(img)

            for dirname, _, filenames in os.walk(f'{root}/val_labels'):
                for filename in filenames:
                    mask = Image.open(os.path.join(dirname, filename)).convert('L')
                    # arr = np.array(mask)[...,np.newaxis]
                    val_label.append(mask)
            self.img = val_data
            self.mask = val_label

        self.img = self.split_patches(self.img,is_img=True,patch_size=300)
        self.mask = self.split_patches(self.mask,is_img=False,patch_size=300)    

    def __getitem__(self,idx):
        img,mask = self.img[idx],self.mask[idx]
        img = np.array(img)
        mask = np.array(mask)
        if self.transforms is not None:
            img,mask = self.transforms(img, mask)
        return img,mask
    
    def __len__(self):
        return len(self.img)
    
    def split_patches(self,data_list, is_img=True ,patch_size=300):
        patches = []
        if is_img == True:
            print("True")
            for arr in data_list:
                arr = np.array(arr)
                print(arr.shape)
                c, h, w = arr.shape
                for i in range(0, h, patch_size):
                    for j in range(0, w, patch_size):
                        patch = arr[i:i+patch_size, j:j+patch_size,:]
                        # 只保留完整patch
                        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                            patches.append(Image.fromarray(np.array(patch)))
                            print("added img: ",np.array(patch).shape)
        else:
            for arr in data_list:
                arr = np.array(arr)
                h, w = arr.shape
                for i in range(0, h, patch_size):
                    for j in range(0, w, patch_size):
                        patch = arr[i:i+patch_size, j:j+patch_size]
                        # 只保留完整patch
                        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                            patches.append(Image.fromarray(np.array(patch)))
                            # print("added mask: ",np.array(patch).shape)
        
        return patches

    @staticmethod
    def collate_fn(batch):
        return torch.stack(batch, 0)

class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # check files
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB') # [h,w,c]
        manual = Image.open(self.manual[idx]).convert('L')  # [h,w]
        manual = np.array(manual) / 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        roi_mask = 255 - np.array(roi_mask)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        print("mask:",mask)
        print("mask shape: ",mask.shape)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        # batched imgs shape:  torch.Size([1, 3, 584, 565])
        # batched_targets shape:  torch.Size([1, 584, 565])
        print("batched imgs shape: ",batched_imgs.shape)
        print("batched_targets shape: ",batched_targets.shape)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
        # print("batched_imgs",batched_imgs)
        
    return batched_imgs

if __name__ == '__main__':
    from train import SegmentationPresetTrain
    train_dataset = BuildingDataset(r'D:\Code\小项目\unet-biuding\massachusetts-buildings-dataset\versions\2\png',
                                   train=True,
                                   transforms=SegmentationPresetTrain(565, 480, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    print(len(train_dataset))
    