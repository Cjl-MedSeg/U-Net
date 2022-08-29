import os
import cv2
import numpy as np
import mindspore.dataset as ds
import glob
import albumentations as A


def train_transforms(img_size):
    return A.Compose([
            # A.RandomResizedCrop(img_size, img_size),
            A.Resize(img_size, img_size),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.ShiftScaleRotate(p=0.25),
            A.RandomRotate90(p=0.25),
            ], p=1.)

def val_transforms(img_size):
    return A.Compose([
            A.Resize(img_size, img_size),
            ], p=1.)


class Data_Loader:
    def __init__(self, data_path, image_size=None, aug=True):
        # 初始化函数，读取所有data_path下的图片
        self.image_size = image_size
        self.aug = aug
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.label_path = glob.glob(os.path.join(data_path, 'mask/*.png'))
        self.train_aug = train_transforms(image_size)
        self.test_aug = val_transforms(image_size)

    def __getitem__(self, index):
        # 根据index读取图片
        image = cv2.imread(self.imgs_path[index])
        label = cv2.imread(self.label_path[index], cv2.IMREAD_GRAYSCALE)

        if self.aug:
            augments = self.train_aug(image=image, mask=label)
            image, label = augments['image'], augments['mask']
        else:
            augments = self.test_aug(image=image, mask=label)
            image, label = augments['image'], augments['mask']

        image = np.transpose(image, (2, 0, 1))
        label = np.reshape(label, (1, self.image_size, self.image_size))

        image = image / 255.
        label = label / 255.

        return image.astype("float32"), label.astype("float32")

    @property
    def column_names(self):
        column_names = ['image', 'label']
        return column_names

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


def create_dataset(data_dir, img_size, batch_size, augment, shuffle):
    mc_dataset = Data_Loader(data_path=data_dir, image_size=img_size, aug=augment)
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=shuffle)
    dataset = dataset.batch(batch_size, num_parallel_workers=1)
    if augment==True and shuffle==True:
        print("训练集数据量：", len(mc_dataset))
    elif augment==False and shuffle==False:
        print("验证集数据量：", len(mc_dataset))
    else:
        pass

    return dataset


