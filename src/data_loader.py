import os
import cv2
import mindspore.dataset as ds
import glob
import mindspore.dataset.vision as vision_C
import mindspore.dataset.transforms as C_transforms
import random
import mindspore
from PIL import Image
import numpy as np
from mindspore.dataset.vision import Inter

def train_transforms(img_size):
    return C_transforms.Compose([
    vision_C.Resize(img_size, interpolation=Inter.NEAREST),
    vision_C.Rescale(1./255., 0.0),
    vision_C.RandomHorizontalFlip(prob=0.5),
    vision_C.RandomVerticalFlip(prob=0.5),
    vision_C.HWC2CHW()
    ]
    )

def val_transforms(img_size):
    return C_transforms.Compose([
    vision_C.Resize(img_size, interpolation=Inter.NEAREST),
    vision_C.Rescale(1/255., 0),
    vision_C.HWC2CHW()
    ] )


class Data_Loader:
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        self.label_path = glob.glob(os.path.join(data_path, 'mask/*.png'))

    def __getitem__(self, index):
        # 根据index读取图片
        image = cv2.imread(self.imgs_path[index])
        label = cv2.imread(self.label_path[index], cv2.IMREAD_GRAYSCALE)

        '''
        image = Image.open(self.imgs_path[index])
        label = Image.open(self.label_path[index])
        label = label.convert('1')
        image = np.array(image)
        label = np.array(label)
        label = label.reshape((label.shape[0], label.shape[1], 1))
        '''
        label = label.reshape((label.shape[0], label.shape[1], 1))
        return image, label

    @property
    def column_names(self):
        column_names = ['image', 'label']
        return column_names

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


def create_dataset(data_dir, img_size, batch_size, augment, shuffle):
    mc_dataset = Data_Loader(data_path=data_dir)
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=shuffle)

    if augment:
        transform_img = train_transforms(img_size)
    else:
        transform_img = val_transforms(img_size)

    seed = random.randint(1,1000)
    mindspore.set_seed(seed)
    dataset = dataset.map(input_columns='image', num_parallel_workers=1, operations=transform_img)
    mindspore.set_seed(seed)
    dataset = dataset.map(input_columns="label", num_parallel_workers=1, operations=transform_img)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size, num_parallel_workers=1)

    if augment == True and shuffle == True:
        print("训练集数据量：", len(mc_dataset))
    elif augment == False and shuffle == False:
        print("验证集数据量：", len(mc_dataset))
    else:
        pass

    return dataset



if __name__ == '__main__':

    train_dataset = create_dataset('datasets/ISBI/val', img_size=224, batch_size=3, augment=False, shuffle=False)
    for item, (image, label) in enumerate(train_dataset):
        if item < 5:
            print(label.shape)

        # img = image.asnumpy()[3].transpose(1,2,0)  #.squeeze(0)
        # cv2.imshow("image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # lab = label.asnumpy()[3].transpose(1, 2, 0)  #.squeeze(0)
        # cv2.imshow("label", lab)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()