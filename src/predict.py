import os
import cv2
import mindspore.dataset as ds
import glob
import mindspore.dataset.vision as vision_C
import mindspore.dataset.transforms as C_transforms
import random
import mindspore
from mindspore.dataset.vision import Inter
import numpy as np
from tqdm import tqdm
from metric import metrics_
import skimage.io as io
from skimage import img_as_ubyte
from UNet_model import UNet


def val_transforms(img_size):
    return C_transforms.Compose([
    vision_C.Resize(img_size, interpolation=Inter.NEAREST),
    vision_C.Rescale(1/255., 0),
    vision_C.HWC2CHW()
    ])

class Data_Loader:
    def __init__(self, data_path, have_mask):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.have_mask = have_mask
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        if self.have_mask:
            self.label_path = glob.glob(os.path.join(data_path, 'mask/*.png'))

    def __getitem__(self, index):
        # 根据index读取图片
        image = cv2.imread(self.imgs_path[index])
        if self.have_mask:
            label = cv2.imread(self.label_path[index], cv2.IMREAD_GRAYSCALE)
            label = label.reshape((label.shape[0], label.shape[1], 1))
        else:
            label = image
        return image, label

    @property
    def column_names(self):
        column_names = ['image', 'label']
        return column_names

    def __len__(self):
        return len(self.imgs_path)


def create_dataset(data_dir, img_size, batch_size, shuffle, have_mask = False):
    mc_dataset = Data_Loader(data_path=data_dir, have_mask = have_mask)
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=shuffle)
    transform_img = val_transforms(img_size)
    seed = random.randint(1, 1000)
    mindspore.set_seed(seed)
    dataset = dataset.map(input_columns='image', num_parallel_workers=1, operations=transform_img)
    mindspore.set_seed(seed)
    dataset = dataset.map(input_columns="label", num_parallel_workers=1, operations=transform_img)
    dataset = dataset.batch(batch_size, num_parallel_workers=1)
    return dataset

def model_pred(model, test_loader, result_path, have_mask):
    model.set_train(False)
    test_pred = []
    test_label = []
    for batch, (data, label) in enumerate(test_loader.create_tuple_iterator()):
        pred = model(data)

        pred[pred > 0.5] = float(1)
        pred[pred <= 0.5] = float(0)

        preds = np.squeeze(pred, axis=0)
        img = np.transpose(preds,(1, 2, 0))

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        io.imsave(os.path.join(result_path, "%05d.png" % batch), img.asnumpy())

        test_pred.extend(pred.asnumpy())
        test_label.extend(label.asnumpy())

    if have_mask:
        mtr = ['acc', 'iou', 'dice', 'sens', 'spec']
        metric = metrics_(mtr, smooth=1e-5)
        metric.clear()
        metric.update(test_pred, test_label)
        res = metric.eval()
        print(f'丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))
    else:
        print("Evaluation metrics cannot be calculated without Mask")

if __name__ == '__main__':
    net = UNet(3, 1)
    mindspore.load_checkpoint("checkpoint/best_UNet.ckpt", net=net)
    result_path = "predict"
    test_dataset = create_dataset("datasets/ISBI/val/", 224, 1, shuffle=False, have_mask=True)
    model_pred(net, test_dataset, result_path, have_mask=True)

