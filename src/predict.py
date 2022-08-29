import cv2
from mindspore import Tensor
from src.UNet_model import UNet
import mindspore as ms
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import create_dataset, val_transforms
import time
import mindspore.ops as ops
import os
import skimage.io as io
from tqdm import tqdm
import albumentations as A
import glob
import mindspore.dataset as ds
from skimage import img_as_ubyte

class No_mask_Data_Loader:
    def __init__(self, data_path, image_size=None, aug=True):
        # 初始化函数，读取所有data_path下的图片
        self.image_size = image_size
        self.aug = aug
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))
        self.test_aug = val_transforms(image_size)

    def __getitem__(self, index):
        # 根据index读取图片
        image = cv2.imread(self.imgs_path[index])

        augments = self.test_aug(image=image, mask=image)
        image, label = augments['image'], augments['mask']

        image = np.transpose(image, (2, 0, 1))
        image = image / 255.

        label = image

        return image.astype("float32"),label.astype("float32")

    @property
    def column_names(self):
        column_names = ['image','label']
        return column_names

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


def caculate_metrics(metrics, preds, gts, smooth=1e-5):
    pred = np.squeeze(preds.asnumpy())
    gt = np.squeeze(gts.asnumpy())

    metrics_list = [0. for i in range(len(metrics))]

    def Acc_metrics(y_pred, y):
        tp = np.sum(y_pred.flatten() == y.flatten())
        total = len(y_pred.flatten())
        single_acc = float(tp) / float(total)
        return single_acc

    def IoU_metrics(y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten() + y.flatten()) - intersection
        single_iou = float(intersection) / float(unionset + smooth)
        return single_iou

    def Dice_metrics(y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten()) + np.sum(y.flatten())
        single_dice = 2*float(intersection) / float(unionset + smooth)
        return single_dice

    def Sens_metrics(y_pred, y):
        tp = np.sum(y_pred.flatten() * y.flatten())
        actual_positives = np.sum(y.flatten())
        single_sens = float(tp) / float(actual_positives + smooth)
        return single_sens

    def Spec_metrics(y_pred, y):
        true_neg = np.sum((1 - y.flatten()) * (1 - y_pred.flatten()))
        total_neg = np.sum((1 - y.flatten()))
        single_spec = float(true_neg) / float(total_neg + smooth)
        return single_spec

    if "acc" in metrics:
        metrics_list[0] = Acc_metrics(pred, gt)

    if "iou" in metrics:
        metrics_list[1] = IoU_metrics(pred, gt)

    if "dice" in metrics:
        metrics_list[2] = Dice_metrics(pred, gt)

    if "sens" in metrics:
        metrics_list[3] = Sens_metrics(pred, gt)

    if "spec" in metrics:
        metrics_list[4] = Spec_metrics(pred, gt)

    return metrics_list


def model_pred(model, test_loader, result_path, Have_Mask):

    mtr = ['acc', 'iou', 'dice', 'sens', 'spec']
    metrics_list = [0. for i in range(len(mtr))]
    samples_num = 0

    for i, (image,  label) in enumerate(tqdm(test_loader)):
        samples_num += 1

        preds = model(image)

        if preds.min() < 0:
            preds = ops.Sigmoid()(preds)

        preds[preds > 0.5] = float(1)
        preds[preds <= 0.5] = float(0)

        if Have_Mask:
            for item in range(len(mtr)):
                metrics_list[item] += caculate_metrics(mtr, preds, label)[item]

        preds = np.squeeze(preds,axis=0)

        img = np.transpose(preds,(1,2,0))

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        io.imsave(os.path.join(result_path, "%05d.png" % i), img_as_ubyte(img.asnumpy()))


    if Have_Mask:
        print('%s:丨acc: %.4f丨丨iou: %.4f丨丨dice: %.4f丨丨sens: %.4f丨丨spec: %.4f丨' %
              ("Test", metrics_list[0]/samples_num, metrics_list[1]/samples_num, metrics_list[2]/samples_num, metrics_list[3]/samples_num, metrics_list[4]/samples_num))

    else:
        print("Evaluation metrics cannot be calculated without Mask")


def load_test_data(data_dir, img_size, batch_size, augment, shuffle, Have_Mask = False):
    if Have_Mask:
        test_dataset = create_dataset(data_dir, img_size=img_size, batch_size=batch_size, augment=augment, shuffle=shuffle)
    else:
        mc_dataset = No_mask_Data_Loader(data_path=data_dir, image_size=img_size, aug=augment)
        print("数据个数：", len(mc_dataset))
        test_dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=shuffle)
        test_dataset = test_dataset.batch(batch_size, num_parallel_workers=1)

    return test_dataset


Have_Mask = False

net = UNet()

ms.load_checkpoint("checkpoint/best.ckpt", net=net)

result_path = "predict"

test_dataset = load_test_data("datasets/ISBI/test_imgs/", 224, 1, augment = False, shuffle = False, Have_Mask = Have_Mask)

model_pred(net, test_dataset, result_path, Have_Mask)


