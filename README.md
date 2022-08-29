## 基于MindSpore框架的Unet-2D案例实现

### 1 模型简介

Unet模型于2015年在论文《U-Net: Convolutional Networks for Biomedical Image Segmentation》中被提出，最初的提出是为了解决医学图像分割问题，用于细胞层面的图像分割任务。

Unet模型是在FCN网络的基础上构建的，但由于FCN无法获取上下文信息以及位置信息，导致准确性较低，Unet模型由此引入了U型结构获取上述两种信息，并且模型结构简单高效、容易构建，在较小的数据集上也能实现较高的准确率。

#### 1.1 模型结构
Unet模型的整体结构由两部分组成，即特征提取网络和特征融合网络，其结构也被称为“编码器-解码器结构”，并且由于网络整体结构类似于大写的英文字母“U”，故得名Unet，在其原始论文中定义的网络结构如图1所示。

<center>
    <img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h5bw5io8tqj20n10fa3zo.jpg" alt="image-20220819101847606" style="zoom:67%;" />
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">图1 Unet网络结构图</div>
</center>

整个模型结构就是在原始图像输入后，首先进行特征提取，再进行特征融合：

a) 左半部分负责特征提取的网络结构（即编码器结构）需要利用两个3x3的卷积核与2x2的池化层组成一个“下采样模块”，每一个下采样模块首先会对特征图进行两次valid卷积，再进行一次池化操作。由此经过4个下采样模块后，原始尺寸为572x572大小、通道数为1的原始图像，转换为了大小为28x28、通道数为1024的特征图。

b) 右半部分负责进行上采样的网络结构（即解码器结构）需要利用1次反卷积操作、特征拼接操作以及两个3x3的卷积核作为一个“上采样模块”，每一个上采样模块首先会对特征图通过反卷积操作使图像尺寸增加1倍，再通过拼接编码器结构中的特征图使得通道数增加，最后经过两次valid卷积。由此经过4个上采样模块后，经过下采样模块的、大小为28x28、通道数为1024的特征图，转换为了大小为388x388、通道数为64的特征图。

c) 网络结构的最后一部分是通过两个1x1的卷积核将经过上采样得到的通道数为64的特征图，转换为了通道数为2的图像作为预测结果输出。

#### 1.2 模型特点

a) 利用拼接操作将低级特征图与高级特征图进行特征融合

b) 完全对称的U型结构使得高分辨率信息和低分辨率信息在目标图片中增加，前后特征融合更为彻底。

c) 结合了下采样时的低分辨率信息（提供物体类别识别依据）和上采样时的高分辨率信息（提供精准分割定位依据），此外还通过融合操作填补底层信息以提高分割精度。

### 2 案例实现

#### 2.1 **环境准备与数据读取**

本案例基于MindSpore-CPU版本实现，在CPU上完成模型训练。

案例实现所使用的数据即ISBI果蝇电镜图数据集，可以从http://brainiac2.mit.edu/isbi_challenge/中下载，下载好的数据集包括3个tif文件，分别对应测试集样本、训练集标签、训练集样本，文件路径结构如下：

```
.datasets/
└── ISBI
    ├── test-volume.tif
    ├── train-labels.tif
    └── train-volume.tif
```

其中每个tif文件都由30副图片压缩而成，所以接下来需要获取每个tif文件中所存储的所有图片，将其转换为png格式存储，得到训练集样本对应的30张png图片、训练集标签对应的30张png图片以及测试集样本对应的30张png图片。

具体的实现方式首先是将tif文件转换为数组形式，之后通过io操作将每张图片对应的数组存储为png图像，处理过后的训练集样本及其对应的标签图像如图2所示。将3个tif文件转换为png格式后，针对训练集的样本与标签，将其以2:1的比例，重新划分为了训练集与验证集，划分完成后的文件路径结构如下：

```
.datasets/
└── ISBI
    ├── test_imgs
    │   ├── 00000.png
    │   ├── 00001.png
    │   └── . . . . .
    ├── train
    │   ├── image
    │   │   ├── 00001.png
    │   │   ├── 00002.png
    │   │   └── . . . . .
    │   └── mask
    │       ├── 00001.png
    │       ├── 00002.png
    │       └── . . . . .
    └── val
        ├── image
        │   ├── 00000.png
        │   ├── 00003.png
        │   └── . . . . .
        └── mask
            ├── 00000.png
            ├── 00003.png
            └── . . . . .
```

<center>
    <img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h5bwhda4g2j20fz0chabb.jpg" alt="image-20220819101847606" style="zoom:50%;" />
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">图2 训练集样本及其对应标签</div>
</center>

#### 2.2 数据集创建

在进行上述tif文件格式转换，以及测试集和验证集的进一步划分后，就完成了数据读取所需的所有工作，接下来就需要利用处理好的图像数据，通过一定的图像变换来进行数据增强，并完成数据集的创建。

数据增强部分是引入了albumentations库，针对训练集样本和标签，首先通过A.resize()方法将图像尺寸重新调整为统一大小，之后再进行转置以及水平翻转、垂直翻转，最后进行平移、尺度加旋转变换以及随机旋转90度，就完成了针对训练集样本和标签的数据增强。针对验证集的样本和标签，仅通过A.resize()方法将图像尺寸重新调整为统一大小。

其次数据集的创建部分，首先是定义了Data_Loader类，在该类的__init__函数中，根据传入的data_path参数，确定在数据读取阶段设置好的、训练集和验证集的存储路径，再设置对应的样本和标签路径，并针对训练集和验证集的不同数据增强方法。在该类的__getitem__函数中，通过传入索引值读取训练集或验证集存储路径下的样本和标签图像，并对图像进行对应的数据增强操作，之后再对样本和标签的形状进行转置，就完成了__getitem__函数对样本和标签图像的读取。最后通过定义create_dataset函数，传入data_dir、batch_size等参数，在函数中实例化Data_Loader类获取data_dir，也就是训练集或验证集对应路径下的样本和标签元组对，再通过mindspore.dataset中的GeneratorDataset将元组转换为Tensor，最后通过设定好的batch_size将样本和标签按照batch_size大小分组，由此完成数据集的创建，上述流程对应代码如下：

```python
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
    print("数据个数：", len(mc_dataset))
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=shuffle)
    dataset = dataset.batch(batch_size, num_parallel_workers=1)
    return dataset
```

#### 2.3 模型构建

本案例实现中所构建的Unet模型结构与2015年论文中提出的Unet结构大致相同，但本案例中Unet网络模型的“下采样模块”与“上采样模块”使用的卷积类型都为Same卷积，而原论文中使用的是Valid卷积。此外，原论文的网络模型最终使用两个1x1的卷积核，输出了通道数2的预测图像，而本案例的网络模型最终使用的是1个1x1的卷积核，输出通道数为1的灰度图，和标签图像格式保持一致。实际构建的Unet模型结构如图3所示。

<center>
    <img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h5bwhpesqpj20no0g7jso.jpg" alt="image-20220819101847606" style="zoom:67%;" />
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">图3 实际构建的Unet模型结构</div>
</center>

Mindspore框架构建网络的流程与pytorch类似，在定义模型类时需要继承Cell类，并重写__init__和construct方法。具体的实现方式首先是定义了一个double_conv模型类，在类中重写__init__方法，通过使用nn.Conv2d层定义“下采样模块”与“上采样模块”中都使用到的两个卷积函数，并且在每个卷积层后加入nn.BatchNorm2d层来对每次卷积后的特征图进行标准化，防止过拟合，以及使用nn.ReLU层加入非线性的激活函数。之后在construct方法中使用定义好的运算构建前向网络。

在doubel_conv模型类定义好之后，接下来就是通过定义UNet模型类来完成整个UNet网络的构建。在UNet模型类的__init__方法中实例化double_conv类来表示两个连续的卷积层，接着使用nn.MaxPool2d来进行最大池化，由此完成了1个“下采样模块”的构建，重复4次即可完成网络中的编码器部分。针对解码器部分，使用了nn.ResizeBilinear层来表示反卷积层，接着实例化了double_conv类来表示两个卷积层，由此完成了1个“上采样模块”的构建，重复4次即完成网络中解码器部分的搭建。之后通过1个nn.Conv2d层来完成预测图像的输出。最后在construct方法中使用定义好的运算构建前向网络，由此完成整个Unet网络模型的构建。上述构建流程的对应代码如下所示：

```python
from mindspore import nn
import mindspore.numpy as np
from mindspore import Tensor
import mindspore.ops as ops

class double_conv(nn.Cell):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.double_conv = nn.SequentialCell(nn.Conv2d(in_ch, out_ch, 3),
                                             nn.BatchNorm2d(out_ch), nn.ReLU(),
                                             nn.Conv2d(out_ch, out_ch, 3),
                                             nn.BatchNorm2d(out_ch), nn.ReLU())

    def construct(self, x):
        x = self.double_conv(x)
        return x

class UNet(nn.Cell):
    def __init__(self, in_ch=3):
        super(UNet, self).__init__()
        # Encoder
        # [N,3,224,224]->[N,64,224,224]
        self.double_conv1 = double_conv(in_ch, 64)
        # [N,64,224,224]->[N,64,112,112]
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,64,112,112]->[N,128,112,112]
        self.double_conv2 = double_conv(64, 128)
        # [N,128,112,112]->[N,128,56,56]
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,128,56,56]->[N,256,56,56]
        self.double_conv3 = double_conv(128, 256)
        # [N,256,56,56]->[N,256,28,28]
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,256,28,28]->[N,512,28,28]
        self.double_conv4 = double_conv(256, 512)
        # [N,512,28,28]->[N,512,14,14]
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # [N,512,14,14]->[N,1024,14,14]
        self.double_conv5 = double_conv(512, 1024)

        # Decoder
        # [N,1024,14,14]->[N,1024,28,28]
        self.upsample1 = nn.ResizeBilinear()
        # [N,1024+512,28,28]->[N,512,28,28]
        self.double_conv6 = double_conv(1024 + 512, 512)
        # [N,512,28,28]->[N,512,56,56]
        self.upsample2 = nn.ResizeBilinear()
        # [N,512+256,56,56]->[N,256,56,56]
        self.double_conv7 = double_conv(512 + 256, 256)
        # [N,256,56,56]->[N,256,112,112]
        self.upsample3 = nn.ResizeBilinear()
        # [N,256+128,112,112]->[N,128,112,112]
        self.double_conv8 = double_conv(256 + 128, 128)
        # [N,128,112,112]->[N,128,224,224]
        self.upsample4 = nn.ResizeBilinear()
        # [N,128+64,224,224]->[N,64,224,224]
        self.double_conv9 = double_conv(128 + 64, 64)

        self.final = nn.Conv2d(64, 1, 1)
        self.sigmoid = ops.Sigmoid()

    def construct(self, x):

        feature1 = self.double_conv1(x)
        tmp = self.maxpool1(feature1)
        feature2 = self.double_conv2(tmp)
        tmp = self.maxpool2(feature2)
        feature3 = self.double_conv3(tmp)
        tmp = self.maxpool3(feature3)
        feature4 = self.double_conv4(tmp)
        tmp = self.maxpool4(feature4)
        feature5 = self.double_conv5(tmp)

        up_feature1 = self.upsample1(feature5, scale_factor=2)
        tmp = np.concatenate((feature4, up_feature1), axis=1)
        tmp = self.double_conv6(tmp)
        up_feature2 = self.upsample2(tmp, scale_factor=2)
        tmp = np.concatenate((feature3, up_feature2), axis=1)
        tmp = self.double_conv7(tmp)
        up_feature3 = self.upsample3(tmp, scale_factor=2)
        tmp = np.concatenate((feature2, up_feature3), axis=1)
        tmp = self.double_conv8(tmp)
        up_feature4 = self.upsample4(tmp, scale_factor=2)
        tmp = np.concatenate((feature1, up_feature4), axis=1)
        tmp = self.double_conv9(tmp)
        output = self.sigmoid(self.final(tmp))

        return output
```

#### 2.4 自定义评估指标

为了能够更加全面和直观的观察网络模型训练效果，本案例实现中还使用了MindSpore框架来自定义Metrics，在自定义的metrics类中使用了多种评价函数来评估模型的好坏，分别为准确率Acc、交并比IoU、Dice系数、灵敏度Sens、特异性Spec。

a) 其中准确率Acc是图像中正确分类的像素百分比。即分类正确的像素占总像素的比例，用公式可表示为：
$$ Acc=\frac{TP+TN}{TP+TN+FP+FN} $$
其中：

- TP：真阳性数，在label中为阳性，在预测值中也为阳性的个数。
- TN：真阴性数，在label中为阴性，在预测值中也为阴性的个数。
- FP：假阳性数，在label中为阴性，在预测值中为阳性的个数。
- FN：假阴性数，在label中为阳性，在预测值中为阴性的个数。

b) 交并比IoU是预测分割和标签之间的重叠区域除以预测分割和标签之间的联合区域（两者的交集/两者的并集），是语义分割中最常用的指标之一，其计算公式为：
$$
\text {IoU}=\frac{|A\capB|}{|A\cupB|}=\frac{TP}{TP+FP+FN}
$$
c) Dice系数定义为两倍的交集除以像素和，也叫F1score，与IoU呈正相关关系，其计算公式为：
$$
\text {Dice}=\frac{2|A\capB|}{|A|+|B|}=\frac{2TP}{2TP+FP+FN}
$$
d) 敏感度Sens和特异性Spec分别是描述识别出的阳性占所有阳性的比例，以及描述识别出的负例占所有负例的比例，计算公式分别为：
$$
\text {Sens}=\frac{TP}{TP+FN}
$$

$$
\text {Spec}=\frac{TN}{FP+TN}
$$

具体的实现方法首先是自定义metrics_类，并按照MindSpore官方文档继承nn.Metric父类，接着根据上述5个评价指标的计算公式，在类中定义5个指标的计算方法，之后通过重新实现clear方法来初始化相关参数；重新实现update方法来传入模型预测值和标签，通过上述定义的各评价指标计算方法，计算每个指标的值并存入一个列表；最后通过重新实现eval方法来讲存储各评估指标值的列表返回。上述流程对应的代码如下：

```python
import numpy as np
from mindspore._checkparam import Validator as validator
from mindspore.nn.metrics import *
from mindspore import Tensor
from mindspore.nn import Metric

def _convert_data(data):

    if isinstance(data, Tensor):
        data = data.asnumpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError(f"For 'Metric' and its derived classes, the input data type must be tensor, list or "
                        f"numpy.ndarray, but got {type(data)}.")
    return data

class metrics_(Metric):

    def __init__(self, metrics, smooth=1e-5):
        super(metrics_, self).__init__()
        self.metrics = metrics
        self.smooth = validator.check_positive_float(smooth, "smooth")
        self.metrics_list = [0. for i in range(len(self.metrics))]
        self._samples_num = 0
        self.clear()

    def Acc_metrics(self,y_pred, y):
        tp = np.sum(y_pred.flatten() == y.flatten(), dtype=y_pred.dtype)
        total = len(y_pred.flatten())
        single_acc = float(tp) / float(total)
        return single_acc

    def IoU_metrics(self,y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten() + y.flatten()) - intersection
        single_iou = float(intersection) / float(unionset + self.smooth)
        return single_iou

    def Dice_metrics(self,y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten()) + np.sum(y.flatten())
        single_dice = 2*float(intersection) / float(unionset + self.smooth)
        return single_dice

    def Sens_metrics(self,y_pred, y):
        tp = np.sum(y_pred.flatten() * y.flatten())
        actual_positives = np.sum(y.flatten())
        single_sens = float(tp) / float(actual_positives + self.smooth)
        return single_sens

    def Spec_metrics(self,y_pred, y):
        true_neg = np.sum((1 - y.flatten()) * (1 - y_pred.flatten()))
        total_neg = np.sum((1 - y.flatten()))
        single_spec = float(true_neg) / float(total_neg + self.smooth)
        return single_spec

    def clear(self):
        """Clears the internal evaluation result."""
        self.metrics_list = [0. for i in range(len(self.metrics))]
        self._samples_num = 0

    def update(self, *inputs):

        if len(inputs) != 2:
            raise ValueError("For 'update', it needs 2 inputs (predicted value, true value), "
                             "but got {}.".format(len(inputs)))

        y_pred = np.round(_convert_data(inputs[0]))

        y = _convert_data(inputs[1])

        self._samples_num += y.shape[0]

        if y_pred.shape != y.shape:
            raise ValueError(f"For 'update', predicted value (input[0]) and true value (input[1]) "
                             f"should have same shape, but got predicted value shape: {y_pred.shape}, "
                             f"true value shape: {y.shape}.")

        for i in range(y.shape[0]):
            if "acc" in self.metrics:
                single_acc = self.Acc_metrics(y_pred[i], y[i])
                self.metrics_list[0] += single_acc
            if "iou" in self.metrics:
                single_iou = self.IoU_metrics(y_pred[i], y[i])
                self.metrics_list[1] += single_iou
            if "dice" in self.metrics:
                single_dice = self.Dice_metrics(y_pred[i], y[i])
                self.metrics_list[2] += single_dice
            if "sens" in self.metrics:
                single_sens = self.Sens_metrics(y_pred[i], y[i])
                self.metrics_list[3] += single_sens
            if "spec" in self.metrics:
                single_spec = self.Spec_metrics(y_pred[i], y[i])
                self.metrics_list[4] += single_spec

    def eval(self):

        if self._samples_num == 0:
            raise RuntimeError("The 'metrics' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, or has "
                               "called update method before calling eval method.")
        for i in range(len(self.metrics_list)):
            self.metrics_list[i] = self.metrics_list[i] / float(self._samples_num)

        return self.metrics_list
```

#### 2.5 自定义回调机制

本案例还利用MindSpore框架，基于Callback基于自定义了回调类EvalCallBack，为了满足在模型训练时，能够在每一次epoch结束后计算自定义metrics_类中定义的5个评价指标，并根据评价指标的变化保存当前最优的模型的特定需求。

具体的实现流程首先是在自定义的EvalCallBack类中，通过__init__方法初始化相关参数，并重新实现epoch_end方法，在该方法中通过MindSpore的字典类型变量RunContext.original_args()获取模型训练时记录的相关属性，如cur_epoch_num，即当前epoch数，以及train_network,即训练的网络模型，在epoch_end方法中还计算了metrics_类中定义的5个评估指标，并比较当前epoch的IoU值与记录的最优IoU值，利用先前获取到的train_network将表现最优的模型保存在特定路径下。综上实现了在模型训练时，每个epoch结束后计算metric_类中定义的各个评价指标，并根据IoU指标保存最优模型。上述流程对应的代码如下：

```python
from mindspore.train.callback import Callback
from mindspore import nn
from mindspore.ops import operations as ops
import os
import stat
from mindspore import save_checkpoint
from mindspore import log as logger

class TempLoss(nn.Cell):
    """A temp loss cell."""

    def __init__(self):
        super(TempLoss, self).__init__()
        self.identity = ops.identity()

    def construct(self, logits, label):
        return self.identity(logits)

def apply_eval(eval_param_dict):
    """run Evaluation"""
    model = eval_param_dict["model"]
    dataset = eval_param_dict["dataset"]
    metrics_name = eval_param_dict["metrics_name"]
    eval_score = model.eval(dataset, dataset_sink_mode=False)[metrics_name]
    return eval_score

class EvalCallBack(Callback):

    def __init__(self, eval_function, eval_param_dict, interval=1, eval_start_epoch=1, save_best_ckpt=True,
                 ckpt_directory="./", besk_ckpt_name="best.ckpt", monitor="IoU", flag = "Train"):
        super(EvalCallBack, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.eval_start_epoch = eval_start_epoch
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.bast_ckpt_path = os.path.join(ckpt_directory, besk_ckpt_name)
        self.monitor_name = monitor
        self.flag = flag
    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def epoch_end(self, run_context):
        """Callback when epoch end."""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            res = self.eval_function(self.eval_param_dict)

            print("epoch: {} {}_metrics:".format(cur_epoch, self.flag), '丨acc: %.4f丨丨iou: %.4f丨丨dice: %.4f丨丨sens: %.4f丨丨spec: %.4f丨' %
                  (res[0], res[1], res[2], res[3],res[4]), flush=True)

            if self.flag != "train":
                if res[1] > self.best_res:
                    print('IoU improved from %0.4f to %0.4f' % (self.best_res, res[1]), flush=True)
                    self.best_res = res[1]
                    self.best_epoch = cur_epoch

                    if self.save_best_ckpt:
                        if os.path.exists(self.bast_ckpt_path):
                            self.remove_ckpoint_file(self.bast_ckpt_path)
                        save_checkpoint(cb_params.train_network, self.bast_ckpt_path)
                        print("saving best checkpoint at: {} ".format(self.bast_ckpt_path), flush=True)
                else:
                    print('IoU did not improve from %0.4f' % (self.best_res))
                print("="*100)

    def end(self, run_context):
        print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.monitor_name, self.best_res, self.best_epoch), flush=True)
```

#### 2.6 模型训练及评估

在模型训练时，首先是设置模型训练的epoch次数为50，再通过2.1节中自定义的create_dataset方法创建了训练集和验证集，其中训练集batch_size大小为4，验证集batch_size大小为2，图像尺寸统一调整为224x224；损失函数使用nn.BCELoss，优化器使用nn.Adam，并设置学习率为0.01。回调函数方面使用了LossMonitor和TimeMonitor来监控训练过程中每个epoch结束后，损失值Loss的变化情况以及每个epoch、每个step的运行时间，还实例化了2.5节中自定义的回调类EvalCallBack，实现计算每个epoch结束后，在2.4节中定义的5个评估指标，并保存当前最优模型。在50个epcoh结束后，模型在训练集和验证集上的评估指标如表1所示：

模型训练部分的代码如下：

```python
from metric import  metrics_
from eval_callback import TempLoss,EvalCallBack, apply_eval
from UNet_model import UNet
import mindspore.nn as nn
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore import Model
from data_loader import create_dataset
from mindspore.train.loss_scale_manager import FixedLossScaleManager

epochs = 50

train_dataset = create_dataset("datasets/ISBI/train/", img_size=224, batch_size= 4, augment=True, shuffle = True)
val_dataset = create_dataset("datasets/ISBI/val/", img_size=224, batch_size= 2, augment=False, shuffle = False)


iters_per_epoch = train_dataset.get_dataset_size()
total_train_steps = iters_per_epoch * epochs
print('iters_per_epoch: ', iters_per_epoch)
print('total_train_steps: ', total_train_steps)

criterion = nn.BCELoss()
criterion.add_flags_recursive(fp32=True)

net = UNet()
manager_loss_scale = FixedLossScaleManager(drop_overflow_update=False)

optimizer = nn.Adam(params=net.trainable_params(), learning_rate=0.001, weight_decay=1e-6)

metrics_name = ["acc", "iou", "dice", "sens", "spec"]

model = Model(net,
              optimizer=optimizer,
              amp_level="O0",
              loss_fn=criterion,
              loss_scale_manager=None,
              metrics={'metrics':metrics_(metrics_name, smooth=1e-5)})

time_cb = TimeMonitor(data_size=iters_per_epoch)
loss_cb = LossMonitor(per_print_times = iters_per_epoch)

# config_ckpt = CheckpointConfig(save_checkpoint_steps=iters_per_epoch, keep_checkpoint_max=5)
# cbs_1 = ModelCheckpoint(prefix="UNet", directory='checkpoint', config=config_ckpt)

eval_param_dict_train = {"model": model, "dataset": train_dataset, "metrics_name": 'metrics'}
eval_cb_train = EvalCallBack(apply_eval, eval_param_dict_train,
                       interval=1, eval_start_epoch=1,
                       save_best_ckpt=False, ckpt_directory="checkpoint",
                       besk_ckpt_name="best.ckpt", monitor= "IoU", flag = 'train')  # monitor

eval_param_dict_val = {"model": model, "dataset": val_dataset, "metrics_name": 'metrics'}
eval_cb_val = EvalCallBack(apply_eval, eval_param_dict_val,
                       interval=1, eval_start_epoch=1,
                       save_best_ckpt=True, ckpt_directory="checkpoint",
                       besk_ckpt_name="best.ckpt", monitor= "IoU", flag = 'Val')  # monitor

cbs = [time_cb, loss_cb, eval_cb_train, eval_cb_val]

model.train(epochs, train_dataset, callbacks=cbs)
```

|                | Acc  | IoU  | Dice | Sens | Spec |
| :------------: | :--: | :--: | :--: | ---: | ---- |
|   Train set    | 0.90 | 0.88 | 0.94 | 0.94 | 0.76 |
| Validation set | 0.92 | 0.90 | 0.95 | 0.96 | 0.77 |

根据表1评价指标结构，本案例构建的网络模型具有较好的性能，能够实现对测试集进行较为准确的预测，针对测试集的部分预测结果如图4所示。

<center>
    <img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h5bwigmy47j20l2064dgy.jpg" alt="image-20220819101847606" style="zoom:50%;" />
    <br>
    <div style="color:orange;
    display: inline-block;
    color: #999;
    padding: 2px;">图4 模型预测结果</div>
</center>

### 3 总结

本案例基于MindSpore框架针对ISBI数据集，完成了数据读取、数据集创建、Unet模型构建，并根据特定需求自定义了评估指标和回调函数，进行了模型训练和评估，顺利完成了预测结果的输出。通过此案例进一步加深了对Unet模型结构和特性的理解，并结合MindSpore框架提供的文档和教程，掌握了利用Mindspore框架实现特定案例的流程，以及多种API的使用方法，为以后在实际场景中应用MindSpore框架提供支持。
