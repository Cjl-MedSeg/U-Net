import numpy as np
from mindspore._checkparam import Validator as validator
from mindspore.nn import Metric

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
            raise ValueError("For 'update', it needs 2 inputs (predicted value, true value), ""but got {}.".format(len(inputs)))


        y_pred = np.array(inputs[0])
        y_pred[y_pred > 0.5] = float(1)
        y_pred[y_pred <= 0.5] = float(0)

        y = np.array(inputs[1])

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
