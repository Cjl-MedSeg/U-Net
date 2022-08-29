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
        print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.monitor_name,
                                                                                     self.best_res,
                                                                                     self.best_epoch), flush=True)