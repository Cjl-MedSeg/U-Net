from src.metric import metrics_
from src.eval_callback import TempLoss,EvalCallBack, apply_eval
from src.UNet_model import UNet
import mindspore.nn as nn
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore import Model
from src.data_loader import create_dataset
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from src.configs import get_config

cfg = get_config()

train_dataset = create_dataset(cfg.train_data_path, img_size=cfg.imgsize, batch_size= cfg.batch_size, augment=True, shuffle = True)
val_dataset = create_dataset(cfg.val_data_path, img_size=cfg.imgsize, batch_size= cfg.batch_size, augment=False, shuffle = False)


iters_per_epoch = train_dataset.get_dataset_size()
total_train_steps = iters_per_epoch * cfg.epochs
print('iters_per_epoch: ', iters_per_epoch)
print('total_train_steps: ', total_train_steps)

criterion = nn.BCELoss()
criterion.add_flags_recursive(fp32=True)

net = UNet(cfg.in_channel, cfg.n_classes)
manager_loss_scale = FixedLossScaleManager(drop_overflow_update=False)

optimizer = nn.Adam(params=net.trainable_params(), learning_rate=cfg.lr, weight_decay=1e-6)

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
                       besk_ckpt_name="best.ckpt", monitor= "IoU", flag = 'val')  # monitor

cbs = [time_cb, loss_cb, eval_cb_train, eval_cb_val]

model.train(cfg.epochs, train_dataset, callbacks=cbs)

