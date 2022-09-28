from src.metric import metrics_
from src.UNet_model import UNet
import mindspore.nn as nn
from src.data_loader import create_dataset
from src.configs import get_config
from mindspore import ops
import mindspore
from mindspore import ms_function

cfg = get_config()
train_dataset = create_dataset(cfg.train_data_path, img_size=cfg.imgsize, batch_size= cfg.batch_size, augment=True, shuffle = True)
val_dataset = create_dataset(cfg.val_data_path, img_size=cfg.imgsize, batch_size= cfg.batch_size, augment=False, shuffle = False)


def train(model, dataset, loss_fn, optimizer, met):
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits
    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # Define function of one-step training
    @ms_function
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits

    size = dataset.get_dataset_size()
    model.set_train(True)
    train_loss = 0
    train_pred = []
    train_label = []
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss, logits = train_step(data, label)
        train_loss += loss.asnumpy()
        train_pred.extend(logits.asnumpy())
        train_label.extend(label.asnumpy())

    train_loss /= size
    metric = metrics_(met, smooth=1e-5)
    metric.clear()
    metric.update(train_pred, train_label)
    res = metric.eval()
    print(f'Train loss:{train_loss:>4f}','丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))


def val(model, dataset, loss_fn, met):
    size = dataset.get_dataset_size()
    model.set_train(False)
    val_loss = 0
    val_pred = []
    val_label = []
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        pred = model(data)
        val_loss += loss_fn(pred, label).asnumpy()
        val_pred.extend(pred.asnumpy())
        val_label.extend(label.asnumpy())

    val_loss /= size
    metric = metrics_(met, smooth=1e-5)
    metric.clear()
    metric.update(val_pred, val_label)
    res = metric.eval()

    print(f'Val loss:{val_loss:>4f}','丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))

    checkpoint = res[1]
    return checkpoint


net = UNet(cfg.in_channel, cfg.n_classes)

criterion = nn.BCEWithLogitsLoss()
optimizer = nn.SGD(params=net.trainable_params(), learning_rate=cfg.lr)

iters_per_epoch = train_dataset.get_dataset_size()
total_train_steps = iters_per_epoch * cfg.epochs
print('iters_per_epoch: ', iters_per_epoch)
print('total_train_steps: ', total_train_steps)

metrics_name = ["acc", "iou", "dice", "sens", "spec"]

best_iou = 0
ckpt_path = 'checkpoint/best_UNet.ckpt'
for epoch in range(cfg.epochs):
    print(f"Epoch [{epoch+1} / {cfg.epochs}]")
    train(net, train_dataset, criterion, optimizer, metrics_name)
    checkpoint_best = val(net, val_dataset, criterion, metrics_name)
    if checkpoint_best > best_iou:
        print('IoU improved from %0.4f to %0.4f' % (best_iou, checkpoint_best))
        best_iou = checkpoint_best
        mindspore.save_checkpoint(net, ckpt_path)
        print("saving best checkpoint at: {} ".format(ckpt_path))
    else:
        print('IoU did not improve from %0.4f' % (best_iou),"\n-------------------------------")
print("Done!")
