import ml_collections

def get_config():

    config = ml_collections.ConfigDict()

    config.epochs = 10

    config.train_data_path = "./datasets/ISBI/train/"

    config.val_data_path = "./datasets/ISBI/val/"

    config.imgsize = 224

    config.batch_size = 4

    config.pretrained_path = None

    config.in_channel = 3

    config.n_classes = 1

    config.lr = 0.0001

    return config