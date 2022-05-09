import os
import datetime
from torch.utils.tensorboard import SummaryWriter


def get_writers(name):
    tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or 'tensorboard'
    revision = os.environ.get("REVISION") or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    message = os.environ.get('MESSAGE')

    train_writer = SummaryWriter(tensorboard_dir + '/%s/%s/train/%s' % (name, revision, message))
    val_writer = SummaryWriter(tensorboard_dir + '/%s/%s/val/%s' % (name, revision, message))
    return train_writer, val_writer