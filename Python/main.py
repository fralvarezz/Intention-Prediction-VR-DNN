import os
from pathlib import Path

from selection_model import SelectionModel
from data import TrackingDataset
from torch.utils.data.dataloader import DataLoader

#basedir = str(Path(__file__).parent.absolute())
model_name = "model_lstm"
batch_size = 16
sequence_length = 30
n_batches = 1_000
latent_size = 64  # 512 #256
learning_rate = 3e-4
num_classes = 10

sufix = model_name + "_bs" + str(batch_size) + "_sl" + str(sequence_length) + "_nb" + str(n_batches) + \
        "_ls" + str(latent_size) + "_lr" + str(learning_rate)

os.environ["MESSAGE"] = sufix  # TODO can't remember why this is here, might be related to tensorboard
model_pathname = "./models/" + sufix + ".pt"
if not Path("./models/").exists():
    os.mkdir("./models/")

train_data = TrackingDataset("./data/train", sequence_length)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_data = TrackingDataset("./data/val", sequence_length, train_data.norm_mean, train_data.norm_std)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

selectionModel = SelectionModel(input_size=train_data.frame_size, hidden_size=latent_size, num_classes=num_classes,
                                learning_rate=learning_rate, sequence_length=sequence_length, device="cuda")

if Path(model_pathname).exists():
    selectionModel.load(model_pathname)
else:
    selectionModel.do_train(train_dataloader, n_batches, val_dataloader)
    selectionModel.save(model_pathname)
