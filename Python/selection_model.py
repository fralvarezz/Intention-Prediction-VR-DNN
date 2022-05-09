
import torch
import torch.nn as nn
import torch.nn.functional as fn

from torch.utils.data import DataLoader

import utils
import tqdm

class SelectionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, learning_rate, sequence_length, device="cuda"):
        super().__init__()

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, hidden_size)
        )  # added some additional fully connected layers, more as a place holder for now

        self.lstm = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size, dtype=torch.float32)
        self.fc = nn.Linear(hidden_size, num_classes, dtype=torch.float32)  # softmax not needed in crossentropy loss implementation

        #self.lstm_b = nn.LSTM(input_size, hidden_size, 2, batch_first=True, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        self.train_writer, self.val_writer = None, None
        self.batch_idx = 0



    def forward(self, sequence):
        h = c = None
        for frame_idx in range(self.sequence_length):
            h_t = self.encoder(sequence[:, frame_idx, :])
            h, c = self.lstm(h_t, None if h is None else (h, c))
        out = self.fc(h)
        return out

        # # Set initial hidden and cell states
        # h0 = torch.zeros(2, sequence.size(0), self.hidden_size, dtype=torch.float32).to(self.device)
        # c0 = torch.zeros(2, sequence.size(0), self.hidden_size, dtype=torch.float32).to(self.device)
        #
        # # Forward propagate LSTM
        # out, _ = self.lstm_b(
        #     sequence, (h0, c0)
        # )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out = out[:, -1, :]
        # out = self.fc(out)
        # return out

    def _report(self, writer, loss):
        if writer is None:
            return
        writer.add_scalar("loss/loss", loss.item(), self.batch_idx)


    def eval_batch(self, val_data, report_loss=False):
        self.eval()

        batch = next(iter(val_data)).float().to(self.device)

        with torch.no_grad():
            out = self.forward(batch[:, :, :-1])
            labels = batch[:, -1, -1]
            labels = labels.type(torch.LongTensor).to(self.device)
            one_hot_labels = fn.one_hot(labels, self.num_classes).float()

            # print("")
            # print(torch.argmax(out, dim=1))
            # print(labels)
            # print("")

            loss = self.loss(out, one_hot_labels)

        if report_loss:
            self._report(self.val_writer, loss)

        return out

    def loss(self, predicted, expected):
        criterium = nn.CrossEntropyLoss()
        return criterium(predicted, expected)

    def do_train(self, train_data: DataLoader, n_train_batches: int, val_data: DataLoader, val_every_n_train_batches=10):
        if self.train_writer is None:
            self.train_writer, self.val_writer = utils.get_writers("selection")

        data_iter = iter(train_data)
        pbar = tqdm.tqdm(range(n_train_batches))
        for _ in pbar:
            self.train()
            batch = next(data_iter).float().to(self.device)

            self.optimizer.zero_grad()

            out = self.forward(batch[:, :, :-1])
            labels = batch[:, -1, -1]
            labels = labels.type(torch.LongTensor).to(self.device)
            one_hot_labels = fn.one_hot(labels, self.num_classes).float()

            loss = self.loss(out, one_hot_labels)
            assert torch.isfinite(loss), "loss is not finite: %f" % loss
            loss.backward()
            self.optimizer.step()

            self._report(self.train_writer, loss)
            pbar.set_description("Training generator. Loss %0.3f" % loss)

            if self.batch_idx % val_every_n_train_batches == 0:
                self.eval_batch(val_data, True)
            self.batch_idx += 1

    def save(self, fn):
        torch.save({
            'batch_idx': self.batch_idx,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, fn)

    def load(self, fn):
        checkpoint = torch.load(fn, map_location=torch.device(self.device))
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.batch_idx = checkpoint["batch_idx"]