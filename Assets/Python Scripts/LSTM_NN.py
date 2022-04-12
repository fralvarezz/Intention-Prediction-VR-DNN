import torch
import torch.nn as nn
import numpy as np

# Device configuration
import torchvision
import torchvision.transforms as transforms

import UNITY_CSV_PARSER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# num_classes = num of items
num_classes = 10  # 9 items and None
num_epochs = 5
batch_size = 1
learning_rate = 0.001

input_size = 19  # num of inputs per frame
sequence_length = 1  # num of frames at a time, I believe
hidden_size = 128
num_layers = 1


class RNN_LSTM(nn.Module):
    def __init__(self, _input_size, _hidden_size, _num_layers, _num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = _hidden_size
        self.num_layers = _num_layers
        self.lstm = nn.LSTM(_input_size, _hidden_size, _num_layers, batch_first=True, dtype=torch.float32)
        self.fc = nn.Linear(_hidden_size * sequence_length, _num_classes, dtype=torch.float32)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

up = UNITY_CSV_PARSER.UnityParser("../CSVs/experiment10.csv")

frames_to_backlabel = 500
up.update_label_frames(frames_to_backlabel)
# data = torch.from_numpy(np.genfromtxt("formatted_success.csv", delimiter=";"))

# TODO: Finish below

n_total_steps = len(up.data) / sequence_length

for epoch in range(num_epochs):
    i = 0
    j = i + sequence_length
    n = len(up)

    if j > n:
        j = n - 1

    while j < n:
        frames_batch = up.get_batch(i, j)

        frames_batch_no_labels = frames_batch[:, :-1]
        labels = frames_batch[-1, [-1]]

        frames_batch_no_labels = torch.from_numpy(frames_batch_no_labels).to(device)
        labels = torch.from_numpy(labels).to(device)
        labels = labels.type(torch.LongTensor).to(device)

        frames_batch_no_labels = frames_batch_no_labels.reshape(batch_size, sequence_length, input_size)

        # frames_batch shape is [batch_size, sequence_length, input_size]

        # Forward pass

        outputs = model(frames_batch_no_labels)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += sequence_length
        j += sequence_length

        if (j + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{j}/{n_total_steps}], Loss: {loss.item():.4f}')
            print(predicted)

dummy_data = torch.randn(batch_size, sequence_length, input_size).to(device)
torch.onnx.export(model, dummy_data, "../NN_Models/predictor_model.onnx",
                  opset_version=9, verbose=True)  # Unity says opset_version=9 has the most support

'''
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

x = torch.randn(100, 28, 28).to(device)
torch.onnx.export(model, x, "../NN_Models/model.onnx",
            opset_version=9, verbose=True)  # Unity says opset_version=9 has the most support


#if __name__ == '__main__':


'''
