import os.path

import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter

# Device configuration
from sklearn.preprocessing import MinMaxScaler

import UNITY_CSV_PARSER
import time

now = time.strftime("%c")
log_dir = ("./runs/" + now).replace(":", ",")

writer = SummaryWriter(log_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# num_classes = num of items
num_classes = 10  # 9 items and None
batch_size = 1
learning_rate = 0.0001
num_epochs = 5

input_size = 24  # num of inputs per frame
sequence_length = 45  # num of frames in a sequence
hidden_size = 128
num_layers = 2

if not os.path.exists("models"):
    os.mkdir("models")


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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

up = UNITY_CSV_PARSER.UnityParser("../CSVs/Training_Data/finaldata2.csv", "../CSVs/Training_Data/finaldata3.csv",
                                   "../CSVs/Training_Data/finaldata4.csv", "../CSVs/Training_Data/finaldata5.csv",
                                   "../CSVS/Training_Data/finaldata7.csv", "../CSVs/Training_Data/finaldata8.csv",
                                  "../CSVs/Training_Data/fer_data.csv")
#up.normalize()
segments = up.split_data_into_segments()
segments = up.create_buckets_from_split(segments)
up.split_data_2(segments, .2, .1)

'''
frames_to_backlabel = 225
up.update_label_frames(frames_to_backlabel)
up.generate_rand()
'''
#frames_to_backlabel = 10
#up.full_update_label_frames()
#up.balance_data_set()
#up.split_data(use_training=[8,9], use_validation=[7] )

# data = torch.from_numpy(np.genfromtxt("formatted_success.csv", delimiter=";"))

def training_loop2():
    frame_timeseries_jump = 1  # if 1: [1,2,3,4] => [2,3,4,5]       if 10: [1,2,3,4] => [11,12,13,14]
    global_step_count = 0

    for epoch in range(num_epochs):
        y_true = []
        y_pred = []
        running_loss = 0.0
        running_correct = 0
        for k in range(up.training_data_size()):
            '''
            Get and format the data
            '''
            frames_batch = up.get_random_training_segment(sequence_length)
            frames_batch_no_labels = frames_batch[:, :-1]

            # scaler = MinMaxScaler((-1, 1))
            # frames_batch_no_labels = scaler.fit_transform(frames_batch_no_labels)

            labels = frames_batch[-1, [-1]]
            frames_batch_no_labels = torch.from_numpy(frames_batch_no_labels).to(device)
            labels = torch.from_numpy(labels).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            frames_batch_no_labels = frames_batch_no_labels.reshape(batch_size, sequence_length, input_size)

            # frames_batch shape is [batch_size, sequence_length, input_size]

            # Forward pass

            optimizer.zero_grad()
            outputs = model(frames_batch_no_labels)
            one_hot_labels = fn.one_hot(labels, num_classes).float()
            loss = criterion(outputs, one_hot_labels)
            loss.backward()

            # Something Jonas made
            threshold = 10
            torch.nn.utils.clip_grad_norm_(model.parameters(), threshold)

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            y_true.append(labels.item())
            y_pred.append(predicted.item())
            if (k + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{k + 1}/{up.training_data_size()}], Loss: {running_loss:.4f}')
                global_step_count += 100
                writer.add_scalar('training loss', running_loss / 100, global_step_count)
                writer.add_scalar('accuracy', running_correct / 100, global_step_count)
                running_loss = 0.0
                running_correct = 0
        print(classification_report(y_true, y_pred))
        validation()
        up.shuffle_data()
        path = './models/training_model_' + str(global_step_count)
        torch.save(model.cpu().state_dict(), path)
        model.cuda(device)

def validation():
    with torch.no_grad():
        validation_frame_timeseries_jump = 1

        n_correct = 0
        n_samples = 0

        validation_y_true = []
        validation_y_pred = []
        for output in up.validation_data:
            for segment in output:
                starting_frame = 0
                ending_frame = starting_frame + sequence_length

                frame_amount = len(segment)
                if ending_frame > frame_amount:
                    ending_frame = frame_amount - 1

                while ending_frame < frame_amount and ending_frame - starting_frame >= sequence_length:
                    validation_data = segment[starting_frame:ending_frame, :]
                    validation_data_no_labels = validation_data[:, :-1]
                    validation_data_labels = validation_data[-1, [-1]]
                    validation_data_labels = torch.from_numpy(validation_data_labels).to(device)
                    validation_data_labels = validation_data_labels.type(torch.LongTensor).to(device)
                    validation_data_no_labels = torch.from_numpy(validation_data_no_labels).to(device)
                    validation_data_no_labels = validation_data_no_labels.reshape(batch_size, sequence_length, input_size)

                    outputs = model(validation_data_no_labels)
                    _, predicted = torch.max(outputs.data, 1)

                    n_samples += validation_data_labels.size(0)
                    n_correct += (predicted == validation_data_labels).sum().item()

                    validation_y_true.append(validation_data_labels.item())
                    validation_y_pred.append(predicted.item())

                    starting_frame += validation_frame_timeseries_jump
                    ending_frame = starting_frame + sequence_length

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network in VALIDATION: {acc} %')
        print(classification_report(validation_y_true, validation_y_pred))

def testing():
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        testing_frame_timeseries_jump = 1

        n_correct = 0
        n_samples = 0

        testing_y_true = []
        testing_y_pred = []
        for output in up.testing_data:
            for segment in output:
                starting_frame = 0
                ending_frame = starting_frame + sequence_length

                frame_amount = len(segment)
                if ending_frame > frame_amount:
                    ending_frame = frame_amount - 1
                print(starting_frame)
                print(ending_frame)

                while ending_frame < frame_amount and ending_frame - starting_frame >= sequence_length:
                    testing_data = segment[starting_frame:ending_frame, :]
                    testing_data_no_labels = testing_data[:, :-1]
                    testing_data_labels = testing_data[-1, [-1]]
                    testing_data_labels = torch.from_numpy(testing_data_labels).to(device)
                    testing_data_labels = testing_data_labels.type(torch.LongTensor).to(device)
                    testing_data_no_labels = torch.from_numpy(testing_data_no_labels).to(device)
                    testing_data_no_labels = testing_data_no_labels.reshape(batch_size, sequence_length, input_size)

                    outputs = model(testing_data_no_labels)
                    _, predicted = torch.max(outputs.data, 1)

                    n_samples += testing_data_labels.size(0)
                    n_correct += (predicted == testing_data_labels).sum().item()

                    testing_y_true.append(testing_data_labels.item())
                    testing_y_pred.append(predicted.item())

                    starting_frame += testing_frame_timeseries_jump
                    ending_frame = starting_frame + sequence_length

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network IN TESTING: {acc} %')
        print(classification_report(testing_y_true, testing_y_pred))


training_loop2()
testing()

dummy_data = torch.randn(batch_size, sequence_length, input_size).to(device)
onnx_name = ("../NN_Models/model" + now + ".onnx").replace(":", ",")
torch.onnx.export(model, dummy_data, onnx_name,
                  opset_version=9, verbose=True)
                  # Unity says opset_version=9 has the most support
