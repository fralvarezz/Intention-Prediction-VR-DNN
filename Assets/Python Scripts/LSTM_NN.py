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
num_epochs = 20
batch_size = 1
learning_rate = 0.0001

input_size = 19  # num of inputs per frame
sequence_length = 45  # num of frames in a sequence
hidden_size = 128
num_layers = 2


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

up = UNITY_CSV_PARSER.UnityParser("../CSVs/experiment1.csv", "../CSVs/experiment2.csv", "../CSVs/experiment3.csv",
                                  "../CSVs/experiment4.csv", "../CSVs/experiment6.csv", "../CSVs/experiment7.csv",
                                  "../CSVs/experiment9.csv", "../CSVs/experiment10.csv", "../CSVs/experiment11.csv",
                                  "../CSVs/experiment12.csv", keep_every=3)
'''
frames_to_backlabel = 225
up.update_label_frames(frames_to_backlabel)
up.generate_rand()
'''
frames_to_backlabel = 10
up.full_update_label_frames()
# up.generate_rand()
up.split_data()
# data = torch.from_numpy(np.genfromtxt("formatted_success.csv", delimiter=";"))

# TODO: Finish below



def training_loop1():
    frame_timeseries_jump = 1  # if 1: [1,2,3,4] => [2,3,4,5]       if 10: [1,2,3,4] => [11,12,13,14]
    global_step_count = 0

    for epoch in range(num_epochs):
        data_size = len(up)
        y_true = []
        y_pred = []
        for k in range(data_size):
            running_loss = 0.0
            running_correct = 0

            starting_frame = 0
            ending_frame = starting_frame + sequence_length
            frame_amount = len(up.training_data[k])
            if ending_frame > frame_amount:
                ending_frame = frame_amount - 1

            while ending_frame < frame_amount:  # while there are frames left in the data
                '''
                Get and format the data
                '''
                frames_batch = up.get_training_batch(k, starting_frame, ending_frame)
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
                starting_frame += frame_timeseries_jump
                ending_frame = starting_frame + sequence_length

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_correct += (predicted == labels).sum().item()
                y_true.append(labels.item())
                y_pred.append(predicted.item())
                if (starting_frame + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], File [{k+1}/{data_size}], Step [{starting_frame + 1}/{frame_amount - sequence_length}], Loss: {running_loss:.4f}')
                    global_step_count += 100
                    writer.add_scalar('training loss', running_loss / 100, global_step_count)
                    writer.add_scalar('accuracy', running_correct / 100, global_step_count)
                    running_loss = 0.0
                    running_correct = 0
        print(classification_report(y_true, y_pred))
        up.shuffle_data()
        #path = './models/training_model_' + str(global_step_count)
        #torch.save(model.cpu().state_dict(), path)
        #model.cuda(device)

    writer.close()

def training_loop2():
    frame_timeseries_jump = 1  # if 1: [1,2,3,4] => [2,3,4,5]       if 10: [1,2,3,4] => [11,12,13,14]
    global_step_count = 0

    for epoch in range(num_epochs):
        y_true = []
        y_pred = []
        running_loss = 0.0
        running_correct = 0
        for k in range(up.total_data()):
            '''
            Get and format the data
            '''
            frames_batch = up.get_random_training_batch(sequence_length)
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
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{k + 1}/{up.total_data()}], Loss: {running_loss:.4f}')
                global_step_count += 100
                writer.add_scalar('training loss', running_loss / 100, global_step_count)
                writer.add_scalar('accuracy', running_correct / 100, global_step_count)
                running_loss = 0.0
                running_correct = 0
        print(classification_report(y_true, y_pred))
        up.shuffle_data()
        # path = './models/training_model_' + str(global_step_count)
        # torch.save(model.cpu().state_dict(), path)
        # model.cuda(device)

training_loop2()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

with torch.no_grad():
    testing_frame_timeseries_jump = 1

    n_correct = 0
    n_samples = 0

    training_y_true = []
    training_y_pred = []
    for file in up.testing_data:

        starting_frame = 0
        ending_frame = starting_frame + sequence_length

        frame_amount = len(file)
        if ending_frame > frame_amount:
            ending_frame = frame_amount - 1

        while ending_frame < frame_amount and ending_frame - starting_frame >= sequence_length:
            training_data = file[starting_frame:ending_frame, :]
            training_data_no_labels = training_data[:, :-1]
            training_data_labels = training_data[-1, [-1]]
            training_data_labels = torch.from_numpy(training_data_labels).to(device)
            training_data_labels = training_data_labels.type(torch.LongTensor).to(device)
            training_data_no_labels = torch.from_numpy(training_data_no_labels).to(device)
            training_data_no_labels = training_data_no_labels.reshape(batch_size, sequence_length, input_size)

            outputs = model(training_data_no_labels)
            _, predicted = torch.max(outputs.data, 1)

            n_samples += training_data_labels.size(0)
            n_correct += (predicted == training_data_labels).sum().item()

            training_y_true.append(training_data_labels.item())
            training_y_pred.append(predicted.item())

            starting_frame += testing_frame_timeseries_jump
            ending_frame = starting_frame + sequence_length

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    print(classification_report(training_y_true, training_y_pred))

dummy_data = torch.randn(batch_size, sequence_length, input_size).to(device)
torch.onnx.export(model, dummy_data, "../NN_Models/predictor_model.onnx",
                  opset_version=9, verbose=True)
                  # Unity says opset_version=9 has the most support
