import torch
import onnx
import onnxruntime as ort
import numpy as np
import UNITY_CSV_PARSER
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time

up = UNITY_CSV_PARSER.UnityParser("../CSVs/testing_data1.csv", "../CSVs/testing_data2.csv", "../CSVs/testing_data3.csv",
                                  "../CSVs/testing_data4.csv", keep_every=1)
# up.normalize()
segments = up.split_data_into_segments_keep_earlier()
segments = up.create_buckets_from_split(segments, randomize=False)
up.split_data_2(segments, 0.2, 0.1)

model_name = 'Gaze_only_model'
ort_sess = ort.InferenceSession('../NN_Models/' + model_name + '.onnx')

now = time.strftime("%c")
log_dir = ("./graphs2/" + model_name)
writer = SummaryWriter(log_dir)

sequence_length = 45  # num of frames in a sequence
input_size = 9  # num of inputs per frame
batch_size = 1

class_preds = [[] for i in range(10)]

def testing():
    with torch.no_grad():
        testing_frame_timeseries_jump = 1

        n_correct = 0
        n_samples = 0

        testing_y_true = []
        testing_y_pred = []
        for correct_label in range(len(up.data)):
            output = up.testing_data[correct_label]
            for segment in output:
                starting_frame = 0
                ending_frame = starting_frame + sequence_length

                frame_amount = len(segment)
                if ending_frame > frame_amount:
                    ending_frame = frame_amount - 1

                frame_pred = []

                while ending_frame < frame_amount and ending_frame - starting_frame >= sequence_length:
                    testing_data = segment[starting_frame:ending_frame, :]
                    testing_data_no_labels = testing_data[:, :-1]
                    testing_data_labels = testing_data[-1, [-1]]
                    # testing_data_labels = torch.from_numpy(testing_data_labels).to(device)
                    # testing_data_labels = testing_data_labels.type(torch.LongTensor).to(device)
                    # testing_data_no_labels = torch.from_numpy(testing_data_no_labels).to(device)
                    testing_data_no_labels = testing_data_no_labels.reshape(batch_size, sequence_length, input_size)

                    # outputs = model(testing_data_no_labels)
                    outputs = ort_sess.run(None, {'input': testing_data_no_labels})
                    predicted = outputs[0].argmax(axis=1)
                    # print(predicted)
                    # _, predicted = torch.max(outputs.data, 1)
                    frame_pred.append(int(predicted))
                    n_samples += len(testing_data_labels)
                    n_correct += (predicted == testing_data_labels).sum().item()
                    # print(str(predicted[0]) + " was prediction. " + str(testing_data_labels[0]) + " was answer.")
                    testing_y_true.append(testing_data_labels.item())
                    testing_y_pred.append(predicted.item())

                    starting_frame += testing_frame_timeseries_jump
                    ending_frame = starting_frame + sequence_length

                class_preds[correct_label].append(frame_pred)


testing()

average_length = 0

for correct_label in range(1, 10):
    correct = 0
    shortest_seq_length = 99999
    longest_seq_length = 0
    total_length = 0

    length_list = [len(x) for x in class_preds[correct_label]]
    median_length = int(np.median(length_list))

    for li in class_preds[correct_label]:
        total_length += len(li)
        shortest_seq_length = min(shortest_seq_length, len(li))
        longest_seq_length = max(longest_seq_length, len(li))

    average_length += (total_length / len(class_preds[correct_label]))

    print("Shortest sequence: " + str(shortest_seq_length))
    print("Longest sequence: " + str(longest_seq_length))
    print("Average length of sequence: " + str(average_length))
    print("Median length of sequence: " + str(median_length))
    print("-----------------------------------------------------")

average_length = int(average_length / 9)
averages = [[] for i in range(10)]

for correct_label in range(1, 10):
    frame = 0
    num_sequences = 0
    while frame < average_length:
        for sequence in range(len(class_preds[correct_label])):
            if frame < len(class_preds[correct_label][sequence]):
                correct += (class_preds[correct_label][sequence][frame] == correct_label)
                num_sequences += 1
        if num_sequences != 0:
            averages[correct_label].append(correct / num_sequences)
        #  writer.add_scalar('Accuracy/Frame. Label: ' + str(correct_label), (correct / num_sequences) * 100, frame)
        correct = 0
        num_sequences = 0
        frame += 1

print(average_length)
for seq in range(average_length):
    avg = 0
    used = 0
    for correct_label in range(1, 10):
        if len(averages[correct_label]) > seq:
            avg += averages[correct_label][seq]
            used += 1
    avg /= used
    writer.add_scalar('Accuracy/Frame', avg * 100, seq)

