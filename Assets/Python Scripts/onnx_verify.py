import torch
import onnx
import onnxruntime as ort
import numpy as np
import UNITY_CSV_PARSER
import matplotlib.pyplot as plt

up = UNITY_CSV_PARSER.UnityParser("../CSVs/testing_data1.csv", "../CSVs/testing_data2.csv", "../CSVs/testing_data3.csv",
                                  "../CSVs/testing_data4.csv", keep_every=1)
# up.normalize()
segments = up.split_data_into_segments_keep_earlier()
segments = up.create_buckets_from_split(segments)
up.split_data_2(segments, 0.2, 0.1)

ort_sess = ort.InferenceSession('../NN_Models/may10model.onnx')

sequence_length = 45  # num of frames in a sequence
input_size = 24  # num of inputs per frame
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

for correct_label in range(1, 10):
    correct = 0
    shortest_seq_length = 99999
    for li in class_preds[correct_label]:
        shortest_seq_length = min(shortest_seq_length, len(li))

    print("Shortest sequence: " + str(shortest_seq_length))
    accuracy_over_frame = []
    frame = 0
    while frame < shortest_seq_length:
        for sequence in range(len(class_preds[correct_label])):
            if class_preds[correct_label][sequence][frame] == correct_label:
                correct += 1
        accuracy_over_frame.append((correct / len(class_preds[correct_label])) * 100)
        correct = 0
        frame += 1

    plt.plot(accuracy_over_frame)
    plt.show()
    plt.close()

