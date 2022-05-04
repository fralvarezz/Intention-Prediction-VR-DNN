import torch
import onnx
import onnxruntime as ort
import numpy as np
import UNITY_CSV_PARSER
from sklearn.metrics import classification_report


up = UNITY_CSV_PARSER.UnityParser("../CSVs/NewData/fer_data.csv", "../CSVs/NewData/jonas_data.csv",  keep_every=3)
up.normalize()
segments = up.split_data_into_segments()
segments = up.create_buckets_from_split(segments)
up.split_data_2(segments, .2, .1)


ort_sess = ort.InferenceSession('../NN_Models/seventy_percent.onnx')

sequence_length = 45  # num of frames in a sequence
input_size = 19  # num of inputs per frame
batch_size = 1


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
                    # testing_data_labels = torch.from_numpy(testing_data_labels).to(device)
                    # testing_data_labels = testing_data_labels.type(torch.LongTensor).to(device)
                    # testing_data_no_labels = torch.from_numpy(testing_data_no_labels).to(device)
                    testing_data_no_labels = testing_data_no_labels.reshape(batch_size, sequence_length, input_size)

                    # outputs = model(testing_data_no_labels)
                    outputs = ort_sess.run(None, {'input': testing_data_no_labels})
                    predicted = outputs[0].argmax(axis=1)
                    print(predicted)
                    # _, predicted = torch.max(outputs.data, 1)

                    n_samples += len(testing_data_labels)
                    n_correct += (predicted == testing_data_labels).sum().item()
                    # print(str(predicted[0]) + " was prediction. " + str(testing_data_labels[0]) + " was answer.")
                    testing_y_true.append(testing_data_labels.item())
                    testing_y_pred.append(predicted.item())

                    starting_frame += testing_frame_timeseries_jump
                    ending_frame = starting_frame + sequence_length

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network IN TESTING: {acc} %')
        print(classification_report(testing_y_true, testing_y_pred))


testing()