import torch
import onnx
import onnxruntime as ort
import numpy as np
import UNITY_CSV_PARSER
from sklearn.metrics import classification_report


up = UNITY_CSV_PARSER.UnityParser("../CSVs/Training_Data/finaldata2.csv", "../CSVs/Training_Data/finaldata3.csv",
                                  "../CSVs/Training_Data/finaldata4.csv", "../CSVs/Training_Data/finaldata5.csv",
                                  "../CSVS/Training_Data/finaldata7.csv", "../CSVs/Training_Data/finaldata8.csv",
                                  "../CSVs/Training_Data/fer_data.csv",  keep_every=1)
# up.normalize()
segments = up.split_data_into_segments_keep_earlier()
segments = up.create_buckets_from_split(segments)
up.split_data_2(segments, .2, .1)


ort_sess = ort.InferenceSession('../NN_Models/may10model.onnx')

sequence_length = 45  # num of frames in a sequence
input_size = 24  # num of inputs per frame
batch_size = 1

class_preds = [[] for i in range(10)]


def testing():
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)

    global class_preds

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
                # print(starting_frame)
                # print(ending_frame)

                frame_pred = []
                frame_correct = []

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
                    frame_correct.append(int(testing_data_labels[0]))
                    n_samples += len(testing_data_labels)
                    n_correct += (predicted == testing_data_labels).sum().item()
                    # print(str(predicted[0]) + " was prediction. " + str(testing_data_labels[0]) + " was answer.")
                    testing_y_true.append(testing_data_labels.item())
                    testing_y_pred.append(predicted.item())

                    starting_frame += testing_frame_timeseries_jump
                    ending_frame = starting_frame + sequence_length

                print("-------------")
                print(frame_pred)
                print(frame_correct)
                cnt = 0
                for kv in zip(frame_pred, frame_correct):
                    if kv[0] == kv[1]:
                        cnt += 1

                print("accuracy:")
                print(cnt / len(frame_correct))
                print(len(frame_correct))

                class_preds[frame_correct[0]].append(frame_pred)

                print("-------------")

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network IN TESTING: {acc} %')
        print(classification_report(testing_y_true, testing_y_pred))



testing()

#print(class_preds)

class_one_acc = 0



for i in range(1, 9):
    cnt = 0
    longest_list_len = 999
    for li in class_preds[i]:
        longest_list_len = min(longest_list_len, len(li))

    accuracy_over_frame = []

    idx = 0
    while idx < longest_list_len:
        for j in range(len(class_preds[i])):
            if class_preds[i][j][idx] == i:
                cnt += 1
        accuracy_over_frame.append(cnt/len(class_preds[i]))
        cnt = 0
        idx += 1

    print(accuracy_over_frame)




import matplotlib.pyplot as plt

plt.plot(class_preds[1][0][0])
plt.xlabel("frame")
plt.ylabel("pred")

plt.plot(class_preds[1][0][1])

plt.show()