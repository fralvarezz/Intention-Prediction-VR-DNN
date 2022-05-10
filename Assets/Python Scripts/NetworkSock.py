import socket
import struct
import array
import sys
import signal
import onnxruntime as ort
import torch
import numpy as np
from sklearn.metrics import classification_report

frames = []


def handle(data):
    formatted_data = array.array('f')
    formatted_data.frombytes(data)
    return formatted_data.tolist()


def testing():
    ort_sess = ort.InferenceSession('../NN_Models/may10model.onnx')
    sequence_length = 45  # num of frames in a sequence
    input_size = 24  # num of inputs per frame
    batch_size = 1
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    global frames
    frames = np.array(frames, dtype=np.float32)

    with torch.no_grad():
        testing_frame_timeseries_jump = 1

        n_correct = 0
        n_samples = 0

        testing_y_true = []
        testing_y_pred = []

        # for segment in output:
        starting_frame = 0
        ending_frame = starting_frame + sequence_length

        frame_amount = len(frames)
        if ending_frame > frame_amount:
            ending_frame = frame_amount - 1
        print(starting_frame)
        print(ending_frame)

        while ending_frame < frame_amount and ending_frame - starting_frame >= sequence_length:
            testing_data = frames[starting_frame:ending_frame, :]
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


class NetworkRunner:

    def __init__(self, HOST, PORT):
        self.HOST = HOST
        self.PORT = PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ort_sess = ort.InferenceSession("../NN_Models/may10model.onnx")
        self.frames = []
        self.sequence_length = 45
        self.input_size = 24
        self.batch_size = 1
        self.connected = False

    def __del__(self):
        self.sock.close()

    def process_frames(self):
        if len(self.frames) < self.sequence_length:
            return None
        testing_data_no_labels = np.array(self.frames, dtype=np.float32)
        # testing_data_no_labels = testing_data_no_labels[:, :-1]
        testing_data_no_labels = testing_data_no_labels.reshape(self.batch_size, self.sequence_length, self.input_size)
        outputs = self.ort_sess.run(None, {'input': testing_data_no_labels})
        predicted = outputs[0].argmax(axis=1)[0]
        self.frames.pop(0)
        return predicted

    def add_frame_from_bytes(self, frame):
        formatted_data = array.array('f')
        formatted_data.frombytes(frame)
        self.frames.append(formatted_data.tolist())
        print("Received ")
        print(formatted_data.tolist())

    def start_serving(self, first_time=False):
        if first_time:
            print("Started serving on port " + str(self.PORT))
        else:
            print("Waiting for new client")
        conn, addr = self.sock.accept()
        self.connected = True
        with conn:
            print("Client connected")
            i = 0
            while self.connected:
                try:
                    print("Waiting for data")
                    data = conn.recv(96)
                    if not data:
                        self.connected = False
                        print("Client disconnected")
                        break
                    print("Adding data to frame")
                    self.add_frame_from_bytes(data)
                    print("Waiting for prediction")
                    ret = self.process_frames()
                    if ret is not None:
                        val = struct.pack('!i', ret)
                        print(ret)
                        print("Sending response to client")
                        # conn.sendall(val)
                        sent = conn.send(val)
                        print(str(sent) + " bytes sent")
                    i += 1
                    print("Processed " + str(i) + " reqs")
                except Exception as e:
                    print(e)
                    print("Connection closed! starting over")
                    self.connected = False
        self.start_serving()


if __name__ == "__main__":
    HOST, PORT = "localhost", 18500
    nr = NetworkRunner(HOST, PORT)
    nr.start_serving(True)
