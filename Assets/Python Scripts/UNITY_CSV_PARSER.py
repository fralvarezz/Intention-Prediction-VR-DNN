import numpy as np
from os.path import exists
import random

from sklearn.preprocessing import MinMaxScaler


def conv(x):
    return x.replace('.', '').replace(',', '.').encode()


class UnityParser:
    def __init__(self, *args, **kwargs):
        self.i = len(args)
        self.data = []
        for fname in args:
            with open(file=fname, newline="") as f:
                data = np.genfromtxt((conv(x) for x in f), usecols=(
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
                    delimiter=";", skip_header=1, dtype=np.float32)


                data = data[~np.isnan(data).any(axis=1), :]
                data = data[~np.isinf(data).any(axis=1), :]
                if "keep_every" in kwargs:
                    keep_every = kwargs["keep_every"]
                    data = data[::keep_every, :]

                norm_data = data[:, :-1]
                scaler = MinMaxScaler((-1, 1))
                norm_data = scaler.fit_transform(norm_data)
                data[:, :-1] = norm_data
                self.data.append(data)

    def __iter__(self):
        for frame in self.data:
            yield frame

    def __getitem__(self, item):
        return self.data[item]

    def shuffle_data(self):
        random.shuffle(self.data)

    def get_total_it_count(self):
        n = 0
        for d in self.data:
            n += len(d)
        return n

    def update_label_frames(self, backtrack_frame_count):
        cur_data_slice = 0
        for data in self.data:
            n = len(data) - 1
            cur_backtrack_count = 0
            is_backtracking = False
            used_tag = 0
            for frame in reversed(data):
                if is_backtracking:
                    if self.data[cur_data_slice][n][-1] == 0:
                        self.data[cur_data_slice][n][-1] = used_tag
                        cur_backtrack_count += 1
                    else:
                        print("Found nonezero while backtracking, backtrack_frame_count probably too high!")
                        is_backtracking = False
                if frame[-1] != 0 and not is_backtracking:
                    is_backtracking = True
                    cur_backtrack_count = 0
                    used_tag = frame[-1]
                if cur_backtrack_count >= backtrack_frame_count:
                    is_backtracking = False
                    cur_backtrack_count = 0
                n -= 1
            cur_data_slice += 1

    def full_update_label_frames(self):
        cur_data_slice = 0
        for data in self.data:
            used_tag = 0
            n = len(data) - 1
            for frame in reversed(data):
                if self.data[cur_data_slice][n][-1] == 0:
                    self.data[cur_data_slice][n][-1] = used_tag
                else:
                    used_tag = self.data[cur_data_slice][n][-1]
                n -= 1
            cur_data_slice += 1

    def save_to_disc(self, path):
        #  Saves ALL csv files to disk
        if exists(path):
            print("path:" + path + " already exists!")
            exit(1)
        with open(path, "a") as f:
            for data in self.data:
                np.savetxt(f, data, delimiter=";", fmt='%f')

    def get_batch(self, i, j, k):
        return self.data[i][j:k]

    def __len__(self):
        return len(self.data)


#up = UnityParser("../CSVs/experiment12.csv")

#up.update_label_frames(225)

#for f in up:
    #print(f)

# up.save_to_disc("formatted_merge.csv")

#data = np.genfromtxt("formatted_success.csv", delimiter=";")
#print(data)

