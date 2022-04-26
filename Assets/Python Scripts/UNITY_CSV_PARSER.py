import numpy as np
from os.path import exists
import random

import torch.nn
import torch.nn.functional
from sklearn.preprocessing import MinMaxScaler


def conv(x):
    return x.replace('.', '').replace(',', '.').encode()


class UnityParser:
    def __init__(self, *args, **kwargs):
        self.i = len(args)
        self.data = []
        self.max_vals = []
        self.min_vals = []
        self.non_normalized_data = []
        self.training_data = []
        self.testing_data = []
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

                self.non_normalized_data.append(data)
                norm_data = data[:, :-1]
                scaler = MinMaxScaler((-1, 1))
                norm_data = scaler.fit_transform(norm_data)

                if len(self.min_vals) == 0:
                    self.min_vals = data.min(axis=0)
                else:
                    self.min_vals = min(list(self.min_vals), list(data.min(axis=0)))

                if len(self.max_vals) == 0:
                    self.max_vals = data.max(axis=0)
                else:
                    self.max_vals = max(list(self.max_vals), list(data.max(axis=0)))
                # normalized_col = scaler.fit_transform(norm_data[:, 17])

                # max_vals = data.max(axis=0)
                # data[:, :-1] = norm_data
                self.data.append(data)


        l = 1

    def normalize(self):
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[i])):
                for k in range(0, len(self.data[i][j])):
                    # Normalizing data on a column by column basis
                    self.data[i][j][k] = (self.data[i][j][k] - self.min_vals[k]) / (self.max_vals[k] - self.min_vals[k])

    def __iter__(self):
        for frame in self.data:
            yield frame

    def __getitem__(self, item):
        return self.data[item]

    def get_random_interval(self, interval_size):
        random_file_id = random.randint(0, len(self.data) - 1)
        random_file_len = len(self.data[random_file_id])
        random_start = random.randint(0, random_file_len - interval_size - 1)
        return self.data[random_file_id][random_start: random_start + interval_size, :]

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

    def get_training_batch(self, i, j, k):
        return self.training_data[i][j:k]

    def get_random_training_batch(self, length):
        random_file_id = random.randint(0, len(self.training_data) - 1)
        random_file_len = len(self.training_data[random_file_id])
        random_start = random.randint(0, random_file_len - length - 1)
        return self.training_data[random_file_id][random_start: random_start + length, :]

    def __len__(self):
        return len(self.data)

    def generate_rand(self, epsilon=None):
        if not epsilon:
            epsilon = random.uniform(-1, 1)
        generated_data = []
        d, nonzero_list = self.split_data_into_class_dict()
        order = list(range(1, 10))
        random.shuffle(order)
        random.shuffle(nonzero_list)
        for segment in nonzero_list:
            entry = []
            if len(segment) > 5:
                for r in segment:
                    entry.append(self.add_epsilon(r, epsilon))
            if len(entry) > 0:
                self.data.append(np.array(entry))
                self.non_normalized_data.append(np.array(entry))

            # generated_data.append(entry)
        '''for i in order:
            rows = d.get(i)
            if rows:
                for row in rows:
                    print(row)
                    new_row = []
                    for el in row[:-1]:
                        new_row.append(el + epsilon)
                    new_row.append(row[-1])
                    generated_data.append(new_row)
        generated_data = np.array(generated_data)'''
        # return generated_data

    def add_epsilon(self, row, epsilon):
        r = []
        for el in row[:-1]:
            r.append(el + epsilon)
        r.append(row[-1])
        return r

    def split_data_into_class_dict(self):
        new_data = dict()

        #sampled_data = self.data
        #sampled_data = torch.tensor(sampled_data)
        #sampled_data = torch.swapaxes(sampled_data, 1, 2)
        #interpolated = torch.nn.functional.interpolate(sampled_data, scale_factor=2, mode='bilinear')
        #interpolated = torch.swapaxes(interpolated, 1, 2).numpy()
        other_data = []
        for f in self.data:
            cur_rows = []
            v = int(f[0][-1])
            for r in f:
                value_int = int(r[-1])
                if not new_data.get(value_int):
                    new_data[value_int] = []
                if int(r[-1]) != v:
                    if v != 0:
                        other_data.append(np.array(cur_rows))
                    cur_rows = []
                    v = int(r[-1])
                cur_rows.append(r)
                new_data[value_int].append(r)
            if len(cur_rows) > 0 and v != 0:
                other_data.append(np.array(cur_rows))

        return new_data, other_data

    def split_data(self):
        for i in range(len(self.data)):
            percentage_training = 80
            training_frames = int(len(self.data[i]) * (percentage_training / 100))
            training_data = self.data[i][:training_frames, :]
            testing_data = self.non_normalized_data[i][training_frames:, :]
            self.training_data.append(training_data)
            self.testing_data.append(testing_data)

    def total_data(self):
        total_data = 0
        for i in range(len(self.data)):
            total_data += len(self.data[i])
        return total_data


#up = UnityParser("../CSVs/experiment12.csv")
# up = UnityParser("../CSVs/experiment12.csv", "../CSVs/experiment1.csv")
# up.split_data()

# print(len(up.data[0]))
# print(len(up.training_data[0]))
# print(len(up.testing_data[0]))
# print(up.get_random_interval(45))
#up.update_label_frames(225)
#up.generate_rand()
# for f in up:
# print(f)

# up.save_to_disc("formatted_merge.csv")

# data = np.genfromtxt("formatted_success.csv", delimiter=";")
# print(data)
