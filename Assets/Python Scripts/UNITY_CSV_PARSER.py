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
        self.validation_data = []
        self.seq_len = 45

        for fname in args:
            with open(file=fname, newline="") as f:
                data = np.genfromtxt((conv(x) for x in f), usecols=(
                    5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29),
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

    def normalize(self):
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data[i])):
                for k in range(0, len(self.data[i][j]) - 1):
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

    def get_class_ratio(self):
        classes = np.zeros(10)
        for f in self.data:
            for r in f:
                i = int(r[-1])
                classes[i] += 1
        return classes

    def split_data_into_segments(self, omit_zero=True):
        seg_list = []
        cur_class = 0
        for f in self.data:
            seg = []
            for r in f:
                data_class = int(r[-1])
                if data_class != cur_class:
                    # cur_class = data_class
                    if len(seg) > 0:
                        if omit_zero and cur_class != 0:
                            seg_list.append(np.array(seg, dtype=np.float32))
                        elif not omit_zero:
                            seg_list.append(np.array(seg, dtype=np.float32))
                        seg = []
                    cur_class = data_class
                seg.append(r)
        return seg_list

    def split_data_into_segments_keep_earlier(self):
        seg_list = []
        cur_class = 0
        first_idx = 0
        for f in self.data:
            seg = []
            f_len = len(f)
            for i in range(0, f_len):
                r = f[i]
                data_class = int(r[-1])
                if data_class != cur_class:
                    if len(seg) > 0:
                        if cur_class != 0:
                            seg = f[first_idx:i]
                            seg_list.append(np.array(seg, dtype=np.float32))
                        seg = []
                        first_idx = max(0, i - self.seq_len)
                    cur_class = data_class
                seg.append(r)
        return seg_list


    def create_buckets_from_split(self, splitted, randomize=True):
        sub_lists = [[] for x in range(10)]
        for li in splitted:
            sub_lists[int(li[-1][-1])].append(li)
        if randomize:
            for sub_li in sub_lists:
                random.shuffle(sub_li)
        return sub_lists

    def cnt_nonzero_entries(self):
        cnt = 0
        for f in self.data:
            for r in f:
                if int(r[-1]) != 0:
                    cnt += 1
        return cnt

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
                        print("Found nonezero while backtracking, backtrack_frame_count probably too high!")
                    else:
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

    def get_training_batch(self, i, j):
        return self.training_data[i][j]

    def get_random_training_segment(self, length=0):
        random_output = random.randint(1, len(self.training_data) - 1)
        random_segment = random.randint(1, len(self.training_data[random_output]) - 1)
        segment_len = len(self.training_data[random_output][random_segment])
        random_start = random.randint(0, segment_len - length - 1)
        if length == 0:
            return self.training_data[random_output][random_segment][:, :]
        else:
            return self.training_data[random_output][random_segment][random_start: random_start + length, :]

    def __len__(self):
        return len(self.data)

    def generate_rand(self, epsilon=None, single_class=None):
        if not epsilon:
            epsilon = random.uniform(-1, 1)
        generated_data = []
        d, nonzero_list = self.split_data_into_class_dict()
        if single_class:
            entry = []
            for r in d[single_class]:
                entry.append(self.add_epsilon(r, epsilon))
            if len(entry) > 0:
                self.data.append(np.array(entry, dtype=np.float32))
                self.non_normalized_data.append(np.array(entry, dtype=np.float32))
                return
        order = list(range(1, 10))
        random.shuffle(order)
        random.shuffle(nonzero_list)
        for segment in nonzero_list:
            entry = []
            if len(segment) > 5:
                for r in segment:
                    entry.append(self.add_epsilon(r, epsilon))
            if len(entry) > 0:
                self.data.append(np.array(entry, dtype=np.float32))
                self.non_normalized_data.append(np.array(entry, dtype=np.float32))

    def add_epsilon(self, row, epsilon):
        r = []
        for el in row[:-1]:
            r.append(el + epsilon)
        r.append(row[-1])
        return r

    def split_data_into_class_dict(self):
        new_data = dict()
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

    def split_data_2(self, splitted_data, test_percent, validation_percent):
        if len(splitted_data) == 0:
            return
        data_cnt = len(splitted_data[1])
        test_cnt = int(test_percent * data_cnt)
        validation_cnt = int(validation_percent * data_cnt)
        for class_type in splitted_data:
            testing_set = class_type[:test_cnt]
            validation_set = class_type[test_cnt:test_cnt + validation_cnt]
            training_set = class_type[test_cnt + validation_cnt:]

            self.training_data.append(training_set)
            self.validation_data.append(validation_set)
            self.testing_data.append(testing_set)
            self.data = splitted_data
        print("splitted data")

    def split_data(self, use_training=None, use_validation=None):
        if use_training is None:
            for i in range(len(self.data)):
                percentage_training = 80
                training_frames = int(len(self.data[i]) * (percentage_training / 100))
                training_data = self.data[i][:training_frames, :]
                testing_data = self.non_normalized_data[i][training_frames:, :]
                self.training_data.append(training_data)
                self.testing_data.append(testing_data)
        else:
            all_data_idx = range(len(self.data))
            training_file_idx = [item for item in all_data_idx if
                                 item not in use_training and item not in use_validation]
            for i in all_data_idx:
                if i in training_file_idx:
                    self.training_data.append(self.data[i])
                    print("Training file: " + str(i))
                elif i in use_training:
                    self.testing_data.append(self.data[i])
                    print("Testing file: " + str(i))
                elif use_validation is not None and i in use_validation:
                    self.testing_data.append(self.data[i])
                    print("Validation file: " + str(i))

    def populate_smallest_sample(self):
        cls_ratios = self.get_class_ratio()
        min_index = np.where(cls_ratios == np.amin(cls_ratios))[0][0]
        self.generate_rand(single_class=min_index)

    def balance_data_set(self):
        max_val = max(self.get_class_ratio())
        cond = True
        while cond:
            self.populate_smallest_sample()
            if min(self.get_class_ratio()) >= max_val:
                cond = False

    def total_data(self):
        total_data = 0
        for i in range(len(self.data)):
            total_data += len(self.data[i])
        return total_data

    def training_data_size(self):
        total_data = 0
        for output in self.training_data:
            for segment in output:
                total_data += len(segment)
        return total_data
