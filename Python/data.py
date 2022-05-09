import os
import glob
import random
import numpy as np
import tqdm as tqdm
from typing import Optional, Iterable

from torch.utils.data.dataset import IterableDataset

from selection_data import SelectionData

class TrackingDataset(IterableDataset):


    def __init__(self, data_dir: str, sequence_length: int, norm_mean: Optional[np.array] = None, norm_std: Optional[np.array] = None, keep_every: int = 3):
        """
        A dataset of motion capture data read from BVH files.

        :param data_dir: The data dir containing .bvh files. Should NOT end with a '/'
        :param sequence_length: The length of the sequences.
        :param norm_mean: The mean to normalize with. If none will be computed from the data
        :param norm_std: The std to normalize with. If none will be computed from the data
        :param subsample_factor: How much to subsample the sequences.
        """

        assert not data_dir.endswith("/"), "data_dir should not end with a /"
        assert (norm_mean is not None and norm_std is not None) or (norm_mean is None and norm_std is None), "mean and std should both be provided or not at all"
        self.sequence_length = sequence_length
        cache_file = data_dir + '/cache.npz'

        if os.path.exists(cache_file):
            with np.load(cache_file, allow_pickle=True) as data:
                self.tracking = data["tracking"]

        else:
            selection_files = glob.glob(data_dir + "/*.csv")
            assert len(selection_files) > 0, "No .csv files found in " + data_dir
            self.tracking = [SelectionData(file, keep_every).data for file in tqdm.tqdm(selection_files,
                                             desc="Loading bvh files. This will only happen once. Be patient")]

            np.savez_compressed(cache_file, tracking=self.tracking)

        if norm_mean is None:
            self.norm_mean, self.norm_std = self._norm_params(self.tracking)
        else:
            self.norm_mean = norm_mean
            self.norm_std = norm_std
        self.norm_mean[-1] = 0.0 # ensure we do not normalize the label
        self.norm_std[-1] = 1.0  # class
        print(self.norm_mean)
        print(self.norm_std)
        self.tracking = self.normalize(self.tracking)

        self.frame_size = len(self.norm_mean)-1

    @staticmethod
    def _norm_params(tracking: Iterable[np.ndarray]):
        mean, std = np.concatenate(tracking).mean(axis=0), np.concatenate(tracking).std(axis=0)
        std[std < 1e-6] = 1.0 # for numerical stability + prevent div by 0
        return mean, std

    def normalize(self, tracking: Iterable[np.ndarray]):
        return [(segment - self.norm_mean) / self.norm_std for segment in tracking]

    def denormalize(self, tracking: Iterable[np.ndarray]):
        return [segment * self.norm_std + self.norm_mean for segment in tracking]

    def __iter__(self):
        while True:
            n_tracking = len(self.tracking)
            for idx in random.sample(range(n_tracking), n_tracking):
                yield self.random_sequence(self.tracking[idx])

    def random_sequence(self, selection_segment):
        start_idx = random.randint(0, len(selection_segment) - self.sequence_length)
        return selection_segment[start_idx:start_idx + self.sequence_length]

