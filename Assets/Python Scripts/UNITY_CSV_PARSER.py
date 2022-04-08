import numpy as np


class UnityParser:
    def __init__(self, fname):
        with open(file=fname, newline="") as f:
            data = np.genfromtxt(f, usecols=(0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
                                 delimiter=",", skip_header=1)
            self.data = data

    def __iter__(self):
        for frame in self.data:
            yield frame

