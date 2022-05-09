import numpy as np


class SelectionData:

    def __init__(self, data_path, keep_every = 3):
        with open(data_path) as f:
            data = np.genfromtxt((self.fix_decimal(x) for x in f), usecols=(
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24),
                                 delimiter=";", skip_header=1, dtype=np.float32)
            #  5: player_up_x;player_up_y;player_up_z;
            #  8: rel_r_hand_x;rel_r_hand_y;rel_r_hand_z;
            # 11: r_hand_up_x;r_hand_up_y;r_hand_up_z;
            # 14: gaze_vec_x;gaze_vec_y;gaze_vec_z;
            # 17: gaze_p_x;gaze_p_y;gaze_p_z;
            # 20: obj_tag; (looking at)
            # 21: gaze_to_screen_x;gaze_to_screen_y;gaze_to_screen_z;
            # 24: obj_interacted_with (label)

        data = data[~np.isnan(data).any(axis=1), :]  # TODO is it really necessary?
        data = data[~np.isinf(data).any(axis=1), :]
        data = data[::keep_every, :]  # subsample

        self.data = data

    @staticmethod
    def fix_decimal(x):
        return x.replace('.', '').replace(',', '.').encode()