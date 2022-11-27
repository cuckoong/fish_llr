import numpy as np
import pandas as pd
import os

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')


if __name__ == '__main__':
    exp = '1.2W'
    days = [5, 6, 7, 8]
    batch = 2
    plate = 1
    for day in days:
        file = 'Data/Quantization/Tg/{}-batch{}/{}-60h-{}dpf-0{}.csv'.format(exp, batch, exp, day, plate)
        data = pd.read_csv(file)
        # control: 0
        # exp: 1
        # null: -1

        label_list = [0, 1]
        count_label_list = [48, 48]

        # label_list = [0, 1, -1]  # first 3 rows are control; then 3 rows are exp; then 3 rows are null
        # count_label_list = [36, 36, 24]  # 1.2W-60H-batch3

        # label_list = [1, 0]
        # count_label_list = [48, 48] # 5W-60H-1

        # label_list = [0, -1, 0, -1, 0, -1, 1, -1, 1]
        # count_label_list = [2, 1, 37, 1, 9, 9, 34, 1, 2] # 3W-60H-1

        # label_list =       [0,1]
        # count_label_list = [48,48] # 3W-60H-2
        #
        # label_list = [1, -1, 1, 0]
        # count_label_list = [4, 1, 43, 48] # 0W-60H-2

        # label_list = [1, 0]
        # count_label_list = [48, 48] # 0W-60H-1

        # label_list =       [0, -1, 0, 0, -1, 0, -1, 0,  0, -1, 0, -1, 0, 0, -1, 0, -1,
        #                     1, -1, 1, 1, 1, -1, 1, -1, -1]
        # count_label_list = [3,  1, 8, 5,  1, 2,  1, 3,  5,  1, 3,  1, 2, 2,  1, 6, 3,
        #                     4, 2, 6, 12, 5, 2, 4, 1, 12] # 1.2W-60H-2-plate2

        # label_list =       [0,-1, 0, -1, 0,
        #                     0,-1, 0, -1, 0,
        #                     0, -1, 0,
        #                     0, -1, 0,
        #                     1, -1, 1, -1,
        #                     1, -1, 1,
        #                     1, -1, 1, -1, 1,
        #                     1, -1, 1, -1, 1]
        #
        # count_label_list = [4, 1, 2,  2, 3,
        #                     2, 1, 4, 1, 4,
        #                     5, 1, 6,
        #                     6, 2, 4,
        #                     7, 1, 3, 1,
        #                     10, 1, 1,
        #                     5, 1, 1, 1, 4,
        #                     1, 1, 5, 1, 4] # 1.2W-60H-2-plate1

        # label_list =      [0, -1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 1, 1, -1, 1, -1]
        # count_label_list = [2, 1, 9, 3, 1, 3, 1, 4, 24, 2, 10, 24, 6, 2, 1, 3] # 1.2W-60H-1

        # label_list = [-1, 0, -1, 0, -1, 1, -1, 1]
        # count_label_list = [31, 5, 43, 17] # 3W-24H

        # label_list =  [1, -1, 1,  0, -1, 0, -1, 0, 0, 0, -1, 0 ]
        # count_label_list = [36, 1, 11, 2, 1, 7, 1,   1, 24, 5, 1, 6 ] # 0W-60H

        assert sum(count_label_list) == 96

        labels = []

        for label, count in zip(label_list, count_label_list):
            labels.extend([label] * count)

        np.save(file[:-4] + '-label.npy', labels)
