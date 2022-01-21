import numpy as np
import pandas as pd


if __name__ == '__main__':
    exp = '1.2W-60h'
    days = [5, 6, 7, 8]
    batch = 1
    for day in days:
        file = '/home/tmp2/PycharmProjects/fish_llr/Data/burst4/{}-batch{}/{}-{}dpf-01.csv'.format(exp, batch, exp, day)
        data = pd.read_csv(file)
        # control: 0
        # exp: 1
        # null: -1

        # label_list = [0, -1, 0, -1, 0, -1, 1, -1, 1]
        # count_label_list = [2, 1, 37, 1, 9, 9, 34, 1, 2] # 3W-60H-1

        # label_list =       [0,1]
        # count_label_list = [48,48] # 3W-60H-2

        label_list =      [0, -1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 1, 1, -1, 1, -1]
        count_label_list = [2, 1, 9, 3, 1, 3, 1, 4, 24, 2, 10, 24, 6, 2, 1, 3] # 1.2W-60H-1

        # label_list = [-1, 0, -1, 0, -1, 1, -1, 1]
        # count_label_list = [31, 5, 43, 17] # 3W-24H

        # label_list =  [1, -1, 1,  0, -1, 0, -1, 0, 0, 0, -1, 0 ]
        # count_label_list = [36, 1, 11, 2, 1, 7, 1,   1, 24, 5, 1, 6 ] # 0W-60H

        assert sum(count_label_list) == 96

        labels = []

        for label, count in zip(label_list, count_label_list):
            labels.extend([label]*count)

        np.save(file[:-4]+'-label.npy', labels)