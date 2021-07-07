import os
import numpy as np
import pandas as pd
from utilitis import *

if __name__ == '__main__':
    # Load data from csv file
    filename = '5W-60h-5dpf-02'
    batch = 1
    dir = '/home/tmp2/PycharmProjects/fish_llr/Data/burst4/5w_{}/'.format(batch)
    result_dir = '/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/ML_results/burst_4/batch{}/'.format(batch)
    format = '.csv'
    file = dir + filename + format
    data_cp = pd.read_csv(file)  # sheet_name='filename')

    # labels
    labels = np.load(dir + filename + '-label.npy')

    metric_list = ['frect', 'fredur', 'midct', 'middur', 'burct', 'burdur']
    data = data_cp[metric_list].values
    data = data.reshape(-1, 96, len(metric_list))
    data = np.transpose(data, (1, 0, 2))

    # remove well without fish
    data = data[labels >= 0, :, :]

    # what is this?
    # data = np.concatenate((np.zeros((data.shape[0], 30*60, data.shape[2])), data), axis=1)

    periods = [1, 2, 30]
    for period in periods:  # time (minutes) after on/off stimulus
        features = get_features(data, period=period)
        np.save(result_dir + filename + '-' + str(period) + 'min-feature.npy', features)
    np.save(result_dir + filename + '-label.npy', labels[labels >= 0])
