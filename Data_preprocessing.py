import os
import numpy as np
import pandas as pd
from utilitis import *

if __name__ == '__main__':
    # Load data from excel file
    filename = '4W_3h_EM_5dpf_r2'
    dir = '/Users/panpan/PycharmProjects/FIsh/'
    format = '.csv'
    file = dir + filename + format
    data_cp = pd.read_csv(file)#, sheet_name='filename')

    # labels
    labels = [0] * 41 + [1] * 35

    metric_list = ['frect', 'fredur', 'burct', 'burdur']
    data = data_cp[metric_list].values
    data = data.reshape(-1, 96, 4)
    data = np.transpose(data, (1, 0, 2))
    data = data[:len(labels), :, :]
    data = np.concatenate((np.zeros((data.shape[0],30*60, data.shape[2])), data), axis=1)

    periods = [1, 2, 30]
    for period in periods:
        features = get_features(data, period=period)
        np.save(filename + '-'+str(period) +'min-feature.npy', features)
    np.save(filename + '-label.npy', labels)
