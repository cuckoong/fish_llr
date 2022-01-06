import os
import numpy as np
import pandas as pd
from utilitis import *

if __name__ == '__main__':
    # Load data from csv file
    radiation = 0
    batch = 1
    result_dir = '/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/ML_results/burst_4/batch{}/'.format(batch)
    dir = '/home/tmp2/PycharmProjects/fish_llr/Data/burst4/{}w_{}/'.format(radiation, batch)
    format = '.csv'
    metric_list = ['frect', 'fredur', 'midct', 'middur', 'burct', 'burdur',
                   'aname', 'end']

    for day in [5, 6, 7, 8]:
        for plate in [1, 2]:
            filename = '{}W-60h-{}dpf-0{}'.format(radiation, day, plate)
            file = dir + filename + format
            data_cp = pd.read_csv(file)  # sheet_name='filename')
            # labels
            labels = np.load(dir + filename + '-label.npy')
            # sort the value
            # get duration values
            features = data_cp[metric_list].copy(deep=True)
            features.sort_values(['end', 'aname'], inplace=True)
            features['end'] = round(features['end']).astype('int')

            data = data_cp[['frect', 'fredur', 'midct', 'middur', 'burct', 'burdur']].values
            data = data.reshape(-1, 96, 6)
            data = np.transpose(data, (1, 0, 2))

            # remove well without fish
            data = data[labels >= 0, :, :]
            periods = ['baseline', 1, 2, 30]
            for period in periods:  # time (minutes) after on/off stimulus
                features = get_features(data, period=period)
                np.save(result_dir + filename + '-' + str(period) + 'min-feature.npy', features)
            np.save(result_dir + filename + '-label.npy', labels[labels >= 0])
