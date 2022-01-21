import os
import numpy as np
import pandas as pd
from utilitis import *

wk_dir = '/home/tmp2/PycharmProjects/fish_llr/'

if __name__ == '__main__':
    # Load data from csv file
    radiation = 1.2
    batch = 1
    hour = 60
    days = [5, 6, 7, 8]
    plates = [1]
    feature_func = ['num_rest_active_bout']
    result_dir = os.path.join(wk_dir, 'Processed_data/features/')
    data_dir = os.path.join(wk_dir, 'Data/burst4/{}W-{}h-batch{}/'.format(radiation, hour, batch))
    metric_list = ['frect', 'fredur', 'midct', 'middur', 'burct', 'burdur', 'aname', 'end']

    for day in days:
        for plate in plates:
            filename = '{}W-{}h-{}dpf-0{}'.format(radiation, hour, day, plate)
            file = os.path.join(data_dir, filename + '.csv')
            # data
            data_cp = pd.read_csv(file)
            # labels
            labels = np.load(os.path.join(data_dir, filename + '-label.npy'))

            data_cp.sort_values(['end', 'aname'], inplace=True)
            data = data_cp[metric_list].values
            # data = data_cp[['frect', 'fredur', 'midct', 'middur', 'burct', 'burdur', 'aname', 'end']].values
            data = data.reshape(-1, 96, 8)
            data = np.transpose(data, (1, 0, 2))

            # remove well without fish
            data = data[labels >= 0, :, :]
            periods = [1, 2, 30]
            # periods = ['baseline', 1, 2, 30]
            for period in periods:  # time (minutes) after on/off stimulus
                features = get_features(data, period=period, feature_func=feature_func)
                np.save(result_dir + filename + '-' + str(period) + 'min-feature.npy', features)
            np.save(result_dir + filename + '-label.npy', labels[labels >= 0])
