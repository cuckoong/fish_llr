import os
import numpy as np
import pandas as pd
from utilitis import *


def get_features(data, time=30, period=30):
    sr = 1
    bin_well = int(60 / sr)

    data_ON = data[:, :period * bin_well, :]
    data_OFF = data[:, time * bin_well: (time + period) * bin_well:, :]

    # # 2min_ON = 2 * 60 * 96 = 11520 rows
    # data_2min_ON = data_60min[:,2*bin_well,:]
    # # 2min_OFF = 2 * 60 * 96 = 11520 rows
    # data_2min_OFF = data_60min[:,30*bin_well:32*bin_well,:]

    # # 1min_ON = 1 * 60 * 96 = 5760 rows
    # data_1min_ON = data_60min[:,:1*bin_well,:]
    # # 1min_OFF = 1 * 60 * 96 = 5760 rows
    # data_1min_OFF = data_60min[:,30*bin_well:31*bin_well,:]

    ON_features = calculate_features(data_ON)
    OFF_features = calculate_features(data_OFF)

    features = np.concatenate([ON_features, OFF_features], axis=1)

    return features

def calculate_features(df):
    max_amplitude = Maximal_Amplitude(df)
    # mean_total_response = Mean_of_Total_Response(df)
    mean_active_response = Mean_of_Active_Response(df)
    rout_list = num_rest_active_bout(df)
    features_list = [np.array(i) for i in rout_list]
    features_list.append(max_amplitude)
    features_list.append(mean_active_response)
    features = np.stack(features_list, axis=1)

    return features



if __name__ == '__main__':
    # Load data from excel file
    data_cp = pd.read_csv('C:\\Users\\xiaoliwu\\PycharmProjects\\Fish\\0W-60h-5dpf-02.csv')

    # labels
    labels = [0] * 48 + [1] * 45  # 0w 60h 5dpf 02.

    # # correct mistake in the data
    # # for freeze = 0, burst = 0
    # conditions_fre = (data_cp['fredur'] > 0) & (data_cp['frect'] ==0)
    # conditions_bur = (data_cp['burdur'] > 0) & (data_cp['burct'] ==0)
    # data_cp.loc[conditions_fre, 'frect'] = 1 # replace all 0 in 'frect' to 1
    # data_cp.loc[conditions_bur, 'burct'] = 1  # replace all 0 in 'burct' to 1
    # var = data_cp.astype('float64').dtypes

    metric_list = ['frect', 'fredur', 'burct', 'burdur']
    data = data_cp[metric_list].values
    data = data.reshape(-1, 96, 4)
    data = np.transpose(data, (1, 0, 2))
    data = data[:len(labels), :, :]

    features = get_features(data, time=30, period=30)