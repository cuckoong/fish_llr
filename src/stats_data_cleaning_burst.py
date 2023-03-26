import os

import numpy as np
import pandas as pd

'''
Generate the burst duration from raw data for statistical analysis on light-induced locomotor response
'''

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr/')
if __name__ == '__main__':
    days = [5, 6, 7, 8]
    activity_type = ['burdur']  # , 'middur'] # all activity or only one type of activity
    fish_type = 'WT'
    radiation = 3
    plates = [1]
    hour = 60
    burst_threshold = 4
    batch = 2

    feature_dir = 'Data/Quantization/{}/{}W-batch{}/'.format(fish_type, radiation, batch)
    label_dir = 'Data/Quantization/{}/{}W-batch{}/'.format(fish_type, radiation, batch)

    res_dir = f'Processed_data/quantization/{fish_type}/stat_data/'

    df_list = []

    for day in days:
        for plate in plates:
            filename = '{}W-{}h-{}dpf-0{}.csv'.format(radiation, hour, day, plate)
            df = pd.read_csv(feature_dir + filename)

            if len(activity_type) > 1:
                df['activity_sum'] = df[activity_type[0]] + df[activity_type[1]]

            else:
                df['activity_sum'] = df[activity_type[0]]

            # get duration values
            features = df[['aname', 'end', 'activity_sum']].copy(deep=True)
            features['animal'] = str(plate) + '-' + features['aname']
            features['end'] = round(features['end']).astype('int')

            label_file = '{}W-{}h-{}dpf-0{}-label.npy'.format(radiation, hour, day, plate)
            # 1: radiation; 0: control
            label = np.load(label_dir + label_file)
            # extend the label array with no-fish well
            # label = np.append(label, np.repeat(-1, 96-len(label)))
            labels = np.tile(label, int(len(features) / len(label)))
            features.sort_values(['end', 'aname'], inplace=True)
            features['label'] = labels
            features = features[features['label'] >= 0]
            features['radiation'] = radiation
            features['day'] = day
            features.loc[features['label'] == 0, 'radiation'] = 0

            # group by animal, sum of duration, check if animal freezing all the time
            print((features.groupby('aname')['activity_sum'].sum() == 0).sum())

            # keep only the animals that are not freezing all the time
            features = features[features.groupby('aname')['activity_sum'].transform('sum') > 0]

            # group by animal, sum of duration, check if animal freezing all the time
            print('after remove')
            print((features.groupby('aname')['activity_sum'].sum() == 0).sum())

            # result dir
            # res_file = 'burdur_r{}_d{}_p{}.csv'.format(radiation, day, plate)
            # features.to_csv(res_dir+res_file)

            # csv_file = r'C:\Users\xiaoliwu\PycharmProjects\Fish\stat_analysis\burdur_r{}_d{}_p{}.csv'.format(
            # radiation, day, plate) features = pd.read_csv(csv_file)
            df_list.append(features)

    frame = pd.concat(df_list, axis=0, ignore_index=True)
    if len(activity_type) == 1:
        activity_name = activity_type[0]
    else:
        activity_name = 'all'
    frame.to_csv(res_dir + '{}_{}w_{}h_batch{}_burst{}.csv'.format(activity_name, radiation, hour, batch,
                                                                   burst_threshold))
