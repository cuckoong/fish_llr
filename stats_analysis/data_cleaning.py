### This file was used to clean 5W inte_end two only

import numpy as np
import pandas as pd

'''
Generate the burst duration from raw data for statistical analysis on light-induced locomotor response
'''

if __name__ == '__main__':
    days = [5, 6, 7, 8]
    # TODO
    radiations = [5]
    plates = [1]
    hour = 60
    burst = 4
    batch = 1

    feature_dir = '/home/tmp2/PycharmProjects/fish_llr/Data/burst{}/{}W-{}h-batch{}/'.format(burst, radiations[0], hour,
                                                                                             batch)
    # feature_dir = '/home/tmp2/PycharmProjects/fish_llr/Data/burst{}/{}W-{}h/'.format(burst, radiations[0], hour)
    # label_dir = '/home/tmp2/PycharmProjects/fish_llr/Data/burst{}/{}W_{}/'.format(burst, radiations[0], plates[0])
    label_dir = '/home/tmp2/PycharmProjects/fish_llr/Data/burst{}/{}W-{}h-batch{}/'.format(burst, radiations[0], hour,
                                                                                           batch)
    res_dir = '/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/'

    df_list = []

    for day in days:
        for radiation in radiations:
            for plate in plates:
                filename = '{}W-{}h-{}dpf-0{}.csv'.format(radiation, hour, day, plate)
                df = pd.read_csv(feature_dir + filename)

                # get duration values
                features = df[['burdur', 'aname', 'end']].copy(deep=True)
                features['animal'] = str(plate) + '-'+features['aname']
                features['end'] = round(features['end']).astype('int')

                label_file = '{}W-{}h-{}dpf-0{}-label.npy'.format(radiation, hour, day, plate)
                # 1: radiation; 0: control
                label = np.load(label_dir+label_file)
                # extend the label array with no-fish well
                # label = np.append(label, np.repeat(-1, 96-len(label)))
                labels = np.tile(label, int(len(features)/len(label)))
                features.sort_values(['end', 'aname'], inplace=True)
                features['label'] = labels
                features = features[features['label'] >= 0]
                features['radiation'] = radiation
                features['day'] = day
                features.loc[features['label'] == 0, 'radiation'] = 0

                # group by animal, sum of duration
                print((features.groupby('aname')['burdur'].sum() == 0).sum())
                tmp = features.groupby('aname')['burdur'].sum()

                # result dir
                # res_file = 'burdur_r{}_d{}_p{}.csv'.format(radiation, day, plate)
                # features.to_csv(res_dir+res_file)

                # csv_file = r'C:\Users\xiaoliwu\PycharmProjects\Fish\stat_analysis\burdur_r{}_d{}_p{}.csv'.format(
                # radiation, day, plate) features = pd.read_csv(csv_file)
                df_list.append(features)

    frame = pd.concat(df_list, axis=0, ignore_index=True)
    # frame.to_csv('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/burdur_{}w_{}h_burst{}.csv'.format(
        # radiations[0], hour, burst))
    frame.to_csv('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/'+
                 'burdur_{}w_{}h_batch{}_burst{}.csv'.format(radiations[0],hour, batch, burst))


