### This file was used to clean 5W batch One only. Not for batch two

import numpy as np
import pandas as pd

'''
Generate the burst duration from raw data for statistical analysis on light-induced locomotor response
'''

if __name__ == '__main__':
    feature_dir = '/home/tmp2/PycharmProjects/fish_llr/Data/burst4/5w_2/'
    label_dir = '/home/tmp2/PycharmProjects/fish_llr/Data/burst4/5w_2/'
    res_dir = '/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/'
    days = [5, 6, 7, 8]
    radiations = [5]
    plates = [1, 2]

    df_list = []

    for day in days:
        for radiation in radiations:
            for plate in plates:
                filename = '{}W-60h-{}dpf-0{}.csv'.format(radiation, day, plate)
                df = pd.read_csv(feature_dir + filename)

                # get duration values
                features = df[['burdur', 'aname', 'end']].copy(deep=True)
                features['animal'] = str(plate) + '-'+features['aname']
                features['end'] = round(features['end']).astype('int')

                label_file = '{}W-60h-{}dpf-0{}-label.npy'.format(radiation, day, plate)
                # 1: radiation; 0: control
                label = np.load(label_dir+label_file)

                # extend the label array with no-fish well
                # label = np.append(label, np.repeat(-1, 96-len(label)))
                labels = np.tile(label, int(len(features)/len(label)))
                features['label'] = labels
                features.drop(features[features.label < 0].index, inplace=True)
                features['radiation'] = radiation
                features['day'] = day
                features.loc[features['label'] == 0, 'radiation'] = 0

                # result dir
                res_file = 'burdur_r{}_d{}_p{}.csv'.format(radiation, day, plate)
                features.to_csv(res_dir+res_file)

                # csv_file = r'C:\Users\xiaoliwu\PycharmProjects\Fish\stat_analysis\burdur_r{}_d{}_p{}.csv'.format(
                # radiation, day, plate) features = pd.read_csv(csv_file)
                df_list.append(features)

    frame = pd.concat(df_list, axis=0, ignore_index=True)
    frame.to_csv('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/burdur_5w_batch2_burst4.csv')


