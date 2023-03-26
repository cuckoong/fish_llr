import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import num_rest_active_bout

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')


def group_batch_data(activity, power, batches):
    df_list = []
    # group batch together
    for batch in batches:
        file_batch = os.path.join(f'Processed_data/quantization/{fish_type}/stat_data/',
                                  '{}_{}w_60h_batch{}_burst4.csv'.format(activity, power, batch))

        df = pd.read_csv(file_batch, index_col=0)
        # select day 5
        df = df[df['day'] == day].copy()
        df['batch'] = batch

        # split animal string to plate and location
        df['plate'] = df['animal'].apply(lambda x: x.split('-')[0])
        df['location'] = df['animal'].apply(lambda x: x.split('-')[1])

        # concat batch, plate, location to animal_id
        df['animal_id'] = df['batch'].astype(str) + '-' + df['plate'] + '-' + df['location']
        df_list.append(df)

    # concat all batches
    df_all_batches = pd.concat(df_list, axis=0).reset_index(drop=True)

    # select columns
    df_all_batches = df_all_batches[['batch', 'plate', 'location', 'end', 'activity_sum', 'label', 'animal_id']]
    return df_all_batches


def get_peri_stimulus_data(df, stim_time, pre_second, post_second):
    # select data for each stimulus from time - duration to time + duration
    df_periStim = df[(df['end'] >= stim_time - pre_second) & (df['end'] <= stim_time + post_second)].copy()

    # set dim to -1 for each location
    df_periStim['location_light'] = df_periStim['location'].copy()
    df_periStim.loc[df_periStim['end'] <= stim_time, 'location_light'] = -1

    # categorical columns
    df_periStim['label'] = df_periStim['label'].astype('category')
    return df_periStim


def rm_light_baseline(df):
    print('linear regression between activity_sum_scaled and location_light')
    ln_light = LinearRegression()
    # ONLY SELECT THE ON TIME PERIOD
    X_light_ON = df[df['location_light'] != -1]['location'].copy()
    X_light = pd.get_dummies(data=X_light_ON, drop_first=True)
    y_light_ON = df[df['location_light'] != -1]['activity_sum'].copy()

    ln_light.fit(X_light, y_light_ON)
    X_all = pd.get_dummies(data=df['location'].copy(), drop_first=True)
    df['light_effect'] = ln_light.predict(X_all)
    df['activity_sum'] = df['activity_sum'].copy() - df['light_effect'].copy()
    return df


def rm_batch_baseline(df):
    ln_batch = LinearRegression()
    X_batch = pd.get_dummies(data=df[['batch']], drop_first=True)
    ln_batch.fit(X_batch, df['activity_sum'])
    df['batch_effect'] = ln_batch.predict(X_batch)
    df['activity_sum'] = df['activity_sum'].copy() - df['batch_effect'].copy()
    return df


def rm_animal_baseline(df):
    df['baseline'] = df['end'].apply(lambda x: 0 if x <= stimulus_time else 1).values
    df_baseline = df[df['baseline'] == 0].groupby('animal_id')['activity_sum'].mean().reset_index()

    # merge baseline activity to df
    df = pd.merge(df, df_baseline, on='animal_id', how='left', suffixes=('', '_baseline'))
    df['activity_sum'] = df['activity_sum'].copy() - df['activity_sum_baseline'].copy()
    return df


def get_features(df):
    df_postStim = df.loc[df.loc[:, 'end'] > stimulus_time, ['animal_id', 'activity_sum', 'end']]
    # get mean over time (1min)
    df_postStim['end_min'] = df_postStim['end'].apply(lambda x: int((x - 1) // 60 + 1))
    df_postStim = df_postStim.groupby(['animal_id', 'end_min'])['activity_sum'].mean().reset_index()

    # to a 2d array, with animal_id as row, activity_sum as column
    df_postStim = df_postStim.pivot(index='animal_id', columns='end_min',
                                    values='activity_sum').reset_index()
    df_postStim.sort_values(by=['animal_id'], inplace=True)
    # add the 3rd dimension to the array
    df_postStim.drop('animal_id', axis=1, inplace=True)
    df_postStim = df_postStim.values.reshape(df_postStim.shape[0], df_postStim.shape[1], 1)

    features_postStim = num_rest_active_bout(df_postStim)
    features_array = np.stack(features_postStim, axis=1)
    return features_array


if __name__ == '__main__':
    ACTIVITY_TYPE = 'burdur'
    POWER = 3
    BATCH_LIST = [1, 2]
    days = [5, 6, 7, 8]
    pre_duration = 30
    post_duration = 1800  # 30 min
    stimuli_time = [1800, 3600, 5400, 7200]
    fish_type = 'WT'

    output_path = f'Processed_data/quantization/{fish_type}/ML_features/'

    for day in days:
        # save results
        features_period_list = []
        column_list = []
        label_list = []

        df_batches = group_batch_data(ACTIVITY_TYPE, POWER, BATCH_LIST)

        # =============================================================================
        # mean baseline activity for each animal
        # acute response -durations - durations post stimulus (at 1800s, 5400s),
        # =============================================================================
        for stimulus_time in stimuli_time:
            df_stimulus = get_peri_stimulus_data(df_batches, stimulus_time, pre_duration, post_duration)

            # ==== step: remove light intensity effects using linear regression method =========================
            print('linear regression between activity_sum_scaled and location_light')
            df_stimulus_rmLight = rm_light_baseline(df_stimulus)
            # ===================================================================================================

            # ==== step: remove batch effects using linear regression method =====================================
            print('linear regression between activity_sum_scaled and batch')
            df_stimulus_rmBatch = rm_batch_baseline(df_stimulus_rmLight)
            # ===================================================================================================

            # ==== step: remove baseline activity ==============================================================
            print('remove baseline activity')
            df_stimulus_rmBaseline = rm_animal_baseline(df_stimulus_rmBatch)
            # ===================================================================================================

            # =============== calculate the features ==================================================
            features_period_array = get_features(df_stimulus_rmBaseline)
            features_period_list.append(features_period_array)
            column_list.extend(
                item.format(stimulus_time) for item in
                ['total_active_{}', 'waking_{}', 'total_rest_{}', 'max_active_{}',
                 'num_rest_bout_{}', 'rest_bout_avg_{}', 'rest_latency_{}',
                 'num_active_bout_{}', 'active_bout_avg_{}', 'active_latency_{}'])

            # ===================================================================================================

        # ======== concatenate all features ==============================================================
        features = np.concatenate(features_period_list, axis=1)
        # create dataframe for features
        features_df = pd.DataFrame(features)
        features_df.columns = column_list

        df_label = df_stimulus[['animal_id', 'label']].drop_duplicates().reset_index(drop=True). \
            sort_values(by='animal_id').reset_index(drop=True)
        features_df['label'] = df_label['label'].values
        features_df['batch'] = df_label['animal_id'].map(lambda x: x.split('-')[0]).values
        features_df.to_csv(os.path.join(output_path, f'{POWER}W_day{day}_data.csv'), index=False)
        # =============================================================================
