import os

import numpy as np
from itertools import groupby
import pandas as pd
from scipy.stats._mstats_basic import winsorize
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Sample Entropy
def sampen(L, m=2, r=0.2):
    N = len(L)
    B = 0.0
    A = 0.0
    # Split time series and save all templates of length m
    xmi = np.array([L[i: i + m] for i in range(N - m)])
    xmj = np.array([L[i: i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i: i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)


def bout(grouped_L, df, mode='rest'):
    if mode == 'rest':
        mode_idx = 0
    elif mode == 'active':
        mode_idx = 1
    else:
        raise ValueError('mode should be rest or active')
    num_bout = (grouped_L[grouped_L[:, 0] == mode_idx, 1] >= 1).sum()
    bout_list = grouped_L[(grouped_L[:, 0] == mode_idx) & (grouped_L[:, 1] >= 1), 1]
    try:
        first_bout = np.where((grouped_L[:, 0] == mode_idx) & (grouped_L[:, 1] >= 1))[0][0]  # first bout
        bout_latency = np.sum(grouped_L[:first_bout, 1])
    except IndexError:
        print('no rest/active bout here, return np.nan')
        bout_latency = np.nan

    if len(bout_list) > 0:
        average_bout_length = np.mean(bout_list)
        # average_bout = np.sum(bout_list)
    else:
        average_bout_length = 0
    return num_bout, average_bout_length, bout_latency


# Number of Rest Bout & Active Bout
def num_rest_active_bout(df):
    all_active_data = df[:, :, -1]
    waking_filter = all_active_data > 0.1  # 10% time is active
    waking_data = all_active_data * waking_filter
    # rest filter
    rest_filter = all_active_data <= 0.1
    # maximum activity
    max_activity = np.max(all_active_data, axis=1)
    # all activity
    total_active = np.sum(all_active_data, axis=1)
    # waking data
    waking = np.sum(waking_data, axis=1)
    # total rest activity
    total_rest = np.sum(rest_filter, axis=1)

    num_rest_bout_list = []
    num_active_bout_list = []

    average_rest_bout_list = []
    average_active_bout_list = []

    rest_bout_latency_list = []
    active_bout_latency_list = []

    for j in range(len(df)):
        # sample entropy
        # sample_entropy = sampen(burst_data[j, :])
        # entropy_list.append(sample_entropy)

        grouped_L = [(k, sum(1 for _ in g)) for k, g in groupby(waking_filter[j])]
        grouped_L = np.array(grouped_L)

        bout_rest_results = bout(grouped_L, df[j, :, :], mode='rest')
        bout_active_results = bout(grouped_L, df[j, :, :], mode='active')

        num_rest_bout, average_rest_bout_length, rest_bout_latency = bout_rest_results
        num_active_bout, average_active_bout_length, active_bout_latency = bout_active_results

        num_rest_bout_list.append(num_rest_bout)
        average_rest_bout_list.append(average_rest_bout_length)
        rest_bout_latency_list.append(rest_bout_latency)

        num_active_bout_list.append(num_active_bout)
        average_active_bout_list.append(average_active_bout_length)
        active_bout_latency_list.append(active_bout_latency)

    # list to array
    num_rest_bout_array = np.array(num_rest_bout_list)
    num_active_bout_array = np.array(num_active_bout_list)
    average_rest_bout_array = np.array(average_rest_bout_list)
    average_active_bout_array = np.array(average_active_bout_list)
    rest_bout_latency_array = np.array(rest_bout_latency_list)
    active_bout_latency_array = np.array(active_bout_latency_list)

    return total_active, waking, total_rest, max_activity, \
           num_rest_bout_array, average_rest_bout_array, rest_bout_latency_array, \
           num_active_bout_array, average_active_bout_array, active_bout_latency_array


def get_features(data, mode, period=3):
    """

    :param data: quantization data
    :param mode: 'quantization' or 'tracking'
    :param period: Time periods we use to calculate the features (min)
    :return: features for classification
    """
    # each bin is 1 min
    sr = 60
    bin_well = int(60 / sr)

    # if period == 'baseline':
    #     win_start = 0
    #     win_end = 30 * bin_well
    #     data_period = data[:, win_start: win_end, :]
    #     features = num_rest_active_bout(data_period, feature_func)

    # else:
    # Two rounds of ON/OFF
    # Baseline-ON-OFF-ON-OFF

    features_period_list = []
    column_list = []
    for i in range(1, 5):
        win_start = i * 30 * bin_well
        win_end = i * 30 * bin_well + period * bin_well
        data_period = data[:, win_start: win_end, :]
        features_period = num_rest_active_bout(data_period)
        features_period_array = np.stack(features_period, axis=1)
        features_period_list.append(features_period_array)
        column_list.extend(item.format(i) for item in ['total_active_{}', 'waking_{}', 'total_rest_{}', 'max_active_{}',
                                                       'num_rest_bout_{}', 'rest_bout_avg_{}', 'rest_latency_{}',
                                                       'num_active_bout_{}', 'active_bout_avg_{}', 'active_latency_{}'])
    features = np.concatenate(features_period_list, axis=1)

    # create dataframe for features
    features_df = pd.DataFrame(features)
    features_df.columns = column_list

    return features_df


def group_batch_data(fish_type, activity, power, batches, day):
    df_list = []
    # group batch together
    for batch in batches:
        file_batch = os.path.join(f'Processed_data/quantization/{fish_type}/stat_data/',
                                  '{}_{}w_60h_batch{}_burst4.csv'.format(activity, power, batch))

        df = pd.read_csv(file_batch, index_col=0)

        # select day
        df = df[df['day'] == day].copy()
        df['batch'] = batch

        # split animal string to plate and location
        df['plate'] = df['animal'].apply(lambda x: x.split('-')[0])
        df['location'] = df['animal'].apply(lambda x: x.split('-')[1])

        # concat batch, plate, location to animal_id
        df['animal_id'] = df['batch'].astype(str) + '-' + df['plate'] + '-' + df['location']

        # light intensity data
        df_intensities = []
        for stim_time in [1800, 3600, 5400, 7200]:
            file_intensity = os.path.join(f'Processed_data/light_intensity/{fish_type}/'
                                          f'{power}W-batch{batch}/'
                                          f'{power}W-60h-{day}dpf-01/'
                                          f'{(stim_time - 30) * 1000}_'
                                          f'{(stim_time + 30) * 1000}.csv')
            # get light intensity data
            df_intensity = pd.read_csv(file_intensity)
            df_intensities.append(df_intensity.copy())
        df_intensities = pd.concat(df_intensities, axis=0).reset_index(drop=True)

        # merge light intensity and behavior data, then append
        df = pd.merge(df, df_intensities, on=['animal_id', 'end'], how='left')
        df_list.append(df)

    # concat all batches
    df_all_batches = pd.concat(df_list, axis=0).reset_index(drop=True)

    # select columns
    df_all_batches = df_all_batches[['batch', 'plate', 'location', 'end', 'activity_sum',
                                     'label', 'animal_id', 'light_intensity']]
    return df_all_batches


def get_peri_stimulus_data(df, stim_time, pre_second, post_second):
    # select data for each stimulus from time - duration to time + duration
    df_periStim = df[(df['end'] >= stim_time - pre_second) &
                     (df['end'] <= stim_time + post_second)].copy()

    # check if light intensity column has null values
    if df_periStim['light_intensity'].isnull().values.any():
        raise ValueError('light intensity column has null values')

    # categorical columns
    df_periStim['label'] = df_periStim['label'].astype('category')
    return df_periStim


def rm_all_baseline(df, stimulus_time):
    print('linear regression between activity_sum_scaled and'
          ' light_intensity, batch, and pre_stimulus')
    # batch (1, 2) to 0, 1
    df = df.copy()
    df['batch_dummy'] = df['batch'].apply(lambda x: 0 if x == 1 else 1).values
    # convert to categorical
    df['batch_dummy'] = df['batch_dummy'].astype('category')

    # pre_stimulus
    df['baseline'] = df['end'].apply(lambda x: 0 if x <= stimulus_time else 1).values
    df_baseline = df[df['baseline'] == 0].groupby('animal_id')['activity_sum'].mean().reset_index()

    # merge baseline activity to df
    df = pd.merge(df, df_baseline, on='animal_id', how='left', suffixes=('', '_baseline'))

    # linear regression
    X = df[['light_intensity', 'batch_dummy', 'activity_sum_baseline']].copy()
    y = df['activity_sum'].copy()

    # Create a pipeline for data preprocessing and linear regression model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    # fit the pipeline
    pipe.fit(X, y)

    # get fit scores and coefficients
    print('fit score: ', pipe.score(X, y))
    print('coefficients: ', pipe.named_steps['regressor'].coef_)

    df['baseline'] = pipe.predict(X)
    df['activity_sum'] = df['activity_sum'].copy() - df['baseline'].copy()
    return df


def rm_light_baseline(df):
    print('linear regression between activity_sum_scaled and location_light')
    ln_light = LinearRegression()
    X_light = df['light_intensity'].copy().values.reshape(-1, 1)
    y_light = df['activity_sum'].copy().values.reshape(-1, 1)
    ln_light.fit(X_light, y_light)

    df['light_effect'] = ln_light.predict(X_light)
    df['activity_sum'] = df['activity_sum'].copy() - df['light_effect'].copy()
    return df


def rm_batch_baseline(df, baseline_interval):
    """
    remove batch baseline
    :param df:
    :param baseline_interval: tuple, (start, end)
    :return:
    """
    df = df.copy()
    df['batch_baseline'] = df['end'].apply(lambda x: 0 if (x <= baseline_interval[1]) &
                                                          (x >= baseline_interval[0]) else 1).values
    df_baseline_mean = df[df['batch_baseline'] == 0].groupby('batch')['activity_sum'].mean().reset_index()
    df_baseline_std = df[df['batch_baseline'] == 0].groupby('batch')['activity_sum'].std().reset_index()

    # merge baseline activity to df
    df = pd.merge(df, df_baseline_mean, on='batch', how='left', suffixes=('', '_baseline_batch_mean'))
    df = pd.merge(df, df_baseline_std, on='batch', how='left', suffixes=('', '_baseline_batch_std'))

    # check if any std is 0
    if df['activity_sum_baseline_batch_std'].isnull().values.any():
        raise ValueError('std is 0')

    df['activity_sum'] = df['activity_sum'].copy() - df['activity_sum_baseline_batch_mean'].copy()
    df['activity_sum'] = df['activity_sum'].copy() / df['activity_sum_baseline_batch_std'].copy()
    return df


def rm_animal_baseline(df, baseline_interval):
    """
    remove animal baseline
    :param df:
    :param baseline_interval: tuple, (start, end)
    :return:
    """
    df = df.copy()
    df['baseline'] = df['end'].apply(lambda x: 0 if (x <= baseline_interval[1]) &
                                                    (x >= baseline_interval[0]) else 1).values

    # winsorize transform and then do z-score for individual animal
    df_baseline_mean = df[df['baseline'] == 0].groupby('animal_id')['activity_sum']. \
        apply(lambda x: winsorize(x, limits=(0.05, 0.05)).mean()).reset_index()
    df_baseline_std = df[df['baseline'] == 0].groupby('animal_id')['activity_sum']. \
        apply(lambda x: winsorize(x, limits=(0.05, 0.05)).std()).reset_index()

    # merge baseline activity to df
    df = pd.merge(df, df_baseline_mean, on='animal_id', how='left', suffixes=('', '_baseline_mean'))
    df = pd.merge(df, df_baseline_std, on='animal_id', how='left', suffixes=('', '_baseline_std'))
    df['activity_sum'] = (df['activity_sum'].copy() - df['activity_sum_baseline_mean'])
    # df['activity_sum_scaled'] = df['activity_sum'].copy() / df['activity_sum_baseline_std']
    return df


def rm_linear_trend(df):
    df = df.copy()
    df['time'] = df['end'].copy() - df['end'].min()

    # linear regression to remove linear trend
    ln = LinearRegression()
    X = df['time'].copy().values.reshape(-1, 1)
    y = df['activity_sum'].copy().values.reshape(-1, 1)

    ln.fit(X, y)
    df['linear_trend'] = ln.predict(X)
    df['activity_sum'] = df['activity_sum'].copy() - df['linear_trend'].copy()
    return df


