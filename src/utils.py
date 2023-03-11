import numpy as np
from itertools import groupby
import pandas as pd


def Maximal_Amplitude(df):
    """
    :param df:
    :return: max burst duration
    """
    bur_label = 5
    data = df[:, :, bur_label]
    max_burdur = np.max(data, axis=1)
    return max_burdur


# Mean of Total Response
def Mean_of_Total_Response(df):
    mid_label = 3
    bur_label = 5
    # be carefule when busrt is set to 0.
    data = np.sum(df, axis=1)
    mean_total_response = (data[:, mid_label] + data[:, bur_label]) / df.shape[1]
    return mean_total_response


# Mean of the Active Response (burst response)
def Mean_of_Active_Response(df):
    bur_label = 5
    data = np.sum(df, axis=1)
    mean_active_response = data[:, bur_label] / df.shape[1]
    return mean_active_response


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
