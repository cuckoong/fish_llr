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


def bout(grouped_L, df):
    num_bout = (grouped_L[grouped_L[:, 0] == 0, 1] >= 2).sum()
    bout_list = grouped_L[(grouped_L[:, 0] == 0) & (grouped_L[:, 1] >= 2), 1]
    try:
        first_rest_bout = np.where((grouped_L[:, 0] == 0) & (grouped_L[:, 1] >= 10))[0][0]  # first bout
        rest_bout_latency = np.sum(grouped_L[:first_rest_bout, 1])
    except IndexError:
        print('no rest bout here, return length as whole length')
        rest_bout_latency = df.shape[1]
    try:
        first_active_bout = np.where((grouped_L[:, 0] == 1) & (grouped_L[:, 1] >= 2))[0][0]  # first bout
        active_bout_latency = np.sum(grouped_L[:first_active_bout, 1])
    except IndexError:
        print('no active bout here, return length as whole length')
        active_bout_latency = df.shape[1]

    if len(bout_list) > 0:
        average_bout_length = np.mean(bout_list)
        # average_bout = np.sum(bout_list)
    else:
        average_bout_length = 0
    return num_bout, average_bout_length, rest_bout_latency, active_bout_latency


# Number of Rest Bout & Active Bout
def num_rest_active_bout(df):
    bur_label = 5
    mid_label = 3

    # all activity including mid and burst
    all_active_data = df[:, :, bur_label] + df[:, :, mid_label]
    burst_data = df[:, :, bur_label]
    active_filter = burst_data > 0

    # total_active_time, including rest bout
    total_active_time = np.sum(all_active_data, axis=1)

    # not resting filter
    waking_filter = burst_data > (0.1 / 60)
    # waking activity (activity not in rest bout)
    waking_data = burst_data * waking_filter
    waking_time = np.sum(waking_data, axis=1)

    # rest filter
    rest_filter = burst_data <= (0.1 / 60)
    freeze_data = 1 - all_active_data
    # change negative number to 0
    freeze_data[freeze_data < 0] = 0
    # rest activity (activity in rest bout)
    rest_data = freeze_data * rest_filter
    # total rest activity
    total_rest_time = np.sum(rest_data, axis=1)

    num_rest_bout_list = []
    average_rest_bout_list = []
    rest_bout_latency_list = []
    active_bout_latency_list = []

    for j in range(len(df)):
        # sample entropy
        # sample_entropy = sampen(burst_data[j, :])
        # entropy_list.append(sample_entropy)

        grouped_L = [(k, sum(1 for i in g)) for k, g in groupby(waking_filter[j])]
        grouped_L = np.array(grouped_L)

        num_rest_bout, average_rest_bout_length, rest_bout_latency, active_bout_latency = bout(grouped_L, df[j, :, :])
        num_rest_bout_list.append(num_rest_bout)
        average_rest_bout_list.append(average_rest_bout_length)
        rest_bout_latency_list.append(rest_bout_latency)
        active_bout_latency_list.append(active_bout_latency)

    # list to array
    num_rest_bout_array = np.array(num_rest_bout_list)
    average_rest_bout_array = np.array(average_rest_bout_list)
    rest_bout_latency_array = np.array(rest_bout_latency_list)
    active_bout_latency_array = np.array(active_bout_latency_list)

    return total_active_time, waking_time, total_rest_time, num_rest_bout_array,\
           average_rest_bout_array, rest_bout_latency_array, active_bout_latency_array


def get_features(data, time=30, period=30, feature_func='all'):
    """

    :param data: quantization data
    :param time: ON/OFF time duration (min)s
    :param period: Time periods we use to calculate the features (min)
    :param feature_list: list of features we want to extract
    :return: features for classification
    """
    sr = 1
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
        column_list.extend(item.format(i) for item in ['total_active_time_{}', 'waking_time_{}', 'total_rest_time_{}',
                                                       'num_rest_bout_{}', 'average_rest_bout_length_{}',
                                                       'rest_bout_latency_{}', 'active_bout_latency_{}'])
    features = np.concatenate(features_period_list, axis=1)

    # create dataframe for features
    features_df = pd.DataFrame(features)
    features_df.columns = column_list

    return features_df
