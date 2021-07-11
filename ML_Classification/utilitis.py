import numpy as np
from itertools import groupby


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


def bout(grouped_L, df, mode=0):
    num_bout = (grouped_L[grouped_L[:, 0] == mode, 1] >= 2).sum()
    bout_list = grouped_L[(grouped_L[:, 0] == mode) & (grouped_L[:, 1] >= 2), 1]
    try:
        first_bout = np.where((grouped_L[:, 0] == mode) & (grouped_L[:, 1] >= 1))[0][0]  # first bout
        first_bout_length = np.sum(grouped_L[:first_bout, 1])
    except IndexError:
        if mode == 0:
            print('no rest bout here, return length as whole length')
        elif mode == 1:
            print('no active bout here, return length as whole length')
        first_bout_length = df.shape[1]
    if len(bout_list) > 0:
        average_bout = np.mean(bout_list)
    else:
        average_bout = 0
    return num_bout, average_bout, first_bout_length


# Number of Rest Bout & Active Bout
def num_rest_active_bout(df):
    bur_label = 5
    burst_data = df[:, :, bur_label]
    tmp = burst_data > 0
    num_rest_bout_list = []
    num_active_bout_list = []
    length_of_1st_rest_bout = []
    length_of_1st_active_bout = []
    average_rest_bout_list = []
    average_active_bout_list = []
    entropy_list = []

    for j in range(len(df)):
        # sample entropy
        sample_entropy = sampen(burst_data[j, :])
        entropy_list.append(sample_entropy)

        grouped_L = [(k, sum(1 for i in g)) for k, g in groupby(tmp[j])]
        grouped_L = np.array(grouped_L)

        # rest: 0, active 1
        # number of rest
        num_rest_bout, average_rest_bout, first_rest_bout_length = bout(grouped_L, df, mode=0)
        num_active_bout, average_active_bout, first_active_bout_length = bout(grouped_L, df, mode=1)

        # number of active
        num_rest_bout_list.append(num_rest_bout)
        num_active_bout_list.append(num_active_bout)

        length_of_1st_rest_bout.append(first_rest_bout_length)
        length_of_1st_active_bout.append(first_active_bout_length)

        average_rest_bout_list.append(average_rest_bout)
        average_active_bout_list.append(average_active_bout)

    return num_rest_bout_list, num_active_bout_list, \
           length_of_1st_rest_bout, length_of_1st_active_bout, \
           average_rest_bout_list, average_active_bout_list, entropy_list


def get_features(data, time=30, period=30):
    """

    :param data: quantization data
    :param time: ON/OFF time duration
    :param period: Time periods we use to calculate the features
    :return: features for classification
    """
    sr = 1
    bin_well = int(60 / sr)

    if period == 'baseline':
        win_start = 0
        win_end = 30 * bin_well
        data_period = data[:, win_start: win_end, :]
        features = calculate_features(data_period)

    else:
        # Two rounds of ON/OFF
        # Baseline-ON-OFF-ON-OFF
        features_period_list = []
        for i in range(1, 5):
            win_start = i * 30 * bin_well
            win_end = i * 30 * bin_well + period * bin_well
            data_period = data[:, win_start: win_end, :]
            features_period = calculate_features(data_period)
            features_period_list.append(features_period)
        features = np.concatenate(features_period_list, axis=1)

    return features


def calculate_features(df):
    max_amplitude = Maximal_Amplitude(df)
    mean_total_response = Mean_of_Total_Response(df)
    mean_active_response = Mean_of_Active_Response(df)
    rout_list = num_rest_active_bout(df)
    features_list = [np.array(i) for i in rout_list]
    features_list.append(max_amplitude)
    features_list.append(mean_total_response)
    features_list.append(mean_active_response)
    features = np.stack(features_list, axis=1)
    return features
