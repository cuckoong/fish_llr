import numpy as np
import pandas as pd
from itertools import groupby

def Maximal_Amplitude(df):
    data = df[:, :, 3]
    max_burdur = np.max(data, axis=1)
    return max_burdur


# Mean of Total Response
def Mean_of_Total_Response(df):
    # be carefule when busrt is set to 0.
    data = np.sum(df, axis=1)
    mean_total_response = data[:,3]/df.shape[1]
    return mean_total_response

# Mean of the Active Response
def Mean_of_Active_Response(df):
    data = np.sum(df, axis=1)
    mean_active_response = data[:,3]/df.shape[1]
    return mean_active_response

# Number of Rest Bout & Active Bout
def num_rest_active_bout(df):
    burst_data = df[:, :, 3]
    tmp = burst_data > 0
    num_rest_bout = []
    num_active_bout = []
    length_of_1st_rest_bout = []
    length_of_1st_active_bout = []
    average_rest_bout_list = []
    average_active_bout_list = []
    entropy_list = []

    for j in range(len(df)):
        # sample entropy
        sample_entropy = sampen(burst_data[j,:])
        entropy_list.append(sample_entropy)

        grouped_L = [(k, sum(1 for i in g)) for k, g in groupby(tmp[j])]
        grouped_L = np.array(grouped_L)

        # rest: 0, active 1
        # number of rest
        num_rest_bout.append((grouped_L[grouped_L[:, 0] == 0, 1] >= 2).sum())
        average_rest_bout = (grouped_L[(grouped_L[:, 0] == 0) & (grouped_L[:, 1] >= 1), 1]).mean()
        average_rest_bout_list.append(average_rest_bout)

        # number of active
        num_active_bout.append((grouped_L[grouped_L[:, 0] == 1, 1] >= 2).sum())
        if j == 20:
            average_active_bout = (grouped_L[(grouped_L[:, 0] == 1) & (grouped_L[:, 1] >= 1), 1]).mean()
        else:
            average_active_bout = (grouped_L[(grouped_L[:, 0] == 1) & (grouped_L[:, 1] >= 1), 1]).mean()
        average_active_bout_list.append(average_active_bout)

        try:
            first_rest_bout = np.where((grouped_L[:, 0] == 0) & (grouped_L[:, 1] >= 2))[0][0] # first rest_bout
            rest_bout = np.sum(grouped_L[:first_rest_bout, 1])
            # rest_bout = grouped_L[np.where((grouped_L[:, 0] == 0) & (grouped_L[:, 1] >= 2))][0, 1]
        except IndexError:
            print('no rest bout here, return length as whole length')
            rest_bout = df.shape[1]
        length_of_1st_rest_bout.append(rest_bout)

        try:
            first_active_bout = np.where((grouped_L[:, 0] == 1) & (grouped_L[:, 1] >= 2))[0][0]
            active_bout = np.sum(grouped_L[:first_active_bout, 1])
        except IndexError:
            print('no active bout here, return length as as length')
            active_bout = df.shape[1]
        length_of_1st_active_bout.append(active_bout)

    return num_rest_bout, num_active_bout, \
           length_of_1st_rest_bout, length_of_1st_active_bout, \
           average_rest_bout_list, average_active_bout_list, entropy_list


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
