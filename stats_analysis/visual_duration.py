import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    power = 1.2
    hour = 60
    batch = 1
    burst = 4
    df = pd.read_csv('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/'\
                     'burdur_{}w_{}h_batch{}_burst{}.csv'.format(power, hour, batch, burst))

    # df.end divide by 60 to get miniutes
    df['end_min'] = df['end'] // 60

    # mean burst duration for each minute, andd groupby columns
    df_sum = df.groupby(['label', 'aname', 'end_min'])['burdur'].sum().reset_index()
    # df_group = df_sum.groupby(['label','end_min'])['burdur'].median().reset_index()

    # 30 min, 60 min, 90 min, 120 min burst duration for two label
    df_sum_30_0 = df_sum[(df_sum['end_min'] == 30) & (df_sum['label'] == 0)]
    df_sum_60_0 = df_sum[(df_sum['end_min'] == 60) & (df_sum['label'] == 0)]
    df_sum_90_0 = df_sum[(df_sum['end_min'] == 90) & (df_sum['label'] == 0)]
    df_sum_120_0 = df_sum[(df_sum['end_min'] == 120) & (df_sum['label'] == 0)]

    df_sum_30_1 = df_sum[(df_sum['end_min'] == 30) & (df_sum['label'] == 1)]
    df_sum_60_1 = df_sum[(df_sum['end_min'] == 60) & (df_sum['label'] == 1)]
    df_sum_90_1 = df_sum[(df_sum['end_min'] == 90) & (df_sum['label'] == 1)]
    df_sum_120_1 = df_sum[(df_sum['end_min'] == 120) & (df_sum['label'] == 1)]

    # df_30 = df_sum[df_sum['end_min'] == 30]
    # df_60 = df_sum[df_sum['end_min'] == 60]
    # df_90 = df_sum[df_sum['end_min'] == 90]
    # df_120 = df_sum[df_sum['end_min'] == 120]
    #
    # boxplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.boxplot([df_sum_30_0['burdur'], df_sum_30_1['burdur']])
    ax.set_xticklabels(['Control', 'Radiation'])
    ax.set_ylabel('Burst Duration (min)')
    ax.set_xlabel('')
    ax.set_title('Burst Duration')
    # y log scale
    # ax.set_yscale('symlog')
    plt.show()

    # sum of 30-31 min, 60-61min, 90-91 min, 120-121 min
    # 30 and 31 min, 60 and 61 min, 90 and 91 min, 120 and 121 min burst duration
    df_30_31 = df_sum[(df_sum['end_min'] == 30) | (df_sum['end_min'] == 31)]
    df_60_61 = df_sum[(df_sum['end_min'] == 60) | (df_sum['end_min'] == 61)]
    df_90_91 = df_sum[(df_sum['end_min'] == 90) | (df_sum['end_min'] == 91)]
    df_120_121 = df_sum[(df_sum['end_min'] == 120) | (df_sum['end_min'] == 121)]
    # boxplot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.boxplot([df_30_31['burdur'],df_90_91['burdur']])#,  df_60_61['burdur'], df_120_121['burdur']])
    # ax.set_xticklabels(['30 min', '60 min', '90 min', '120 min'])
    ax.set_ylabel('Burst Duration (s)')
    ax.set_xlabel('Burst Duration (min)')
    ax.set_title('Burst Duration')
    # y log scale
    # ax.set_yscale('symlog')
    plt.show()


    # plot burdu duration and sta for two label group
    plt.figure(figsize=(10, 5))
    plt.plot(df_group[df_group['label'] == 0]['end_min'],
             df_group[df_group['label'] == 0]['burdur'],
             'r-', label='Control')

    plt.plot(df_group[df_group['label'] == 0]['end_min'],
             df_group[df_group['label'] == 0][['burdur','std']].sum(axis=1), 'r-',
             label='Control', alpha=0.5)

    plt.plot(df_group[df_group['label'] == 1]['end_min'],
             df_group[df_group['label'] == 1]['burdur'], 'b-', label='Radiation')

    plt.plot(df_group[df_group['label'] == 1]['end_min'],
                     df_group[df_group['label'] == 1][['burdur','std']].sum(axis=1),'b-',
                     label='Radiation', alpha=0.5)
    plt.xlabel('Minute')
    plt.ylabel('Burst Duration (Minute)')
    plt.legend()
    plt.show()

