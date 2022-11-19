import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy import stats

plt.style.use('science')
import seaborn as sns
from lets_plot import *


def visualize_activity(power, day, mode, choice, scale=True):
    '''
    Visualize the power of the bursts in the day.
    :param power:
    :param day:
    :param batch:
    :param choice:
    :param time_block:
    :return:
    '''
    wkdir = '/Users/panpan/PycharmProjects/old_project/fish_llr/Processed_data/{}/'.format(mode)
    figdir = '/Users/panpan/PycharmProjects/old_project/fish_llr/Figures/activity_exploration/'
    if scale:
        data_dir = 'scale_features'
    else:
        data_dir = 'raw_features'
    corename = '{}W-60h-{}dpf-01-'.format(power, day)
    df_batch_list = []

    for batch in [3]:
        df = pd.read_csv(os.path.join(wkdir, data_dir, corename + 'batch{}.csv'.format(batch)))

        # check if row has label less than 0
        assert df.loc[df['label'] < 0, 'label'].shape[0] == 0
        #
        # df_ctrl = df[df['label'] == 0]
        # df_rad = df[df['label'] == 1]
        df_batch_list.append(df)

    df_smooth = pd.concat(df_batch_list, axis=0)
    df_smooth['label_name'] = df_smooth['label'].apply(lambda x: 'Rad' if x == 1 else 'Ctrl')
    # df_smooth['feature'] = df_smooth.groupby(['aname'])['feature'].transform(lambda x: x / x.max())

    fig, ax = plt.subplots(figsize=(10, 5))
    tform = blended_transform_factory(ax.transData, ax.transAxes)
    colors = ['red', 'blue']
    lights = ['OFF', 'ON']
    label_name = ['Ctrl', 'Rad']
    for label in [0, 1]:
        df_label = df_smooth[df_smooth['label'] == label].reset_index(drop=True)
        if choice == 'median':
            estimator_name = np.median
            bounds = df_label.groupby('time')['feature'].quantile((0.25, 0.75)).unstack()
            ax.fill_between(x=bounds.index, y1=bounds.iloc[:, 0], y2=bounds.iloc[:, 1], color=colors[label],
                            alpha=0.5)
        elif choice == 'mean':
            estimator_name = np.mean
            bounds = df_label.groupby('time')['feature'].agg(['mean', 'std'])
            ax.fill_between(x=bounds.index, y1=bounds.iloc[:, 0] - bounds.iloc[:, 1],
                            y2=bounds.iloc[:, 0] + bounds.iloc[:, 1], color=colors[label], alpha=0.5)

        sns.lineplot(x='time', y='feature', data=df_label, label=label_name[label], ax=ax, ci=None,
                     color=colors[label], estimator=estimator_name)
        # sns.lineplot(x='time', y='feature', data=df_label, style='aname', ax=ax,
        #              legend=False, ci=None, alpha=0.1, color=colors[label])

        # iqr = bounds.iloc[:, 1] - bounds.iloc[:, 0]
        # bounds.iloc[:, 1] += 1.5 * iqr
        # bounds.iloc[:, 0] -= 1.5 * iqr

    # add vertical lines
    for i in [29, 59, 89, 119]:
        plt.axvline(i, color='k', linestyle='--')
    # add retangles
    for i in [16, 45, 76, 105, 130]:
        light_idx = i % 2
        ax.text(i, 0.9, lights[light_idx], ha='center', va='center', transform=tform)
    # plt.show()
    # save figure
    fig.savefig(os.path.join(figdir, '{}W_day{}_{}.png'.format(power, day, mode)), dpi=300)

    '''
    # df_std = df.groupby(['time']).std().reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
    tform = blended_transform_factory(ax.transData, ax.transAxes)
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,5))
    # axs = axs.flatten()
    df_list = [df_ctrl, df_rad]
    group_type = ['ctrl', 'rad']
    lights = ['OFF', 'ON']

    # df_smooth = df.copy()
    # # df_smooth['time'] = (df_smooth['time'] - 30 ) % 60 # smooth every 3 minutes
    # # df_smooth = df_smooth.groupby(['aname', 'time']).mean().reset_index()
    # # for each aname, do minmaxscaler for burdur
    # df_smooth['burdur'] = df_smooth.groupby(['aname'])['burdur'].transform(lambda x: x / x.max())
    # p1 = ggplot(df_smooth) + geom_line(aes(x='time', y='burdur',
    #                                        group='aname', color='aname'))
    # p1.show()
    #
    # sns.lineplot(x='time', y='burdur', hue='label', data=df_smooth, estimator=np.median, ci=95)
    # plt.show()

    if choice == 'individual':
        sns.lineplot(x='time', y='burdur', hue='aname', data=df, ax=ax, legend=False)
    else:
        for item in range(len(df_list)):
            # subplots
            df_select = df_list[item]

            if choice == 'median':
                # plot mean burdur data
                df_median = df_select.groupby(['time']).median().reset_index()
                # 25% and 75% quantile
                df_25 = df_select.groupby(['time'])['feature'].quantile(0.25).reset_index()
                df_75 = df_select.groupby(['time'])['feature'].quantile(0.75).reset_index()
                plt.plot(df_median['time'], df_median['feature'], '.', label=group_type[item])
                plt.fill_between(df_median['time'], df_25['feature'], df_75['feature'], alpha=0.2)
                # show legend
                plt.legend()

            elif choice == 'mean':
                # plot mean burdur data
                df_mean = df_select.groupby(['time']).mean().reset_index()
                df_std = df_select.groupby(['time']).std().reset_index()
                plt.plot(df_mean['time'], df_mean['burdur'], '.', label=group_type[item])
                plt.fill_between(df_mean['time'], df_mean['burdur'] - df_std['burdur'],
                                 df_mean['burdur'] + df_std['burdur'], alpha=0.2)
                # show legend
                plt.legend()

            # add vertical lines
            for i in [29, 59, 89, 119]:
                plt.axvline(i, color='k', linestyle='--')
            # add retangles
            for i in [16, 45, 76, 105, 136]:
                light_idx = i % 2
                ax.text(i, 0.9, lights[light_idx], ha='center', va='center', transform=tform)

    # title for the figure
    plt.title('{}W-60h-{}dpf-data'.format(power, day))
    plt.show()
    # save figure
    plt.savefig(os.path.join(figdir, 'batch{}'.format(batch), '{}W-60h-{}dpf-01-data.png'.format(power, day)),
                dpi=300)
    '''


if __name__ == '__main__':
    powers = [1.2] # [0, 1.2, 3, 5]
    days = [5, 6, 7, 8]
    for power in powers:
        for day in days:
            visualize_activity(power, day, mode='quantization', choice='median', scale=False)
            # visualize_activity(power, day, mode='tracking', choice='median', scale=False)
