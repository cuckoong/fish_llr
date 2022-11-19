import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from scipy import stats
from lets_plot import *

plt.style.use('science')


def gen_activity(power, day, feature, mode="Tracking", parent_dir=None):
    """
    Visualize the activity of the zebrafish.
    :param power:
    :param day:
    :param feature:
    :param mode:
    :param parent_dir:
    :return:
    """

    df_batch_list = []
    corename = '{}W-60h-{}dpf-01'.format(power, day)
    figdir = os.path.join(parent_dir, 'Figures')
    for batch in [1, 2]:
        # label
        label_dir = os.path.join(parent_dir, 'Processed_data/quantization/batch{}/labels'.format(batch))
        labels = pd.read_csv(os.path.join(label_dir, corename + '-label.csv'))

        # data
        wkdir = os.path.join(parent_dir, 'Data/{}/{}W-batch{}/'.format(mode, power, batch))
        df = pd.read_csv(os.path.join(wkdir, corename + '.csv'))
        if mode == "Tracking":
            df['time'] = df['end'] / 60 - 1
        elif mode == 'Quantization':
            df['time'] = df['end'] // 60
            df = df[['aname', 'time']+feature]
            df = df.groupby(['aname', 'time']).sum().reset_index()

        df['label'] = df['aname'].apply(lambda x: labels.loc[labels['aname'] == x, 'label'].values[0])
        # drop row where label less than 0
        df = df[df['label'] >= 0]
        df['feature'] = 0

        if len(feature) > 1:
            for item in feature:
                df['feature'] += df[item]
                df.drop(item, axis=1, inplace=True)
        else:
            df['feature'] = df[feature]

        df['aname'] = df['aname'].apply(lambda x: x + '_' + str(batch))
        df_clean = df[['aname', 'time', 'label', 'feature']].copy()

        ## TODO: How to scaling the distance?
        # median_max_activity = df.groupby(['time'])['feature'].median().max()
        # save non-scale features
        df_clean.to_csv(os.path.join(parent_dir, 'Processed_data/{}/features'.format(mode), corename +
                                     '-batch{}.csv'.format(batch)), index=False)
        # scaling features
        df_clean['feature'] = df.groupby(['aname'])['feature'].transform(lambda x: x / x.max())
        # save scaled features
        df_clean.to_csv(os.path.join(parent_dir, 'Processed_data/{}/scale_features'.format(mode), corename +
                                     '-batch{}.csv'.format(batch)), index=False)
        df_batch_list.append(df_clean)

'''
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
        sns.lineplot(x='time', y='feature', data=df_label, label=label_name[label], ax=ax, ci=None,
                     color=colors[label], estimator=np.median)
        # sns.lineplot(x='time', y='feature', data=df_label, style='aname', ax=ax,
        #              legend=False, ci=None, alpha=0.1, color=colors[label])
        bounds = df_label.groupby('time')['feature'].quantile((0.25, 0.75)).unstack()
        # iqr = bounds.iloc[:, 1] - bounds.iloc[:, 0]
        # bounds.iloc[:, 1] += 1.5 * iqr
        # bounds.iloc[:, 0] -= 1.5 * iqr
        ax.fill_between(x=bounds.index, y1=bounds.iloc[:, 0], y2=bounds.iloc[:, 1], color=colors[label],
                        alpha=0.5)

    # add vertical lines
    for i in [29, 59, 89, 119]:
        plt.axvline(i, color='k', linestyle='--')
    # add retangles
    for i in [16, 45, 76, 105, 130]:
        light_idx = i % 2
        ax.text(i, 0.9, lights[light_idx], ha='center', va='center', transform=tform)
    plt.show()
    # save figure
    fig.savefig(os.path.join(figdir, '{}W_day{}_{}.png'.format(power, day, mode)), dpi=300)
'''

if __name__ == '__main__':
    parent_dir = '/Users/panpan/PycharmProjects/old_project/fish_llr/'
    powers = [1.2]
    days = [5, 6, 7, 8]
    plate = 1
    # features_list = ['inadur', smldur', 'lardur']
    for power in powers:
        for day in days:
            gen_activity(power, day, feature=['smldist', 'lardist'], mode="Tracking", parent_dir=parent_dir)
            # visualize_activity(power, day, feature=['burdur', 'middur'], mode="Quantization", parent_dir=parent_dir)

