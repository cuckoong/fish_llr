import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('paper')
sns.set_palette('Set2')

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')

if __name__ == '__main__':

    ACTIVITY_TYPE = 'burdur'
    POWER = 1.2
    batches = [1, 2]
    days = [5, 6, 7, 8]
    durations = [5, 10, 30]

    for batch in batches:
        for day in days:
            for duration in durations:

                file = os.path.join('Processed_data/quantization/Tg/stat_data/',
                                    '{}_{}w_60h_batch{}_burst4.csv'.format(ACTIVITY_TYPE, POWER, batch))
                colors = ['orange', 'blue']
                df = pd.read_csv(file)

                # select day 5
                df = df[df['day'] == day]

                # select columns
                df = df[['animal', 'end', 'activity_sum', 'label']]

                # categorize label column
                df['label'] = df['label'].apply(lambda x: 'Control' if x == 0 else '2W/Kg SAR')
                df['label'] = df['label'].astype('category')

                # =============================================================================
                # mean baseline activity for each animal
                # acute response -durations - durations post stimulus (at 1800s, 5400s),
                stimuli_time = [1800, 3600, 5400, 7200]

                # step one: plot the activity_sum along to end, label as color, shown legend, group by label
                # subplot
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                for i, ax in enumerate(axes.flatten()):
                    stimulus_time = stimuli_time[i]

                    # select data for each stimulus from time - duration to time + duration
                    df_stimulus = df[(df['end'] >= stimulus_time - duration) &
                                     (df['end'] <= stimulus_time + duration)].copy()

                    # remove baseline activity for each animal before any stimulus
                    df_stimulus['baseline'] = df_stimulus['end'].apply(lambda x: 0 if x <= stimulus_time else 1).values
                    df_baseline = df_stimulus[df_stimulus['baseline'] == 0].groupby('animal') \
                        ['activity_sum'].mean().reset_index()

                    # merge baseline activity to df
                    df_stimulus = pd.merge(df_stimulus, df_baseline, on='animal', how='left',
                                           suffixes=('', '_baseline'))

                    # step one: remove baseline activity
                    df_stimulus['activity_sum_scaled'] = df_stimulus['activity_sum'] - df_stimulus[
                        'activity_sum_baseline']

                    # step two: remove batch effects using linear regression method

                    # visualize the activity_sum_scaled
                    sns.lineplot(x='end', y='activity_sum_scaled', hue='label', data=df_stimulus,
                                 legend='full', ax=ax)

                    # vertical line for stimulus time, and add text
                    ax.axvline(x=stimulus_time, color='black', linestyle='--', linewidth=1)
                    ax.text(stimulus_time, 0, 'stimulus', rotation=90, fontsize=10, alpha=0.5)

                    # legend without frame
                    ax.legend(title='')

                    # set y axis labels
                    ax.set_ylabel('Burst Activity - mean baseline activity')
                    ax.set_xlabel('Time (s)')

                # set title for whole figure
                fig.suptitle('Day {} - {} - {}w - batch {}'.format(day, ACTIVITY_TYPE, POWER, batch))

                plt.tight_layout()
                plt.savefig('Figures/Stats/Quantization/Tg/burst/acute_response/'
                            '{}_{}w_60h_batch{}_day{}_duration{}_burst4.png'.format(ACTIVITY_TYPE, POWER,
                                                                                    batch, day, duration), dpi=300)

                # =============================================================================
