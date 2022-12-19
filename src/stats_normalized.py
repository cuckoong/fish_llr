import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

from sklearn.linear_model import LinearRegression

sns.set_style('whitegrid')
sns.set_context('paper')
sns.set_palette('Set2')

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')


def hotelling_t2(df, stim_time, is_before):
    if is_before:
        df_slicing = df[df['end'] < stim_time].copy()
    else:
        df_slicing = df[df['end'] > stim_time].copy()

    # comparison between control and sar
    compare_group = ['Control', '2W/Kg SAR']

    control = narrow_2_wide(df_slicing, compare_group[0])
    radiation = narrow_2_wide(df_slicing, compare_group[1])

    # Hotelling’s T-squared (https://pingouin-stats.org/generated/pingouin.multivariate_ttest.html)
    res = pg.multivariate_ttest(control, radiation)

    score = res.F.values[0]
    p_value = res.pval.values[0]

    return {'score': score, 'p_value': p_value}


def narrow_2_wide(df, compare_group_name):
    narrow_df = df[df['label'].isin([compare_group_name])][['activity_sum', 'end', 'animal_id']]
    wide_df = narrow_df.pivot(index='animal_id', columns='end', values='activity_sum')
    return wide_df


if __name__ == '__main__':

    ACTIVITY_TYPE = 'burdur'
    POWER = 1.2
    batches = [1, 2]
    days = [5, 6, 7, 8]
    pre_duration = 30
    post_durations = [20]
    colors = ['orange', 'blue']

    for post_duration in post_durations:
        for day in days:
            df_batches = []
            for batch in batches:
                file_batch = os.path.join('Processed_data/quantization/Tg/stat_data/',
                                          '{}_{}w_60h_batch{}_burst4.csv'.format(ACTIVITY_TYPE, POWER, batch))

                df = pd.read_csv(file_batch, index_col=0)
                # select day 5
                df = df[df['day'] == day]
                df['batch'] = batch

                # split animal string to plate and location
                df['plate'] = df['animal'].apply(lambda x: x.split('-')[0])
                df['location'] = df['animal'].apply(lambda x: x.split('-')[1])

                # concat batch, plate, location to animal_id
                df['animal_id'] = df['batch'].astype(str) + '-' + df['plate'] + '-' + df['location']
                df_batches.append(df)

            # concat all batches
            df_batches = pd.concat(df_batches, axis=0).reset_index(drop=True)

            # select columns
            df_batches = df_batches[['batch', 'plate', 'location', 'end', 'activity_sum', 'label', 'animal_id']]

            # =============================================================================
            # mean baseline activity for each animal
            # acute response -durations - durations post stimulus (at 1800s, 5400s),
            stimuli_time = [1800, 3600, 5400, 7200]

            # plot the activity_sum along to end, label as color, shown legend, group by label
            # subplot
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            for i, ax in enumerate(axes.flatten()):
                stimulus_time = stimuli_time[i]

                # select data for each stimulus from time - duration to time + duration
                df_stimulus = df_batches[(df_batches['end'] >= stimulus_time - pre_duration) &
                                         (df_batches['end'] <= stimulus_time + post_duration)].copy()

                # set dim to -1 for each location
                df_stimulus['location_light'] = df_stimulus['location'].copy()
                df_stimulus.loc[df_stimulus['end'] <= stimulus_time, 'location_light'] = -1

                # categorical columns
                df_stimulus['label'] = df_stimulus['label'].apply(lambda x: 'Control' if x == 0 else '2W/Kg SAR')
                df_stimulus['label'] = df_stimulus['label'].astype('category')

                # categorize label column ['batch', 'plate', 'location', 'label']
                for col in ['batch', 'plate', 'location_light', 'animal_id']:
                    df_stimulus[col] = df_stimulus[col].astype('category')
                    # df_stimulus[col] = df_stimulus[col].cat.codes

                # ==== step: remove light intensity effects using linear regression method =========================
                ln_light = LinearRegression()

                # ONLY SELECT THE ON TIME PERIOD
                X_light_ON = df_stimulus[df_stimulus['location_light'] != -1]['location'].copy()
                X_light = pd.get_dummies(data=X_light_ON, drop_first=True)
                y_light_ON = df_stimulus[df_stimulus['location_light'] != -1]['activity_sum'].copy()

                ln_light.fit(X_light, y_light_ON)
                X_all = pd.get_dummies(data=df_stimulus['location'].copy(), drop_first=True)
                df_stimulus['light_effect'] = ln_light.predict(X_all)
                df_stimulus['activity_sum'] = df_stimulus['activity_sum'].copy() - df_stimulus['light_effect']
                # ===================================================================================================

                # ==== step: remove batch effects using linear regression method =====================================
                # linear regression between activity_sum_scaled and batch
                ln_batch = LinearRegression()
                X_batch = pd.get_dummies(data=df_stimulus[['batch']], drop_first=True)
                ln_batch.fit(X_batch, df_stimulus['activity_sum'])
                df_stimulus['batch_effect'] = ln_batch.predict(X_batch)
                df_stimulus['activity_sum'] = df_stimulus['activity_sum'].copy() - df_stimulus['batch_effect']
                # ===================================================================================================

                # ==== step: remove baseline activity ==============================================================
                df_stimulus['baseline'] = df_stimulus['end'].apply(lambda x: 0 if x <= stimulus_time else 1).values
                df_baseline = df_stimulus[df_stimulus['baseline'] == 0].groupby('animal_id')[
                    'activity_sum'].mean().reset_index()

                # merge baseline activity to df
                df_stimulus = pd.merge(df_stimulus, df_baseline, on='animal_id', how='left', suffixes=('', '_baseline'))
                df_stimulus['activity_sum'] = df_stimulus['activity_sum'].copy() - df_stimulus['activity_sum_baseline']
                # ===================================================================================================

                # ======== Hotelling’s T-squared test for before and after stimulus =================================
                # before stimulus
                before_res = hotelling_t2(df_stimulus, stimulus_time, is_before=True)

                # after stimulus
                after_res = hotelling_t2(df_stimulus, stimulus_time, is_before=False)
                # ===================================================================================================

                # =============== visualize the activity_sum_scaled ==================================================
                sns.lineplot(x='end', y='activity_sum', hue='label', data=df_stimulus, legend='full', ax=ax)

                # vertical line for stimulus time, and add text
                ax.axvline(x=stimulus_time, color='black', linestyle='--', linewidth=1)
                ax.text(stimulus_time, 0, 'stimulus', rotation=90, fontsize=10, alpha=0.5)

                # add p-value to before and after stimulus, center
                ax.text(stimulus_time - pre_duration / 2 - 5, 0.15, 'p= {:.3f}'.format(before_res['p_value']),
                        fontsize=10, horizontalalignment='center')
                ax.text(stimulus_time + post_duration / 2 - 5, 0.15, 'p= {:.3f}'.format(after_res['p_value']),
                        fontsize=10, horizontalalignment='center')

                # add ON/OFF indication in plot, text in bold
                on_off_list = ['OFF', 'ON']
                ax.text(stimulus_time - pre_duration / 2 - 5, 0.18, on_off_list[i % 2],
                        fontsize=14, fontweight='bold', horizontalalignment='center')
                ax.text(stimulus_time + post_duration / 2 - 5, 0.18, on_off_list[(i + 1) % 2],
                        fontsize=14, fontweight='bold', horizontalalignment='center')

                # legend without frame
                ax.legend(title='')

                # set y axis labels
                ax.set_ylabel('Normalized Burst Activity ')
                ax.set_xlabel('Time (s)')

                # y axis limit
                ax.set_ylim(-0.05, 0.26)

                # ===================================================================================================

            # set title for whole figure
            fig.suptitle('Day {} - {} - {}w - batch {}'.format(day, ACTIVITY_TYPE, POWER, batch))

            plt.tight_layout()
            plt.savefig('Figures/Stats/Quantization/Tg/burst/acute_response/'
                        '{}_{}w_60h_day{}_post_duration{}_burst4.png'.format(ACTIVITY_TYPE, POWER, day, post_duration),
                        dpi=300)

            # =============================================================================
