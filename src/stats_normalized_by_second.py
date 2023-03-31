import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from utils import rm_all_baseline

sns.set_style('whitegrid')
sns.set_context('paper')
sns.set_palette('Set2')

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')


def hotelling_t2(df, stim_time, is_before, compare_group):
    if is_before:
        df_slicing = df[df['end'] < stim_time].copy()
    else:
        df_slicing = df[df['end'] > stim_time].copy()

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
    fish_type = 'Tg'
    POWER = 1
    batches = [1, 2]
    days = [5, 6, 7, 8]
    pre_duration = 30
    post_duration = 1770
    colors = ['orange', 'blue']

    if POWER == 0:  # 0W/Kg vertical, wt
        compare_group = ['Control', '0W/Kg SAR']
    elif POWER == 1:  # 1.6W/Kg vertical, tg
        compare_group = ['control', '1.6W/Kg SAR']
    elif POWER == 1.2:  # 2W/Kg vertical, wt+tg
        compare_group = ['Control', '2W/Kg SAR']
    elif POWER == 3:  # 4.9W/Kg vertical, wt
        compare_group = ['Control', '4.9W/Kg SAR']
    elif POWER == 5:  # horizontal, wt
        compare_group = ['Control', '0.009W/Kg SAR']

    for day in days:
        df_batches = []
        for batch in batches:
            # behavior data
            file_batch = os.path.join(f'Processed_data/quantization/{fish_type}/stat_data/',
                                      f'{ACTIVITY_TYPE}_{POWER}w_60h_batch{batch}_burst4.csv')

            df = pd.read_csv(file_batch, index_col=0)
            # select day 5
            df = df[df['day'] == day]
            df['batch'] = batch

            # split animal string to plate and location
            df['plate'] = df['animal'].apply(lambda x: x.split('-')[0])
            df['location'] = df['animal'].apply(lambda x: x.split('-')[1])

            # concat batch, plate, location to animal_id
            df['animal_id'] = df['batch'].astype(str) + '-' + df['plate'] + '-' + df['location']

            # light intensity data
            df_intensities = []
            for stim_time in [1800, 3600, 5400, 7200]:
                file_intensity = os.path.join(f'Processed_data/light_intensity/{fish_type}/{POWER}W-batch{batch}/'
                                              f'{POWER}W-60h-{day}dpf-01/'
                                              f'{(stim_time - 30) * 1000}_{(stim_time + 30) * 1000}.csv')
                # get light intensity data
                df_intensity = pd.read_csv(file_intensity)
                df_intensities.append(df_intensity)
            df_intensities = pd.concat(df_intensities, axis=0).reset_index(drop=True)

            # merge light intensity and behavior data, then append
            df = pd.merge(df, df_intensities, on=['animal_id', 'end'], how='left')
            df_batches.append(df)

        # concat all batches
        df_batches = pd.concat(df_batches, axis=0).reset_index(drop=True)

        # select columns
        df_batches = df_batches[['batch', 'plate', 'location', 'end', 'activity_sum', 'label', 'animal_id',
                                 'light_intensity']]

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

            # check if light intensity column has null values
            if df_stimulus['light_intensity'].isnull().values.any():
                raise ValueError('light intensity column has null values')

            # categorical columns
            df_stimulus['label'] = df_stimulus['label'].apply(
                lambda x: compare_group[0] if x == 0 else compare_group[1])
            df_stimulus['label'] = df_stimulus['label'].astype('category')

            # categorize label column ['batch', 'plate', 'location', 'label']
            for col in ['batch', 'plate', 'animal_id']:
                df_stimulus[col] = df_stimulus[col].astype('category')
                # df_stimulus[col] = df_stimulus[col].cat.codes

            # ==== step: remove all three effects using linear regression method ===============================
            df_rm = rm_all_baseline(df_stimulus, stimulus_time)

            # ======== Hotelling’s T-squared test for before and after stimulus =================================
            # before stimulus
            # comparison between control and sar
            before_res = hotelling_t2(df_stimulus, stimulus_time, is_before=True, compare_group=compare_group)

            # after stimulus
            after_res = hotelling_t2(df_stimulus, stimulus_time, is_before=False, compare_group=compare_group)
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
            ax.set_ylim(-0.1, 0.26)

            # ===================================================================================================

        # set title for whole figure
        fig.suptitle('Day {} - {} - {}'.format(day, ACTIVITY_TYPE, compare_group[1]))

        plt.tight_layout()
        plt.savefig(f'Figures/Stats/Quantization/{fish_type}/burst/acute_response_intensity/'
                    f'{ACTIVITY_TYPE}_{POWER}w_60h_day{day}_post_duration{post_duration}_burst4.png', dpi=300)
        # =============================================================================
