import os
import matplotlib.pyplot as plt
import seaborn as sns

from utils import group_batch_data, rm_animal_baseline

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')

if __name__ == '__main__':
    ACTIVITY_TYPE = 'burdur'
    fish_type = 'Tg'
    POWERS = [1, 1.2]
    batches = [1, 2]
    days = [5, 6, 7, 8]
    ON_OFF_DURATION = 1800
    PRE_DURATION = 600
    BATCHES = [1, 2, 'all']

    for POWER in POWERS:
        power_sar = '1.6W/kg' if POWER == 1 else '2W/kg'
        for day in days:
            # ===== load data and group by animal_id =============================================
            df_batches = group_batch_data(fish_type, ACTIVITY_TYPE, POWER, batches, day)
            # stimulus timestamp
            on_off = [3600, 7200]
            off_on = [1800, 5400]
            # ===== normalizing data, and remove baseline ===========================================
            # step-one: remove baseline by pre-stimulus period
            df = rm_animal_baseline(df_batches, (off_on[0] - PRE_DURATION, off_on[0]))

            for batch in BATCHES:
                if batch != 'all':
                    df_batches = df[df['batch'] == batch].copy()
                else:
                    df_batches = df.copy()

                # ==== per minutes activity ==============================================================
                df_batches['end_min'] = df_batches['end'] // 60

                df_batches_min = df_batches.groupby(['label', 'animal_id', 'end_min']). \
                    agg({'activity_sum': 'sum'}).reset_index()
                df_batches_min['label_name'] = df['label'].apply(lambda x: 'EM' if x == 1 else 'Ctrl')

                # visualize median and iqr
                plt.figure()
                sns.lineplot(x='end_min', y='activity_sum', hue='label_name', data=df_batches_min)
                plt.title(f'SAR={power_sar}W {day} day')
                plt.legend(title=None, loc='upper left')
                plt.xlabel('Time (min)')
                plt.ylabel('Busrt seconds (with Baseline Correction)')

                # add vertical line in stimulus period
                plt.axvline(x=on_off[0] // 60 - 1, color='black', linestyle='--')
                plt.axvline(x=on_off[1] // 60 - 1, color='black', linestyle='--')
                plt.axvline(x=off_on[0] // 60 - 1, color='blue', linestyle='--')
                plt.axvline(x=off_on[1] // 60 - 1, color='blue', linestyle='--')

                # add text in stimulus period
                plt.text(x=on_off[0] // 60 - 20, y=5, s='ON', color='black')
                plt.text(x=on_off[1] // 60 - 20, y=5, s='ON', color='black')
                plt.text(x=off_on[0] // 60 - 20, y=5, s='OFF', color='blue')
                plt.text(x=off_on[1] // 60 - 20, y=5, s='OFF', color='blue')
                plt.text(x=off_on[1] // 60 + 40, y=5, s='OFF', color='blue')

                # legend
                plt.legend(loc='upper left')
                plt.savefig(f'Figures/activity_per_min/{fish_type}/{POWER}W_{day}day_{batch}batch_min.png')

            '''
            # ==== per seconds activity ==============================================================
            # baseline_to_on (pre-stimulus period 10s + stimulus period 10s)
            pre_stim = 10
            post_stim = 10
            df_batches_subset = df_batches[(df_batches['end'] >= off_on[0] - pre_stim) &
                                              (df_batches['end'] <= off_on[0] + post_stim)]
            # visualize
            plt.figure()
            sns.lineplot(x='end', y='activity_sum', hue='label', data=df_batches_subset)
            plt.title(f'{fish_type} {POWER}W {day} day')
            plt.xlabel('Time (s)')
            plt.ylabel('Busrt seconds (with Baseline Correction)')

            # add vertical line in stimulus period
            # plt.axvline(x=on_off[0] // 60 - 1, color='black', linestyle='--')
            # plt.axvline(x=on_off[1] // 60 - 1, color='black', linestyle='--')
            # plt.axvline(x=off_on[0] // 60 - 1, color='blue', linestyle='--')
            # plt.axvline(x=off_on[1] // 60 - 1, color='blue', linestyle='--')
            #
            # # add text in stimulus period
            # plt.text(x=on_off[0] // 60 - 20, y=5, s='ON', color='black')
            # plt.text(x=off_on[0] // 60 - 20, y=5, s='OFF', color='blue')

            # legend
            plt.legend(loc='upper left')
            plt.savefig(f'Figures/activity_per_min/{fish_type}/{POWER}W_{day}day_off_on.png')
            '''

