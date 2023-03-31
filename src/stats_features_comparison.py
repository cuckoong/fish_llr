import os
import seaborn as sns
from utils import group_batch_data
from utils import rm_animal_baseline, rm_batch_baseline, rm_linear_trend
from behaviour_features_utils import *

sns.set_style('whitegrid')
sns.set_context('paper')
sns.set_palette('Set2')

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')


def get_compare_group(power_level):
    if power_level == 0:  # 0W/Kg vertical, wt
        groups = ['Control', '0W/Kg SAR']
    elif power_level == 1:  # 1.6W/Kg vertical, tg
        groups = ['control', '1.6W/Kg SAR']
    elif power_level == 1.2:  # 2W/Kg vertical, wt+tg
        groups = ['Control', '2W/Kg SAR']
    elif power_level == 3:  # 4.9W/Kg vertical, wt
        groups = ['Control', '4.9W/Kg SAR']
    elif power_level == 5:  # horizontal, wt
        groups = ['Control', '0.009W/Kg SAR']
    return groups


if __name__ == '__main__':
    ACTIVITY_TYPE = 'burdur'
    fish_type = 'Tg'
    POWER = 1
    batches = [1, 2]
    days = [5, 6, 7, 8]
    on_off_duration = 1800
    pre_duration = 600
    post_duration = 1770
    colors = ['orange', 'blue']
    compare_group = get_compare_group(POWER)

    for day in days:
        df_batches = group_batch_data(fish_type, ACTIVITY_TYPE, POWER, batches, day)

        # stimulus timestamp
        on_off = [3601, 7201]
        # metrics: startle response (latency and maximum); active bouts (latency, duration, and intensity)
        # Transition from light OFF to ON: When the light is suddenly turned on after a period of darkness,
        # zebrafish often display a startle response, characterized by a brief burst of intense locomotor activity.
        # This response is thought to be a defensive reaction to sudden changes in the environment.
        # After the initial startle response, the zebrafish gradually adjust to the light conditions and
        # return to their typical active phase behavior.

        off_on = [1801, 5401]
        # metrics: Brief increase of activity after light off (latency and maximum);
        # metrics (2): rest bouts (latency, duration, and intensity)
        #  When the light is turned off after a period of illumination, zebrafish may initially show a brief increase in
        #  activity, possibly as a response to the sudden change in their environment. However, they soon adjust to the
        #  dark conditions and enter the rest phase, exhibiting reduced locomotor activity.

        # ===== normalizing data, and remove baseline ===========================================
        # step-one: remove baseline by pre-stimulus period
        df_batches = rm_animal_baseline(df_batches, (off_on[0] - pre_duration, off_on[0]))

        # ===== visualize data for individual fish =============================================
        '''
        import seaborn as sns
        sns.set_style('whitegrid')
        sns.set_context('paper')
        sns.lineplot(data=df_batches, x='end', y='activity_sum_scaled', hue='label', palette='Set2')
        '''

        # step-two: remove baseline by batch mean
        # df_batches = rm_batch_baseline(df_batches)

        # step-three: remove baseline by light intensity

        # ===== startle response after light ON ================================================
        # metrics: startle response (latency and maximum), and adjust to light interval
        startle_intensities, startle_latencies, adjustment_intervals, bout_intensities, bout_counts \
            = measure_startle_response(df_batches, on_off[0], startle_threshold=3, startle_window=5,
                                       stable_threshold=2, activity_threshold=0.5, on_window=on_off_duration,
                                       min_stable_duration=3)

        # ===== startle response after light OFF ===============================================
        # metrics: brief increase of activity after light off (maximum), and adjust to dark interval
        # metrics (2): rest bouts (latency, duration, and intensity)
        increase_intensities, increase_latencies, rest_intervals, rest_bout_intensities, rest_bout_counts \
            = measure_dark_adjustment_metrics(df_batches, off_on[0], activity_threshold=0.5, rest_threshold=0.5,
                                              min_dark_stable_duration=3, off_window=on_off_duration)


        # ===== plot startle response after light ON ===========================================
