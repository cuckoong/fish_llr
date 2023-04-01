import os
import seaborn as sns
from utils import group_batch_data
from utils import rm_animal_baseline
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
    POWERS = [1, 1.2]
    batches = [1, 2]
    days = [5, 6, 7, 8]
    ON_OFF_DURATION = 1800
    PRE_DURATION = 600
    POST_DURATION = 1770

    for POWER in POWERS:
        '''
        for day in days:
            # ===== load data and group by animal_id =============================================
            df_batches = group_batch_data(fish_type, ACTIVITY_TYPE, POWER, batches, day)

            # stimulus timestamp
            on_off = [3600, 7200]
            # metrics: startle response (latency and maximum); active bouts (latency, duration, and intensity)
            # Transition from light OFF to ON: When the light is suddenly turned on after a period of darkness,
            # zebrafish often display a startle response, characterized by a brief burst of intense locomotor activity.
            # This response is thought to be a defensive reaction to sudden changes in the environment.
            # After the initial startle response, the zebrafish gradually adjust to the light conditions and
            # return to their typical active phase behavior.

            off_on = [1800, 5400]
            # metrics: Brief increase of activity after light off (latency and maximum);
            # metrics (2): rest bouts (latency, duration, and intensity) When the light is turned off after a period
            # of illumination, zebrafish may initially show a brief increase in activity, possibly as a response to
            # the sudden change in their environment. However, they soon adjust to the dark conditions and enter the
            # rest phase, exhibiting reduced locomotor activity.

            # ===== normalizing data, and remove baseline ===========================================
            # step-one: remove baseline by pre-stimulus period
            df_batches = rm_animal_baseline(df_batches, (off_on[0] - PRE_DURATION, off_on[0]))

            # ===== visualize data for individual fish =============================================
            # step-two: remove baseline by batch mean
            # df_batches = rm_batch_baseline(df_batches)

            # step-three: remove baseline by light intensity

            # ===== startle response after light ON ================================================
            # metrics: startle response (latency and maximum), and adjust to light interval
            startle_intensities, startle_latencies, adjustment_intervals, active_bout_intensities, active_bout_counts \
                = measure_startle_response(df_batches, on_off[0], startle_threshold=3, startle_window=3,
                                           stable_threshold=3, activity_threshold=0, on_window=ON_OFF_DURATION,
                                           min_stable_duration=2)

            # ===== startle response after light OFF ===============================================
            # metrics: brief increase of activity after light off (maximum), and adjust to dark interval
            # metrics (2): rest bouts (latency, duration, and intensity)
            increase_intensities, increase_latencies, rest_intervals, rest_bout_intensities, rest_bout_counts \
                = measure_dark_adjustment_metrics(df_batches, off_on[0], activity_threshold=3, rest_threshold=3,
                                                  min_dark_stable_duration=2, off_window=ON_OFF_DURATION)

            # ===== Get features from feature dicts =============================================
            features = pd.DataFrame([startle_intensities, startle_latencies, adjustment_intervals,
                                     active_bout_intensities, active_bout_counts,
                                     increase_intensities, increase_latencies, rest_intervals,
                                     rest_bout_intensities, rest_bout_counts]).T

            features.reset_index(inplace=True)
            features.rename(columns={'index': 'animal_id'}, inplace=True)

            features.columns = ['animal_id',
                                'startle_intensity', 'startle_latency', 'adjustment_interval',
                                'active_bout_intensity', 'active_bout_count',
                                'increase_intensity', 'increase_latency', 'rest_interval',
                                'rest_bout_intensity', 'rest_bout_count']

            # sync label and batch from df_batches by animal_id
            df_label_batch = df_batches[['animal_id', 'label', 'batch']].drop_duplicates()
            features = pd.merge(features, df_label_batch, on='animal_id', how='left')

            # ===== save features to csv file =============================================
            features.to_csv(f'Processed_data/behaviour_pattern/{fish_type}/{POWER}W_day{day}_features.csv',
                            index=False)
        '''
        # ============== group all features together into one dataframe ===========================
        features_all = pd.DataFrame()
        for day in days:
            filename = f'Processed_data/behaviour_pattern/{fish_type}/{POWER}W_day{day}_features.csv'
            df = pd.read_csv(filename)
            df['day'] = day
            features_all = pd.concat([features_all, df], axis=0)

        # save features to csv file
        features_all.to_csv(f'Processed_data/behaviour_pattern/Tg/{POWER}W_features.csv', index=False)
