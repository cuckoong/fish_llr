import os
import pandas as pd
import seaborn as sns
from utils import group_batch_data
from utils import rm_animal_baseline
from behaviour_features_utils import add_baseline
from behaviour_features_utils import measure_startle_response
from behaviour_features_utils import measure_dark_adjustment_metrics

sns.set_style('whitegrid')
sns.set_context('paper')
sns.set_palette('Set2')

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')


if __name__ == '__main__':
    ACTIVITY_TYPE = 'burdur'
    fish_type = 'Tg'
    POWERS = [1, 1.2]
    batches = [1, 2]
    days = [5, 6, 7, 8]
    ON_OFF_DURATION = 1800
    PRE_DURATION = 600

    for POWER in POWERS:
        # '''
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
            df_active_mean, df_active_std = add_baseline(df_batches, (on_off[0] - PRE_DURATION - 1, on_off[0] - 1))
            df_batches = pd.merge(df_batches, df_active_mean, on='animal_id', how='left',
                                  suffixes=('', '_active_baseline_mean'))
            df_batches = pd.merge(df_batches, df_active_std, on='animal_id', how='left',
                                  suffixes=('', '_active_baseline_std'))

            light_response = measure_startle_response(df_batches, off_on[0], startle_threshold=3, startle_window=3,
                                                      stable_threshold=3, activity_threshold=1,
                                                      on_window=ON_OFF_DURATION, min_stable_duration=3,
                                                      min_bout_duration=3)

            startle_intensities, startle_latencies, light_adjustment_intervals, light_active_bout_intensities, \
                light_active_bout_counts, light_rest_bout_intensities, light_rest_bout_counts = light_response

            # ===== dark response after light OFF ===============================================
            # metrics: brief increase of activity after light off (maximum), and adjust to dark interval
            # metrics (2): rest bouts (latency, duration, and intensity)
            df_rest_mean, df_rest_std = add_baseline(df_batches, (off_on[1] - PRE_DURATION - 1, off_on[1] - 1))
            df_batches = pd.merge(df_batches, df_rest_mean, on='animal_id', how='left',
                                  suffixes=('', '_rest_baseline_mean'))
            df_batches = pd.merge(df_batches, df_rest_std, on='animal_id', how='left',
                                  suffixes=('', '_rest_baseline_std'))

            dark_response = measure_dark_adjustment_metrics(df_batches, on_off[0], stable_threshold=3,
                                                            activity_threshold=1, min_dark_stable_duration=3,
                                                            off_window=ON_OFF_DURATION, min_bout_duration=3)

            increase_intensities, increase_latencies, dark_adjustment_intervals, dark_rest_bout_intensities, \
                dark_rest_bout_counts, dark_active_bout_intensities, dark_active_bout_counts = dark_response

            # ===== Get features from feature dicts =============================================
            features = pd.DataFrame([startle_intensities, startle_latencies, light_adjustment_intervals,
                                     light_active_bout_intensities, light_active_bout_counts,
                                     light_rest_bout_intensities, light_rest_bout_counts,
                                     increase_intensities, increase_latencies, dark_adjustment_intervals,
                                     dark_rest_bout_intensities, dark_rest_bout_counts,
                                     dark_active_bout_intensities, dark_active_bout_counts]).T

            features.reset_index(inplace=True)
            features.rename(columns={'index': 'animal_id'}, inplace=True)

            features.columns = ['animal_id',
                                'startle_intensity', 'startle_latency', 'light_adjustment_intervals',
                                'light_active_bout_intensity', 'light_active_bout_count',
                                'light_rest_bout_intensity', 'light_rest_bout_count',
                                'increase_intensity', 'increase_latency', 'dark_adjustment_intervals',
                                'dark_rest_bout_intensity', 'dark_rest_bout_count',
                                'dark_active_bout_intensity', 'dark_active_bout_count']

            # sync label and batch from df_batches by animal_id
            df_label_batch = df_batches[['animal_id', 'label', 'batch']].drop_duplicates()
            features = pd.merge(features, df_label_batch, on='animal_id', how='left')

            # ===== save features to csv file =============================================
            features.to_csv(f'Processed_data/behaviour_pattern/{fish_type}/{POWER}W_day{day}_features.csv',
                            index=False)
        # '''
        # ============== group all features together into one dataframe ===========================
        features_all = pd.DataFrame()
        for day in days:
            filename = f'Processed_data/behaviour_pattern/{fish_type}/{POWER}W_day{day}_features.csv'
            df = pd.read_csv(filename)
            df['day'] = day
            features_all = pd.concat([features_all, df], axis=0)

        # save features to csv file
        features_all.to_csv(f'Processed_data/behaviour_pattern/Tg/{POWER}W_features.csv', index=False)
