import os
import numpy as np
import pandas as pd
from utils import num_rest_active_bout, group_batch_data, get_peri_stimulus_data, rm_all_baseline

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')


def get_features(df):
    df_postStim = df.loc[df.loc[:, 'end'] > stimulus_time, ['animal_id', 'activity_sum', 'end']]
    # get mean over time (1min)
    df_postStim['end_min'] = df_postStim['end'].apply(lambda x: int((x - 1) // 60 + 1))
    df_postStim = df_postStim.groupby(['animal_id', 'end_min'])['activity_sum'].mean().reset_index()

    # to a 2d array, with animal_id as row, activity_sum as column
    df_postStim = df_postStim.pivot(index='animal_id', columns='end_min',
                                    values='activity_sum').reset_index()
    df_postStim.sort_values(by=['animal_id'], inplace=True)
    # add the 3rd dimension to the array
    df_postStim.drop('animal_id', axis=1, inplace=True)
    df_postStim = df_postStim.values.reshape(df_postStim.shape[0], df_postStim.shape[1], 1)

    features_postStim = num_rest_active_bout(df_postStim)
    features_array = np.stack(features_postStim, axis=1)
    return features_array


if __name__ == '__main__':
    ACTIVITY_TYPE = 'burdur'
    POWER = 1.2
    BATCH_LIST = [1, 2]
    days = [5, 6, 7, 8]
    pre_duration = 30
    post_duration = 30  # 30s
    stimuli_time = [1800, 3600, 5400, 7200]
    fish_type = 'Tg'

    output_path = f'Processed_data/quantization/{fish_type}/ML_features_intensity/'

    for day in days:
        # save results
        features_period_list = []
        column_list = []
        label_list = []

        df_batches = group_batch_data(fish_type, ACTIVITY_TYPE, POWER, BATCH_LIST, day)

        # =============================================================================
        # mean baseline activity for each animal
        # acute response -durations - durations post stimulus (at 1800s, 5400s),
        # =============================================================================
        for stimulus_time in stimuli_time:
            df_stimulus = get_peri_stimulus_data(df_batches, stimulus_time, pre_duration, post_duration)

            # ==== step: remove all three effects using linear regression method ===============================
            df_rm = rm_all_baseline(df_stimulus, stimulus_time)

            '''
            # ==== step: remove light intensity effects using linear regression method =========================
            print('linear regression between activity_sum_scaled and location_light')
            df_stimulus_rmLight = rm_light_baseline(df_stimulus)
            # ===================================================================================================

            # ==== step: remove batch effects using linear regression method =====================================
            print('linear regression between activity_sum_scaled and batch')
            df_stimulus_rmBatch = rm_batch_baseline(df_stimulus_rmLight)
            # ===================================================================================================

            # ==== step: remove baseline activity ==============================================================
            print('remove baseline activity')
            df_stimulus_rmBaseline = rm_animal_baseline(df_stimulus_rmBatch)
            # ===================================================================================================
            '''

            # =============== calculate the features ==================================================
            features_period_array = get_features(df_rm)
            features_period_list.append(features_period_array)
            column_list.extend(
                item.format(stimulus_time) for item in
                ['total_active_{}', 'waking_{}', 'total_rest_{}', 'max_active_{}',
                 'num_rest_bout_{}', 'rest_bout_avg_{}', 'rest_latency_{}',
                 'num_active_bout_{}', 'active_bout_avg_{}', 'active_latency_{}'])

            # ===================================================================================================

        # ======== concatenate all features ==============================================================
        features = np.concatenate(features_period_list, axis=1)
        # create dataframe for features
        features_df = pd.DataFrame(features)
        features_df.columns = column_list

        df_label = df_stimulus[['animal_id', 'label']].drop_duplicates().reset_index(drop=True). \
            sort_values(by='animal_id').reset_index(drop=True)
        features_df['label'] = df_label['label'].values
        features_df['batch'] = df_label['animal_id'].map(lambda x: x.split('-')[0]).values
        features_df.to_csv(os.path.join(output_path, f'{POWER}W_day{day}_data.csv'), index=False)
        # =============================================================================
