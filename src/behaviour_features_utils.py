import numpy as np
import pandas as pd
from scipy.stats._mstats_basic import winsorize


def add_baseline(df, time_interval):
    df = df.copy()
    df['add_baseline'] = df['end'].apply(lambda x: 0 if (x < time_interval[1]) &
                                                        (x >= time_interval[0]) else 1).values

    # winsorize transform and then do z-score for individual animal
    df_baseline_mean = df[df['add_baseline'] == 0].groupby('animal_id')['activity_sum']. \
        apply(lambda x: winsorize(x, limits=(0.05, 0.05)).mean()).reset_index()
    df_baseline_std = df[df['add_baseline'] == 0].groupby('animal_id')['activity_sum']. \
        apply(lambda x: winsorize(x, limits=(0.05, 0.05)).std()).reset_index()

    return df_baseline_mean, df_baseline_std


def moving_average(df, window_size):
    # data sorted by end
    data = df.copy()
    data.sort_values(by='end', inplace=True, ascending=True)
    data['activity_sum_smooth'] = data['activity_sum'].rolling(window=window_size, min_periods=1).mean()
    return data


def detect_startle(activity_data, startle_threshold, light_onset):
    startle_time_list = activity_data[activity_data['activity_sum'] >
                                      activity_data['activity_sum_baseline_std'] * startle_threshold]['end']

    if len(startle_time_list) > 0:
        startle_time = startle_time_list.min()
        startle_latency = startle_time - light_onset
        startle_intensity = activity_data['activity_sum'].max()
    else:
        startle_time = None
        startle_latency = None
        startle_intensity = None

    return startle_time, startle_latency, startle_intensity


def detect_stable(activity_data, stable_threshold, min_stable_duration, onset_time, resting):
    stable_count = 0
    stabilization_time = None

    if resting:
        baseline_mean_name = 'activity_sum_rest_baseline_mean'
        baseline_std_name = 'activity_sum_rest_baseline_std'
    else:
        baseline_mean_name = 'activity_sum_active_baseline_mean'
        baseline_std_name = 'activity_sum_active_baseline_std'

    for _, row in activity_data.iterrows():
        if abs(row['activity_sum_smooth'] - row[baseline_mean_name]) <= \
                (stable_threshold * row[baseline_std_name]):
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= min_stable_duration:
            stabilization_time = row['end'] - min_stable_duration + 1
            break

    if stabilization_time is not None:
        adjustment_interval = stabilization_time - onset_time
    else:
        adjustment_interval = None

    return adjustment_interval


def identify_bouts(activity_data, activity_threshold_lower=-np.inf, activity_threshold_upper=np.inf, min_duration=2):
    activity_data = activity_data.copy()
    activity_data['within_threshold'] = (activity_data['activity_sum'] >= activity_threshold_lower) & \
                                        (activity_data['activity_sum'] <= activity_threshold_upper)
    activity_data['bout_id'] = (activity_data['within_threshold'] != activity_data['within_threshold'].shift()).cumsum()

    bout_summary = activity_data.groupby('bout_id').agg(
        duration=('within_threshold', 'count'),
        is_active=('within_threshold', 'first'),
    )

    bouts = bout_summary[(bout_summary['duration'] >= min_duration) &
                         (bout_summary['is_active'])].reset_index(drop=True)
    bout_count = len(bouts)
    if bout_count > 0:
        bout_mean_duration = bouts['duration'].mean()
    else:
        bout_mean_duration = 0
    return bout_mean_duration, bout_count


def measure_startle_response(data, light_onset, startle_threshold, startle_window, stable_threshold,
                             activity_threshold, on_window, min_stable_duration, smooth_win_size=3,
                             min_bout_duration=2):
    """
    Detect the startle response and measure the latency for individual animals in the given DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the activity data for multiple animals, with columns 'end', 'animal_id',
        and 'activity_percentage'.
    light_onset : int
        The time point at which the light is turned on.
    startle_threshold : float between 0 and 1
        The threshold for detecting a startle response, as a percentage of activity.
    startle_window : int
        The time window after the light onset during which the startle response should be detected.
    stable_threshold : float between 0 and 1
        The threshold for determining that the activity has stabilized, as a percentage of activity.
    activity_threshold : float between 0 and 1
        The threshold for determining that the animal is active, as a percentage of activity. default 0.1
    on_window : int
        The time window after the light onset and before next light offset (default 30 minutes)
    min_stable_duration : int
        The minimum number of consecutive time points with activity below the stabilization threshold to consider
        the activity stabilized.
    smooth_win_size: int
        The size of the moving average window for smoothing the activity data (default 3s).
    min_bout_duration:int
        The minimum duration of a bout to be considered an active bout (default 2s).

    Returns
    -------
    startle_latencies : dict
        A dictionary containing the startle latency for each animal (keyed by animal_id).
        If no startle response is detected, the value is set to None.

    """
    startle_latencies = {}
    startle_intensities = {}
    adjustment_intervals = {}
    active_bout_intensities = {}
    active_bout_counts = {}
    rest_bout_intensities = {}
    rest_bout_counts = {}

    for animal_id, animal_data in data.groupby('animal_id'):
        # Select activity data within the startle window
        startle_data = animal_data[(animal_data['end'] >= light_onset) &
                                   (animal_data['end'] < light_onset + startle_window)].copy()

        # ===== startle latency =======================
        # Find the first time point where the activity percentage exceeds the threshold
        startle_time, startle_latency, startle_intensity = detect_startle(startle_data, startle_threshold, light_onset)
        startle_latencies[animal_id] = startle_latency
        startle_intensities[animal_id] = startle_intensity

        # ===== startle transition to adjustment =======
        # Find the transition interval after startle and before the activity stabilizes
        if startle_time is None:
            adjustment_interval = None
        else:
            # smooth activity data after startle
            # Find the time when the activity stabilizes
            post_startle_data = animal_data[(animal_data['end'] > startle_time) &
                                            (animal_data['end'] < light_onset + on_window)].copy()

            # smooth activity data with moving average over 3s window
            post_startle_data = moving_average(post_startle_data, smooth_win_size)
            adjustment_interval = detect_stable(post_startle_data, stable_threshold, min_stable_duration, light_onset,
                                                resting=False)
        adjustment_intervals[animal_id] = adjustment_interval

        # ===== active bouts after adjustment =======
        post_light_data = animal_data[(animal_data['end'] >= light_onset) &
                                      (animal_data['end'] < light_onset + on_window)].copy()

        # smooth activity data with moving average over 3s window
        post_light_data = moving_average(post_light_data, smooth_win_size)

        activity_std = post_light_data['activity_sum_active_baseline_std'].unique()[0]
        activity_mean = post_light_data['activity_sum_active_baseline_mean'].unique()[0]

        active_bout_intensity, active_bout_count = identify_bouts(
            post_light_data,
            activity_threshold_lower=activity_mean + activity_threshold * activity_std,
            min_duration=min_bout_duration)
        rest_bout_intensity, rest_bout_count = identify_bouts(
            post_light_data,
            activity_threshold_upper=activity_mean - activity_threshold * activity_std,
            min_duration=min_bout_duration)

        active_bout_intensities[animal_id] = active_bout_intensity
        active_bout_counts[animal_id] = active_bout_count
        rest_bout_intensities[animal_id] = rest_bout_intensity
        rest_bout_counts[animal_id] = rest_bout_count

    return startle_intensities, startle_latencies, adjustment_intervals, active_bout_intensities, active_bout_counts, \
           rest_bout_intensities, rest_bout_counts


def measure_dark_adjustment_metrics(data, light_off, stable_threshold, activity_threshold,
                                    min_dark_stable_duration, off_window, smooth_win_size=3, min_bout_duration=2):
    """
    Measure the maximum activity after light off, adjust to dark interval, and rest bout metrics
    (latency, count, and density) after adjustment for individual animals in the given DataFrame.

    Parameters
  batches, off_on[0], stable_threshold=3,  ----------
    data : pd.DataFrame
        A DataFrame containing the activity data for multiple animals, with columns 'end', 'animal_id',
         and 'activity_sum'.
    light_off : int
        The time point at which the light is turned off.
    activity_threshold : float
        The threshold for distinguishing between active and inactive bouts, as a percentage of activity.
    stable_threshold : float
        The threshold for determining that the activity has stabilized in the dark, as a percentage of activity.
    min_dark_stable_duration : int
        The minimum number of consecutive time points with activity below the rest threshold to consider
        the activity stabilized in the dark.
    off_window : int
        The time window after the light offset and before next light onset.
    smooth_win_size : int
        The size of the moving average window for smoothing the activity data.
    min_bout_duration: int
        The minimum duration of a bout to be considered an active or rest bout. default:2s

    Returns
    -------
    dark_adjustment_metrics : dict
        A dictionary containing the maximum activity, dark adjustment interval, rest latency, rest count, and
        rest density for each animal (keyed by animal_id). The value is a tuple (max_activity, dark_adjustment_interval,
         rest_latency, rest_count, rest_density). If the activity does not stabilize within the data, the value is
         set to None.
    """

    increase_intensities = {}
    increase_latencies = {}
    dark_adjustment_intervals = {}
    rest_bout_intensities = {}
    rest_bout_counts = {}
    active_bout_intensities = {}
    active_bout_counts = {}

    for animal_id, animal_data in data.groupby('animal_id'):
        post_light_off_data = animal_data[(animal_data['end'] >= light_off) &
                                          (animal_data['end'] < light_off + off_window)].copy()

        # ======== maximum activity after light off =========
        # Measure maximum activity after light off
        increase_intensities[animal_id] = post_light_off_data['activity_sum'].max()
        increase_latency_time = post_light_off_data.loc[post_light_off_data['activity_sum'].idxmax(), 'end']
        increase_latencies[animal_id] = increase_latency_time - light_off

        # ======== dark adjustment interval =========
        # Find the transition interval after maximum activity and before the next light onset
        post_increase_data = post_light_off_data[
            (post_light_off_data['end'] > increase_latency_time) &
            (post_light_off_data['end'] < light_off + off_window)].copy()
        post_increase_data = moving_average(post_increase_data, smooth_win_size)
        dark_adjustment_interval = detect_stable(post_increase_data, stable_threshold, min_dark_stable_duration,
                                                 light_off, resting=True)
        dark_adjustment_intervals[animal_id] = dark_adjustment_interval

        # ========= rest bout ===================================
        # Find the time when the activity stabilizes in the dark
        # smooth the activity data
        post_dark_data = post_light_off_data[(post_light_off_data['end'] >= light_off) &
                                             (post_light_off_data['end'] < light_off + off_window)].copy()

        # smooth the activity data
        post_dark_data = moving_average(post_dark_data, smooth_win_size)
        activity_std = post_dark_data['activity_sum_rest_baseline_std'].unique()[0]
        activity_mean = post_dark_data['activity_sum_rest_baseline_mean'].unique()[0]

        rest_density, rest_count = identify_bouts(
            post_dark_data,
            activity_threshold_upper=activity_mean - activity_threshold * activity_std,
            min_duration=min_bout_duration)
        active_density, active_count = identify_bouts(
            post_dark_data,
            activity_threshold_lower=activity_mean + activity_threshold * activity_std,
            min_duration=min_bout_duration)

        rest_bout_counts[animal_id] = rest_count
        rest_bout_intensities[animal_id] = rest_density
        active_bout_intensities[animal_id] = active_density
        active_bout_counts[animal_id] = active_count

    return increase_intensities, increase_latencies, dark_adjustment_intervals, rest_bout_intensities, \
        rest_bout_counts, active_bout_intensities, active_bout_counts
