import numpy as np
import pandas as pd


def light_intensity_adjustment(data, light_intensities, reference_intensity):
    """Adjust data for differences in light intensity.

    Args:
        data (array-like): Array containing the normalized activity percentages.
        light_intensities (array-like): Array of light intensities for each zebrafish.
        reference_intensity (float): Reference light intensity for normalization.

    Returns:
        array-like: Activity percentages adjusted for differences in light intensity.
    """
    adjusted_data = data * (reference_intensity / light_intensities[:, np.newaxis])
    return adjusted_data


def measure_startle_response(data, light_onset, startle_threshold, startle_window, stable_threshold, activity_threshold,
                             on_window, min_stable_duration):
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
        the activity stabilized (default 10s).

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
                                   (animal_data['end'] < light_onset + startle_window)]

        # ===== startle latency =======
        # Find the first time point where the activity percentage exceeds the threshold
        startle_time_list = startle_data[startle_data['activity_sum'] >
                                         startle_data['activity_sum_baseline_std'] *
                                         startle_threshold]['end']

        if len(startle_time_list) > 0:
            startle_time = startle_time_list.min()
            startle_latency = startle_time - light_onset
            startle_intensity = startle_data['activity_sum'].max()
        else:
            startle_time = None
            startle_latency = None
            startle_intensity = None

        startle_latencies[animal_id] = startle_latency
        # Find maximum activity within the startle window
        startle_intensities[animal_id] = startle_intensity

        # ===== startle transition to adjustment =======
        # Find the transition interval after startle and before the activity stabilizes
        if startle_time is not None:
            # Find the time when the activity stabilizes
            post_startle_data = animal_data[(animal_data['end'] > startle_time) &
                                            (animal_data['end'] < light_onset + on_window)]

            stable_count = 0
            stabilization_time = None

            for _, row in post_startle_data.iterrows():
                if row['activity_sum'] < stable_threshold * row['activity_sum_baseline_std']:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= min_stable_duration:
                    stabilization_time = row['end'] - min_stable_duration + 1
                    break

            if stabilization_time is not None:
                adjustment_interval = stabilization_time - light_onset
            else:
                adjustment_interval = None
        else:
            adjustment_interval = None

        adjustment_intervals[animal_id] = adjustment_interval

        # ===== active bouts after adjustment =======
        if adjustment_interval is not None:
            post_adjustment_data = animal_data[(animal_data['end'] >= light_onset + adjustment_interval) &
                                               (animal_data['end'] < light_onset + on_window)]

            # count bout for rest and active
            active_bout_count = 0
            active_bout_duration_sum = 0
            in_active_bout = False

            rest_bout_count = 0
            rest_bout_duration_sum = 0
            in_rest_bout = False

            for _, row in post_adjustment_data.iterrows():
                if row['activity_sum'] > activity_threshold * row['activity_sum_baseline_std']:
                    if not in_active_bout:
                        active_bout_count += 1
                        in_active_bout = True
                    active_bout_duration_sum += 1
                    in_rest_bout = False
                else:
                    if not in_rest_bout:
                        rest_bout_count += 1
                        in_rest_bout = True
                    rest_bout_duration_sum += 1
                    in_active_bout = False

            if active_bout_count > 0:
                active_bout_intensity = active_bout_duration_sum / active_bout_count
            else:
                active_bout_intensity = 0

            if rest_bout_count > 0:
                rest_bout_intensity = rest_bout_duration_sum / rest_bout_count
            else:
                rest_bout_intensity = 0

        else:
            active_bout_intensity = None
            active_bout_count = None
            rest_bout_intensity = None
            rest_bout_count = None

        active_bout_intensities[animal_id] = active_bout_intensity
        active_bout_counts[animal_id] = active_bout_count
        rest_bout_intensities[animal_id] = rest_bout_intensity
        rest_bout_counts[animal_id] = rest_bout_count

    return startle_intensities, startle_latencies, adjustment_intervals, active_bout_intensities, active_bout_counts, \
               rest_bout_intensities, rest_bout_counts


def measure_dark_adjustment_metrics(data, light_off, activity_threshold, rest_threshold, min_dark_stable_duration,
                                    off_window):
    """
    Measure the maximum activity after light off, adjust to dark interval, and rest bout metrics
    (latency, count, and density) after adjustment for individual animals in the given DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the activity data for multiple animals, with columns 'end', 'animal_id',
         and 'activity_sum'.
    light_off : int
        The time point at which the light is turned off.
    activity_threshold : float
        The threshold for distinguishing between active and inactive bouts, as a percentage of activity.
    rest_threshold : float
        The threshold for determining that the activity has stabilized in the dark, as a percentage of activity.
    min_dark_stable_duration : int
        The minimum number of consecutive time points with activity below the rest threshold to consider
        the activity stabilized in the dark.
    off_window : int
        The time window after the light offset and before next light onset.

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
                                          (animal_data['end'] < light_off + off_window)]

        # ======== maximum activity after light off =========
        # Measure maximum activity after light off
        increase_intensities[animal_id] = post_light_off_data['activity_sum'].max()
        increase_latency_time = post_light_off_data.loc[post_light_off_data['activity_sum'].idxmax(), 'end']
        increase_latencies[animal_id] = increase_latency_time - light_off

        # ======== dark adjustment interval =========
        # Find the transition interval after maximum activity and before the next light onset
        post_increase_data = post_light_off_data[
            (post_light_off_data['end'] >= increase_latency_time) &
            (post_light_off_data['end'] < light_off + off_window)]

        # ========= rest bout ===================================
        # Find the time when the activity stabilizes in the dark
        stable_count = 0
        dark_stabilization_time = None

        for _, row in post_increase_data.iterrows():
            if row['activity_sum'] < rest_threshold * row['activity_sum_baseline_std']:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= min_dark_stable_duration:
                dark_stabilization_time = row['end'] - min_dark_stable_duration + 1
                break

        if dark_stabilization_time is not None:
            dark_adjustment_interval = dark_stabilization_time - light_off
            post_dark_stabilization_data = post_light_off_data[(post_light_off_data['end'] >= dark_stabilization_time) &
                                                               (post_light_off_data['end'] < light_off + off_window)]

            # Calculate rest bout latency, count, and density
            rest_count = 0
            rest_duration_sum = 0
            in_rest_bout = False

            active_count = 0
            active_duration_sum = 0
            in_active_bout = False

            for _, row in post_dark_stabilization_data.iterrows():
                if row['activity_sum'] < activity_threshold * row['activity_sum_baseline_std']:
                    if not in_rest_bout:
                        in_rest_bout = True
                        rest_count += 1
                    rest_duration_sum += 1
                    in_active_bout = False
                else:
                    if not in_active_bout:
                        in_active_bout = True
                        active_count += 1
                    active_duration_sum += 1
                    in_rest_bout = False

            if rest_count > 0:
                rest_density = rest_duration_sum / rest_count
            else:
                rest_count = 0
                rest_density = 0

            if active_count > 0:
                active_density = active_duration_sum / active_count
            else:
                active_count = 0
                active_density = 0

        else:
            dark_adjustment_interval = None
            rest_count = None
            rest_density = None
            active_count = None
            active_density = None

        dark_adjustment_intervals[animal_id] = dark_adjustment_interval
        rest_bout_counts[animal_id] = rest_count
        rest_bout_intensities[animal_id] = rest_density
        active_bout_intensities[animal_id] = active_density
        active_bout_counts[animal_id] = active_count

    return increase_intensities, increase_latencies, dark_adjustment_intervals, rest_bout_intensities, \
           rest_bout_counts, active_bout_intensities, active_bout_counts
