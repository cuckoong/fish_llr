import numpy as np
import pandas as pd


def baseline_correction(data, pre_stimulus_period):
    """Apply baseline correction to the data.

    Args:
        data (array-like): Array containing the raw activity percentages.
        pre_stimulus_period (int): Number of time points before stimulus onset.

    Returns:
        array-like: Baseline-corrected activity percentages.
    """
    baseline = np.mean(data[:pre_stimulus_period])
    corrected_data = data - baseline
    return corrected_data


def batch_normalization(data, batch_indices, pre_stimulus_period):
    """Normalize data within each batch.

    Args:
        data (array-like): Array containing the baseline-corrected activity percentages.
        batch_indices (list): List of index ranges defining each batch.
        pre_stimulus_period (int): Number of time points before stimulus onset.

    Returns:
        array-like: Normalized activity percentages within each batch.
    """
    normalized_data = []
    for batch_idx in batch_indices:
        batch_data = data[batch_idx]
        batch_mean = np.mean(batch_data[:, :pre_stimulus_period], axis=1, keepdims=True)
        batch_std = np.std(batch_data[:, :pre_stimulus_period], axis=1, keepdims=True)
        normalized_batch_data = (batch_data - batch_mean) / batch_std
        normalized_data.append(normalized_batch_data)
    return np.vstack(normalized_data)


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


def measure_startle_response(data, light_onset, startle_threshold, startle_window,
                             stabilization_threshold, activity_threshold,
                             on_window=1800, min_stable_duration=10):
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
    stabilization_threshold : float between 0 and 1
        The threshold for determining that the activity has stabilized, as a percentage of activity.
    activity_threshold : float between 0 and 1
        The threshold for determining that the animal is active, as a percentage of activity.
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
    bout_intensities = {}
    bout_counts = {}

    for animal_id, animal_data in data.groupby('animal_id'):
        # Select activity data within the startle window
        startle_data = animal_data[(animal_data['end'] >= light_onset) &
                                   (animal_data['end'] < light_onset + startle_window)]

        # ===== startle latency =======
        # Find the first time point where the activity percentage exceeds the threshold
        startle_time_list = startle_data[startle_data['activity_sum'] > startle_threshold]['end']

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
                if row['activity_sum'] < stabilization_threshold:
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

        # ===== active bout after adjustment =======
        if adjustment_interval is not None:
            post_adjustment_data = animal_data[(animal_data['end'] >= light_onset + adjustment_interval) &
                                               (animal_data['end'] < light_onset + on_window)]

            bout_count = 0
            bout_intensity_sum = 0
            in_bout = False

            for _, row in post_adjustment_data.iterrows():
                if row['activity_sum'] > activity_threshold:
                    if not in_bout:
                        bout_count += 1
                        in_bout = True
                    bout_intensity_sum += row['activity_sum']
                else:
                    in_bout = False

            if bout_count > 0:
                bout_intensity = bout_intensity_sum / bout_count
            else:
                bout_intensity = 0

        else:
            bout_intensity = None
            bout_count = None

        bout_intensities[animal_id] = bout_intensity
        bout_counts[animal_id] = bout_count

    return startle_intensities, startle_latencies, adjustment_intervals, bout_intensities, bout_counts


def measure_dark_adjustment_metrics(data: pd.DataFrame, light_off: int, activity_threshold: float,
                                    rest_threshold: float, min_dark_stable_duration: int,
                                    off_window: int = 1800):
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
        The time window after the light offset and before next light onset (default 30 minutes)

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
    bout_intensities = {}
    bout_counts = {}

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
            if row['activity_sum'] < rest_threshold:
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
            rest_density_sum = 0
            in_rest_bout = False

            for _, row in post_dark_stabilization_data.iterrows():
                activity = row['activity_sum']

                if activity < activity_threshold:
                    if not in_rest_bout:
                        in_rest_bout = True
                        rest_count += 1
                    rest_density_sum += activity
                else:
                    in_rest_bout = False

            if rest_count > 0:
                rest_density = rest_density_sum / rest_count
            else:
                rest_count = 0
                rest_density = 0

        else:
            dark_adjustment_interval = None
            rest_count = None
            rest_density = None

        dark_adjustment_intervals[animal_id] = dark_adjustment_interval
        bout_counts[animal_id] = rest_count
        bout_intensities[animal_id] = rest_density

    return increase_intensities, increase_latencies, dark_adjustment_intervals, bout_intensities, bout_counts


def calculate_startle_threshold(data, light_onset, startle_window, std_multiplier=2):
    """Calculate the startle response threshold.

    Args:
        data (array-like): Array containing the preprocessed activity percentages.
        light_onset (int): Time point of light onset.
        startle_window (int): Time window to search for startle response after light onset.
        std_multiplier (float, optional): Multiplier for the standard deviation, default is 2.

    Returns:
        float: Threshold for detecting startle response.
    """
    startle_data = data[:, light_onset:light_onset + startle_window]
