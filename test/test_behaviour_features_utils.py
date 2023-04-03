import pandas as pd
from src.behaviour_features_utils import measure_dark_adjustment_metrics, measure_startle_response


def test_measure_dark_adjustment_metrics():
    data = pd.DataFrame({
        'end': [1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808,
                1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808,
                1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808,
                1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, ],
        'animal_id': [1, 1, 1, 1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3, 3, 3, 3,
                      4, 4, 4, 4, 4, 4, 4, 4],
        'activity_sum': [0.1, 0.2, 0.3, 0.1, 0, 0, 0, 0,
                         0.05, 0.15, 0.25, 0.3, 0.3, 0, 0, 0,
                         0.07, 0.17, 0.27, 0, 0.2, 0, 0, 0,
                         0.07, 0.17, 0.27, 0.15, 0.15, 0, 0, 0],
        'activity_sum_baseline_std': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'activity_sum_rest_baseline_std': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'activity_sum_active_baseline_std': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                             0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'activity_sum_rest_baseline_mean': [0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0],
        'activity_sum_active_baseline_mean': [0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0],
    })

    light_off = 1800
    activity_threshold = 1.5
    stable_threshold = 1.2
    min_dark_stable_duration = 1
    off_window = 1800

    dark_response = measure_dark_adjustment_metrics(data,
                                                    light_off,
                                                    stable_threshold=stable_threshold,
                                                    activity_threshold=activity_threshold,
                                                    min_dark_stable_duration=min_dark_stable_duration,
                                                    off_window=off_window)

    increase_intensities, increase_latencies, dark_adjustment_intervals, rest_bout_intensities, rest_bout_counts, \
    active_bout_intensities, active_bout_counts = dark_response

    assert increase_intensities == {1: 0.30, 2: 0.3, 3: 0.27, 4: 0.27}
    assert increase_latencies == {1: 3, 2: 4, 3: 3, 4: 3}
    assert dark_adjustment_intervals == {1: 6, 2: 7, 3: 6, 4: 6}
    assert rest_bout_intensities == {1: 3, 2: 2, 3: 3, 4: 3}
    assert rest_bout_counts == {1: 1, 2: 1, 3: 1, 4: 1}
    assert active_bout_intensities == {1: 0, 2: 0, 3: 0, 4: 0}
    assert active_bout_counts == {1: 0, 2: 0, 3: 0, 4: 0}


def test_measure_startle_response():
    data = pd.DataFrame({
        'end': [1801, 1802, 1803, 1804, 1805,
                1801, 1802, 1803, 1804, 1805,
                1801, 1802, 1803, 1804, 1805,
                1801, 1802, 1803, 1804, 1805,
                1801, 1802, 1803, 1804, 1805],
        'animal_id': [1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3,
                      4, 4, 4, 4, 4,
                      5, 5, 5, 5, 5],
        'activity_sum': [0.1, 0.8, 0.15, 0.15, 0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.5, 0.4, 0.6, 0.5,
                         0.0, 0.6, 0.0, 0.0, 0.3,
                         0.0, 0.6, 0.0, 0.0, 0.1],
        'activity_sum_baseline_std': [0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1],
        'activity_sum_rest_baseline_std': [0.1, 0.1, 0.1, 0.1, 0.1,
                                           0.1, 0.1, 0.1, 0.1, 0.1,
                                           0.1, 0.1, 0.1, 0.1, 0.1,
                                           0.1, 0.1, 0.1, 0.1, 0.1,
                                           0.1, 0.1, 0.1, 0.1, 0.1],
        'activity_sum_active_baseline_std': [0.1, 0.1, 0.1, 0.1, 0.1,
                                             0.1, 0.1, 0.1, 0.1, 0.1,
                                             0.1, 0.1, 0.1, 0.1, 0.1,
                                             0.1, 0.1, 0.1, 0.1, 0.1,
                                             0.1, 0.1, 0.1, 0.1, 0.1],
        'activity_sum_rest_baseline_mean': [0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 0],
        'activity_sum_active_baseline_mean': [0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0],
    })

    light_on = 1800
    startle_threshold = 5
    startle_window = 3
    stabilization_threshold = 2
    activity_threshold = 1
    on_window = 1800
    min_stable_duration = 2

    light_response = measure_startle_response(data, light_on, startle_threshold=startle_threshold,
                                              startle_window=startle_window, stable_threshold=stabilization_threshold,
                                              activity_threshold=activity_threshold, on_window=on_window,
                                              min_stable_duration=min_stable_duration)

    startle_intensities, startle_latencies, adjustment_intervals, active_bout_intensities, active_bout_counts, \
    rest_bout_intensities, rest_bout_counts = light_response

    assert startle_intensities == {1: 0.8, 2: None, 3: None, 4: 0.6, 5: 0.6}
    assert startle_latencies == {1: 2, 2: None, 3: None, 4: 2, 5: 2}

    assert adjustment_intervals == {1: 3, 2: None, 3: None, 4: 3, 5: 3}
    assert active_bout_intensities == {1: 3, 2: None, 3: None, 4: 3, 5: 3}
    assert active_bout_counts == {1: 1, 2: None, 3: None, 4: 1, 5: 1}
    assert rest_bout_intensities == {1: 0, 2: None, 3: None, 4: 0, 5: 0}
    assert rest_bout_counts == {1: 0, 2: None, 3: None, 4: 0, 5: 0}

    # You can add more test cases with different input data and expected output to cover more scenarios.
