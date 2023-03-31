import pandas as pd
from src.behaviour_features_utils import measure_dark_adjustment_metrics, measure_startle_response


# def test_measure_dark_adjustment_metrics():
#     data = pd.DataFrame({
#         'end': [1801, 1802, 1803, 1804, 1805, 1801, 1802, 1803, 1804, 1805, 1801, 1802, 1803, 1804, 1805],
#         'animal_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
#         'activity_sum': [0.1, 0.2, 0.3, 0.1, 0, 0.05, 0.15, 0.25, 0.3, 0.3, 0.07, 0.17, 0.27, 0, 0.2]
#     })
#
#     light_off = 1800
#     activity_threshold = 0.15
#     rest_threshold = 0.12
#     min_dark_stable_duration = 1
#     off_window = 1800
#
#     increase_intensities, increase_latencies, dark_adjustment_intervals, bout_intensities, bout_counts = \
#         measure_dark_adjustment_metrics(data, light_off, activity_threshold, rest_threshold, min_dark_stable_duration,
#                                         off_window)
#
#     assert increase_intensities == {1: 0.30, 2: 0.3, 3: 0.27}
#     assert increase_latencies == {1: 3, 2: 4, 3: 3}
#     assert dark_adjustment_intervals == {1: 4, 2: None, 3: 4}
#     assert bout_intensities == {1: 0.1, 2: None, 3: 0}
#     assert bout_counts == {1: 1, 2: None, 3: 1}


def test_measure_startle_response():
    data = pd.DataFrame({
        'end': [1801, 1802, 1803, 1804, 1805,
                1801, 1802, 1803, 1804, 1805,
                1801, 1802, 1803, 1804, 1805,
                1801, 1802, 1803, 1804, 1805],
        'animal_id': [1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3,
                      4, 4, 4, 4, 4],
        'activity_sum': [0.1, 0.8, 0.1, 0.1, 0,
                         0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.5, 0.4, 0.6, 0.5,
                         0.0, 0.6, 0.0, 0.0, 0.3]
    })

    light_on = 1800
    startle_threshold = 0.5
    startle_window = 3
    stabilization_threshold = 0.2
    activity_threshold = -1
    on_window = 1800
    min_stable_duration = 2

    startle_intensities, startle_latencies, adjustment_intervals, bout_intensities, bout_counts = \
        measure_startle_response(data, light_on, startle_threshold, startle_window, stabilization_threshold,
                                 activity_threshold, on_window=on_window, min_stable_duration=min_stable_duration)

    assert startle_intensities == {1: 0.8, 2: None, 3: None, 4: 0.6}
    assert startle_latencies == {1: 2, 2: None, 3: None, 4: 2}
    assert adjustment_intervals == {1: 3, 2: None, 3: None, 4: 3}
    assert bout_intensities == {1: 0.2, 2: None, 3: None, 4: 0.3}
    assert bout_counts == {1: 1, 2: None, 3: None, 4: 1}

    # You can add more test cases with different input data and expected output to cover more scenarios.
