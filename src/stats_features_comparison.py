import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

sns.set(style="whitegrid")


def auto_select_test(contingency_table):
    chi2, p_value_chi2, _, _ = chi2_contingency(contingency_table)
    row_totals = np.sum(contingency_table, axis=1)
    col_totals = np.sum(contingency_table, axis=0)
    grand_total = np.sum(contingency_table)

    expected = np.outer(row_totals, col_totals) / grand_total
    use_fisher = np.any(expected < 5)

    if use_fisher:
        odds_ratio, p_value_fisher = fisher_exact(contingency_table)
        return "Fisher's exact test", odds_ratio, p_value_fisher
    else:
        return "Chi-squared test", None, p_value_chi2


def significance_markers(value):
    if value < 0.001:
        return '***'
    elif value < 0.01:
        return '**'
    elif value < 0.05:
        return '*'
    else:
        return ''


def compare_event(df, selected_column):
    df['event'] = df[selected_column].apply(lambda x: 1 if pd.notna(x) else 0)

    # visualize bar chart with label side by side, day as x-axis, and startle percentage as y-axis
    event_summary = df.groupby(['label', 'day'])['event'].agg(['mean', 'sum', 'count']).reset_index()

    # Calculate p-values and add stars to the plot
    days = event_summary['day'].unique()
    group1_data = event_summary[event_summary['label'] == 'Control']
    group2_data = event_summary[event_summary['label'] == 'EM']

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    sns.barplot(x="day", y="mean", hue="label", data=event_summary, ax=ax)

    for day in days:
        group1_count = group1_data[group1_data['day'] == day]['sum'].iloc[0]
        group2_count = group2_data[group2_data['day'] == day]['sum'].iloc[0]
        group1_total = group1_data[group1_data['day'] == day]['count'].iloc[0]
        group2_total = group2_data[group2_data['day'] == day]['count'].iloc[0]

        contingency_table = [[group1_count, group1_total - group1_count],
                             [group2_count, group2_total - group2_count]]

        info, _, p_value = auto_select_test(contingency_table)
        print(f'{selected_column} day {day}: {info} p-value = {p_value:.3f}')

        if p_value < 0.05:
            star_marker = significance_markers(p_value)
            max_ratio = max(group1_count / group1_total, group2_count / group2_total)
            ax.text(day - 5, max_ratio + 0.02, star_marker, fontsize=20, horizontalalignment='center')

    plt.title(selected_column.replace('_', ' ').title() + ' Comparison')
    plt.legend(loc='upper left')
    plt.ylabel('Event Percentage')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'Figures/Stats/Quantization/{fish_type}/behaviour_pattern/'
                f'{power}W_{selected_column}_comparison_event.png', dpi=300)


def compare_value(df, selected_column):
    days = df['day'].unique()
    group1_data = df[df['label'] == 'Control'].copy()
    group2_data = df[df['label'] == 'EM'].copy()

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.boxplot(x="day", y=selected_column, hue="label", data=df, ax=ax)

    for day in days:
        # compare values between labels
        group1_value = group1_data[group1_data['day'] == day][selected_column].dropna()
        group2_value = group2_data[group2_data['day'] == day][selected_column].dropna()

        if len(group1_value) == 0 or len(group2_value) == 0:
            continue
        else:
            _, p_value = mannwhitneyu(group1_value, group2_value)
            print(f'{selected_column} day {day}: p-value = {p_value:.3f}')

        if p_value < 0.05:
            star_marker = significance_markers(p_value)
            max_ratio = max(group1_value.mean(), group2_value.mean())
            ax.text(day - 5, max_ratio + 0.02, star_marker, fontsize=20, horizontalalignment='center')

    plt.title(selected_column.replace('_', ' ').title() + ' Comparison')
    plt.legend(loc='upper left')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'Figures/Stats/Quantization/{fish_type}/behaviour_pattern/'
                f'{power}W_{selected_column}_comparison_value.png', dpi=300)


if __name__ == '__main__':
    os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')
    fish_type = 'Tg'
    powers = [1, 1.2]
    for power in powers:
        features = pd.read_csv(f'Processed_data/behaviour_pattern/{fish_type}/{power}W_features.csv')
        features['label'] = features['label'].apply(lambda x: 'Control' if x == 0 else 'EM')

        # ====== comparing none vs not none between labels ==================================
        compare_event(features, selected_column='startle_latency')
        compare_event(features, selected_column='adjustment_interval')
        compare_event(features, selected_column='active_bout_intensity')
        compare_event(features, selected_column='rest_interval')
        compare_event(features, selected_column='rest_bout_intensity')

        # ====== comparing values between labels ==================================
        compare_value(features, selected_column='adjustment_interval')
        compare_value(features, selected_column='rest_bout_intensity')
        compare_value(features, selected_column='rest_bout_count')
        compare_value(features, selected_column='active_bout_intensity')
        compare_value(features, selected_column='active_bout_count')
        compare_value(features, selected_column='startle_latency')
