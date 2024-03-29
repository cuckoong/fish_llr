import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pymer4.models import Lmer
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

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


def compare_event(df, selected_column, power):
    # NAMES and days
    if selected_column == 'startle_latency':
        event_name = 'Startle Response'
    else:
        event_name = selected_column.replace('_', ' ').title()

    days = df['day'].unique()
    power_sar = '1.6W/kg' if power == 1 else '2W/kg'

    # transform to event
    df['event'] = df[selected_column].apply(lambda x: 1 if pd.notna(x) else 0)

    # visualize bar chart with label side by side, day as x-axis, and startle percentage as y-axis
    event_summary = df.groupby(['label', 'day'])['event'].agg(['mean', 'sum', 'count']).reset_index()
    group1_data = event_summary[event_summary['label'] == 'Control']
    group2_data = event_summary[event_summary['label'] == 'EM']

    # Calculate p-values and add stars to the plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    sns.barplot(x="day", y="mean", hue="label", data=event_summary, ax=ax)
    for day in days:
        group1_count = group1_data[group1_data['day'] == day]['sum'].iloc[0]
        group2_count = group2_data[group2_data['day'] == day]['sum'].iloc[0]
        group1_total = group1_data[group1_data['day'] == day]['count'].iloc[0]
        group2_total = group2_data[group2_data['day'] == day]['count'].iloc[0]
        ''''
        contingency_table = [[group1_count, group1_total - group1_count],
                             [group2_count, group2_total - group2_count]]

        info, _, p_value = auto_select_test(contingency_table)
        print(f'{selected_column} day {day}: {info} p-value = {p_value:.3f}')
        '''
        if pd.unique(df['event']).size == 1:  # all values are the not null
            print('all values are not null')
        else:
            df_subset = df[(df['day'] == day)].copy()
            model = Lmer("event  ~ label  + (1|batch)", data=df_subset, family='binomial')
            res = model.fit()
            p_value = res.loc['labelEM', 'P-val']

            if p_value < 0.05:
                star_marker = significance_markers(p_value)
                max_ratio = max(group1_count / group1_total, group2_count / group2_total)
                ax.text(day - 5, max_ratio + 0.02, star_marker, fontsize=20, horizontalalignment='center')

    plt.title(f'SAR={power_sar}')
    plt.ylabel(event_name + ' Ratio')
    plt.ylim(0, 1)
    # legend outside the plot
    plt.legend(loc='upper left', borderaxespad=0., title='')
    plt.tight_layout()
    plt.savefig(f'Figures/Stats/Quantization/{fish_type}/behaviour_pattern/'
                f'{power}W_{selected_column}_comparison_event.png', dpi=300)


def compare_value(df, selected_column, power):
    df = df.copy()

    # unique days
    days = df['day'].unique()
    group1_data = df[df['label'] == 'Control'].copy()
    group2_data = df[df['label'] == 'EM'].copy()

    # power
    power_sar = '1.6W/kg' if power == 1 else '2W/kg'

    # value name
    if selected_column == 'startle_latency':
        value_name = 'Startle Latency'
    else:
        value_name = selected_column.replace('_', ' ').title()

    #  ========== value minus control mean and divide by control std ===========================
    group1_data_mean = group1_data.groupby(['batch', 'day'])[selected_column].mean().reset_index()
    group1_data_std = group1_data.groupby(['batch', 'day'])[selected_column].std().reset_index()

    df = df.merge(group1_data_mean, on=['batch', 'day'], how='left', suffixes=('', '_mean'))
    df = df.merge(group1_data_std, on=['batch', 'day'], how='left', suffixes=('', '_std'))

    df[selected_column] = (df[selected_column] - df[selected_column + '_mean']) / (df[selected_column + '_std'] + 1e-8)
    star_y = df[selected_column].median()

    #  ========== visualize and statistical analysis ===========================
    # width 400 height 300
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    sns.boxplot(x="day", y=selected_column, hue="label", data=df, ax=ax, showfliers=False)

    for day in days:
        # compare values between labels
        group1_value = group1_data[group1_data['day'] == day][selected_column].dropna()
        group2_value = group2_data[group2_data['day'] == day][selected_column].dropna()

        if len(group1_value) == 0 or len(group2_value) == 0:
            continue
        elif group1_value.unique().size == 1 and group2_value.unique().size == 1:
            print('all values are the same')
            continue
        else:
            df_subset = df[(df['day'] == day)].dropna(subset=[selected_column]).copy()
            group1_value_nona = df_subset[df_subset['label'] == 'Control'][selected_column]
            group2_value_nona = df_subset[df_subset['label'] == 'EM'][selected_column]

            # do wilcoxon test between two groups
            _, p_value = mannwhitneyu(group1_value_nona, group2_value_nona)
            '''
            formula = f"{selected_column} ~ label + (1|batch)"

            if 'count' in selected_column:
                family = 'poisson'
            else:
                family = 'gaussian'

            model = Lmer(formula, data=df_subset, family=family)
            result = model.fit(maxiter=100000)
            p_value = result.loc['labelEM', 'P-val']
            print(f'{selected_column} day {day}: p-value = {p_value:.3f}')

            res = model.fit()
            p_value = res.loc['labelEM', 'P-val']
            '''

            if p_value < 0.05:
                star_marker = significance_markers(p_value)
                ax.text(day - 5, star_y, star_marker, fontsize=35, horizontalalignment='center', color='red')

    # legend under title outside the plot and vertical
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.xlabel('')
    ax.set_title(f'SAR={power_sar} ' + value_name)
    plt.ylabel(value_name+'\n(Normalized to Batch Control)', fontsize=8)
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
        compare_event(features, selected_column='startle_latency', power=power)
        # compare_event(features, selected_column='light_adjustment_intervals', power=power)
        # compare_event(features, selected_column='dark_adjustment_intervals', power=power)

        # ====== comparing values between labels ==================================
        compare_value(features, selected_column='light_adjustment_intervals', power=power)
        compare_value(features, selected_column='light_rest_bout_intensity', power=power)
        compare_value(features, selected_column='light_rest_bout_count', power=power)
        compare_value(features, selected_column='light_active_bout_intensity', power=power)
        compare_value(features, selected_column='light_active_bout_count', power=power)
        compare_value(features, selected_column='dark_adjustment_intervals', power=power)
        compare_value(features, selected_column='dark_rest_bout_intensity', power=power)
        compare_value(features, selected_column='dark_rest_bout_count', power=power)
        compare_value(features, selected_column='dark_active_bout_intensity', power=power)
        compare_value(features, selected_column='dark_active_bout_count', power=power)
        compare_value(features, selected_column='increase_intensity', power=power)
        compare_value(features, selected_column='startle_intensity', power=power)
        compare_value(features, selected_column='startle_latency', power=power)
