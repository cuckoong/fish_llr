import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('/Users/panpan/PycharmProjects/FIsh/nb.csv')
    labels = ['0', '1', '2.5']
    day = 5
    baseline = df[(df['time']=='baseline') & (df['day']==day)]['acc']
    min_1 = df[(df['time']=='1') & (df['day']==day)]['acc']
    min_2 = df[(df['time']=='2') & (df['day']==day)]['acc']
    min_30 = df[(df['time']=='30') & (df['day']==day)]['acc']

    baseline_pvalue = df[(df['time'] == 'baseline') & (df['day'] == day)]['pvalue']
    min_1_pvalue = df[(df['time'] == '1') & (df['day'] == day)]['pvalue']
    min_2_pvalue = df[(df['time'] == '2') & (df['day'] == day)]['pvalue']
    min_30_pvalue = df[(df['time'] == '30') & (df['day'] == day)]['pvalue']

    x = np.arange(len(labels))

    # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 2 * width / 3, baseline, width/3, label='baseline')
    rects2 = ax.bar(x - 1 * width / 3, min_1, width/3, label='1min')
    rects3 = ax.bar(x + 0 * width / 3, min_2, width/3, label='2min')
    rects4 = ax.bar(x + 1 * width / 3, min_30, width/3, label='30min')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Accuracy for fish on {}dpf (NB)'.format(day))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    s = 12
    p_t = 0.05
    ax.scatter(x - 2 * width / 3, (baseline_pvalue <p_t) * (baseline+0.02), marker='*', c='k', s=s)
    ax.scatter(x - 1 * width / 3, (min_1_pvalue <p_t) * (min_1+0.02), marker='*', c='k', s=s)
    ax.scatter(x + 0 * width / 3, (min_2_pvalue <p_t) * (min_2+0.02), marker='*', c='k', s=s)
    ax.scatter(x + 1 * width / 3, (min_30_pvalue <p_t) * (min_30+0.02), marker='*', c='k', s=s)

    p_t = 0.01
    ax.scatter(x - 2 * width / 3, (baseline_pvalue < p_t) * (baseline + 0.04), marker='*', c='k', s=s)
    ax.scatter(x - 1 * width / 3, (min_1_pvalue < p_t) * (min_1 + 0.04), marker='*', c='k', s=s)
    ax.scatter(x + 0 * width / 3, (min_2_pvalue < p_t) * (min_2 + 0.04), marker='*', c='k', s=s)
    ax.scatter(x + 1 * width / 3, (min_30_pvalue < p_t) * (min_30 + 0.04), marker='*', c='k', s=s)

    p_t = 0.001
    ax.scatter(x - 2 * width / 3, (baseline_pvalue < p_t) * (baseline + 0.06), marker='*', c='k', s=s)
    ax.scatter(x - 1 * width / 3, (min_1_pvalue < p_t) * (min_1 + 0.06), marker='*', c='k', s=s)
    ax.scatter(x + 0 * width / 3, (min_2_pvalue < p_t) * (min_2 + 0.06), marker='*', c='k', s=s)
    ax.scatter(x + 1 * width / 3, (min_30_pvalue < p_t) * (min_30 + 0.06), marker='*', c='k', s=s)

    fig.tight_layout()
    plt.ylim((0, 0.9))
    plt.axhline(y=0.5, ls = '--', c = 'k')
    plt.show()