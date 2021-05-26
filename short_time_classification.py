import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    methods = ['SVM4w', 'KNN4w', 'NB4w']
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(7, 2.5))
    days = ['5dpf', '6dpf', '7dpf']  # , 8]
    x = np.arange(len(days))
    # the label locations
    width = 0.35  # the width of the bars
    # , sharey=True, sharex=True)
    axs = axs.flatten()
    for i, method in enumerate(methods):
        df = pd.read_csv('/Users/panpan/PycharmProjects/FIsh/Results/{}.csv'.format(method))
        min_1 = df[df['time'] == 1]['acc']
        min_2 = df[df['time'] == 2]['acc']
        min_30 = df[df['time'] == 30]['acc']

        min_1_pvalue = df[df['time'] == 1]['pvalue']
        min_2_pvalue = df[df['time'] == 2]['pvalue']
        min_30_pvalue = df[df['time'] == 30]['pvalue']

        # rects1 = axs[i].bar(x - 2 * width / 3, baseline, width / 3, label='baseline')
        rects2 = axs[i].bar(x - 1 * width / 3, min_1, width / 3, label='1min')
        rects3 = axs[i].bar(x + 0 * width / 3, min_2, width / 3, label='2min')
        rects4 = axs[i].bar(x + 1 * width / 3, min_30, width / 3, label='30min')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        if i == 0:
            axs[i].set_ylabel('Balanced Accuracy')
        # axs[i].set_title('Accuracy for fish on {}dpf (SVM)'.format(day))
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(days)
        axs[i].set_ylim((0, 1))
        axs[i].legend(loc='upper left')
        axs[i].axhline(y=0.5, ls='--', c='k')
        if method == 'NB4w':
            axs[i].annotate(xy=(1.5, 0.8), text='Naive\nBayes')
        else:
            axs[i].annotate(xy=(1.5, 0.8), text=method[:-2])

        s = 10
        p_t = 0.05
        # axs[i].scatter(x - 2 * width / 3, (baseline_pvalue < p_t) * (baseline + 0.02), marker='*', c='k', s=s)
        axs[i].scatter(x - 1 * width / 3, (min_1_pvalue < p_t) * (min_1 + 0.02), marker='*', c='k', s=s)
        axs[i].scatter(x + 0 * width / 3, (min_2_pvalue < p_t) * (min_2 + 0.02), marker='*', c='k', s=s)
        axs[i].scatter(x + 1 * width / 3, (min_30_pvalue < p_t) * (min_30 + 0.02), marker='*', c='k', s=s)

        p_t = 0.01
        # axs[i].scatter(x - 2 * width / 3, (baseline_pvalue < p_t) * (baseline + 0.04), marker='*', c='k', s=s)
        axs[i].scatter(x - 1 * width / 3, (min_1_pvalue < p_t) * (min_1 + 0.05), marker='*', c='k', s=s)
        axs[i].scatter(x + 0 * width / 3, (min_2_pvalue < p_t) * (min_2 + 0.05), marker='*', c='k', s=s)
        axs[i].scatter(x + 1 * width / 3, (min_30_pvalue < p_t) * (min_30 + 0.05), marker='*', c='k', s=s)

        p_t = 0.001
        # axs[i].scatter(x - 2 * width / 3, (baseline_pvalue < p_t) * (baseline + 0.06), marker='*', c='k', s=s)
        axs[i].scatter(x - 1 * width / 3, (min_1_pvalue < p_t) * (min_1 + 0.08), marker='*', c='k', s=s)
        axs[i].scatter(x + 0 * width / 3, (min_2_pvalue < p_t) * (min_2 + 0.08), marker='*', c='k', s=s)
        axs[i].scatter(x + 1 * width / 3, (min_30_pvalue < p_t) * (min_30 + 0.08), marker='*', c='k', s=s)

    # if method == 'NB4w':
    #     fig.suptitle('Classification Using {}'.format('Naive Bayes'), fontsize=16)
    # else:
    #     fig.suptitle('Classification Using {}'.format(method), fontsize=16)

    fig.tight_layout()
    plt.savefig('4w_classification.png', dpi=600)