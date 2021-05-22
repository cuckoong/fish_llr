import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    method = 'KNN'
    df = pd.read_csv('/Users/panpan/PycharmProjects/FIsh/Results/{}.csv'.format(method))
    labels = ['0W', '1W', '2.5W']
    x = np.arange(len(labels))
    # the label locations
    width = 0.35  # the width of the bars
    days = [5,6,7,8]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))#, sharey=True, sharex=True)
    axs = axs.flatten()

    for i, day in enumerate(days):
        baseline = df[(df['time']=='baseline') & (df['day']==day)]['acc']
        min_1 = df[(df['time']=='1') & (df['day']==day)]['acc']
        min_2 = df[(df['time']=='2') & (df['day']==day)]['acc']
        min_30 = df[(df['time']=='30') & (df['day']==day)]['acc']

        baseline_pvalue = df[(df['time'] == 'baseline') & (df['day'] == day)]['pvalue']
        min_1_pvalue = df[(df['time'] == '1') & (df['day'] == day)]['pvalue']
        min_2_pvalue = df[(df['time'] == '2') & (df['day'] == day)]['pvalue']
        min_30_pvalue = df[(df['time'] == '30') & (df['day'] == day)]['pvalue']

        rects1 = axs[i].bar(x - 2 * width / 3, baseline, width/3, label='baseline')
        rects2 = axs[i].bar(x - 1 * width / 3, min_1, width/3, label='1min')
        rects3 = axs[i].bar(x + 0 * width / 3, min_2, width/3, label='2min')
        rects4 = axs[i].bar(x + 1 * width / 3, min_30, width/3, label='30min')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[i].set_ylabel('Balanced Accuracy')
        # axs[i].set_title('Accuracy for fish on {}dpf (SVM)'.format(day))
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels)
        axs[i].set_ylim((0, 1))
        axs[i].legend(loc='upper left')
        axs[i].annotate(xy = (1,0.8), text = 'Data from {}dpf'.format(day))
        axs[i].axhline(y=0.5, ls='--', c='k')


        s = 12
        p_t = 0.05
        axs[i].scatter(x - 2 * width / 3, (baseline_pvalue <p_t) * (baseline+0.02), marker='*', c='k', s=s)
        axs[i].scatter(x - 1 * width / 3, (min_1_pvalue <p_t) * (min_1+0.02), marker='*', c='k', s=s)
        axs[i].scatter(x + 0 * width / 3, (min_2_pvalue <p_t) * (min_2+0.02), marker='*', c='k', s=s)
        axs[i].scatter(x + 1 * width / 3, (min_30_pvalue <p_t) * (min_30+0.02), marker='*', c='k', s=s)

        p_t = 0.01
        axs[i].scatter(x - 2 * width / 3, (baseline_pvalue < p_t) * (baseline + 0.04), marker='*', c='k', s=s)
        axs[i].scatter(x - 1 * width / 3, (min_1_pvalue < p_t) * (min_1 + 0.04), marker='*', c='k', s=s)
        axs[i].scatter(x + 0 * width / 3, (min_2_pvalue < p_t) * (min_2 + 0.04), marker='*', c='k', s=s)
        axs[i].scatter(x + 1 * width / 3, (min_30_pvalue < p_t) * (min_30 + 0.04), marker='*', c='k', s=s)

        p_t = 0.001
        axs[i].scatter(x - 2 * width / 3, (baseline_pvalue < p_t) * (baseline + 0.06), marker='*', c='k', s=s)
        axs[i].scatter(x - 1 * width / 3, (min_1_pvalue < p_t) * (min_1 + 0.06), marker='*', c='k', s=s)
        axs[i].scatter(x + 0 * width / 3, (min_2_pvalue < p_t) * (min_2 + 0.06), marker='*', c='k', s=s)
        axs[i].scatter(x + 1 * width / 3, (min_30_pvalue < p_t) * (min_30 + 0.06), marker='*', c='k', s=s)

    if method == 'NB':
        fig.suptitle('Classification Using {}'.format('Naive Bayes'), fontsize=16)
    else:
        fig.suptitle('Classification Using {}'.format(method), fontsize=16)

    fig.tight_layout()
    plt.savefig(method+'.png', dpi=600)