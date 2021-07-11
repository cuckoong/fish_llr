import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    power = 0
    batch = 'all'
    methods = ['SVM', 'NB', 'KNN']
    df_svm = pd.read_csv('/home/tmp2/PycharmProjects/fish_llr/'
                         'Analysis_Results/ML_results/ML_acc/SVM_{}w_{}.csv'.format(power, batch))

    df_nb = pd.read_csv('/home/tmp2/PycharmProjects/fish_llr/'
                        'Analysis_Results/ML_results/ML_acc/NB_{}w_{}.csv'.format(power, batch))

    df_knn = pd.read_csv('/home/tmp2/PycharmProjects/fish_llr/'
                         'Analysis_Results/ML_results/ML_acc/KNN_{}w_{}.csv'.format(power, batch))
    df_svm['method'] = 'SVM'
    df_nb['method'] = 'NB'
    df_knn['method'] = 'KNN'

    df_all = pd.concat([df_svm, df_nb, df_knn])
    df = df_all[df_all['power']==power].sort_values(['power', 'method', 'day', 'time'])

    labels = ['1min', '2min', '30min', 'Baseline']
    x = np.arange(len(labels))
    # the label locations
    width = 0.35  # the width of the bars
    days = [5, 6, 7, 8]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))  # , sharey=True, sharex=True)
    axs = axs.flatten()

    for i, day in enumerate(days):
        svm_acc = df[(df['method'] == 'SVM') & (df['day'] == day)]['acc']
        nb_acc = df[(df['method'] == 'NB') & (df['day'] == day)]['acc']
        knn_acc = df[(df['method'] == 'KNN') & (df['day'] == day)]['acc']
        # min_30 = df[(df['time']=='30') & (df['day']==day)]['acc']

        svm_pvalue = df[(df['method'] == 'SVM') & (df['day'] == day)]['pvalue']
        nb_pvalue = df[(df['method'] == 'NB') & (df['day'] == day)]['pvalue']
        knn_pvalue = df[(df['method'] == 'KNN') & (df['day'] == day)]['pvalue']
        # min_30_pvalue = df[(df['time'] == '30') & (df['day'] == day)]['pvalue']

        # rects1 = axs[i].bar(x - 2 * width / 3, baseline, width / 3, label='baseline')
        rects2 = axs[i].bar(x - 1 * width / 3, svm_acc, width / 3, label='SVM')
        rects3 = axs[i].bar(x + 0 * width / 3, nb_acc, width / 3, label='NB')
        rects4 = axs[i].bar(x + 1 * width / 3, knn_acc, width / 3, label='KNN')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axs[i].set_ylabel('Balanced Accuracy')
        # axs[i].set_title('Accuracy for fish on {}dpf (SVM)'.format(day))
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels)
        axs[i].set_ylim((0, 1))
        axs[i].legend(loc='upper left')
        axs[i].annotate(xy=(1, 0.8), text='Data from {}dpf'.format(day))
        # axs[i].axhline(y=0.5, ls='--', c='k')

        s = 12
        p_t = 0.05
        # axs[i].scatter(x - 2 * width / 3, (baseline_pvalue < p_t) * (baseline + 0.02), marker='*', c='k', s=s)
        axs[i].scatter(x - 1 * width / 3, (svm_pvalue < p_t) * (svm_acc + 0.02), marker='*', c='k', s=s)
        axs[i].scatter(x + 0 * width / 3, (nb_pvalue < p_t) * (nb_acc + 0.02), marker='*', c='k', s=s)
        axs[i].scatter(x + 1 * width / 3, (knn_pvalue < p_t) * (knn_acc + 0.02), marker='*', c='k', s=s)

        p_t = 0.01
        # axs[i].scatter(x - 2 * width / 3, (baseline_pvalue < p_t) * (baseline + 0.04), marker='*', c='k', s=s)
        axs[i].scatter(x - 1 * width / 3, (svm_pvalue < p_t) * (svm_acc + 0.04), marker='*', c='k', s=s)
        axs[i].scatter(x + 0 * width / 3, (nb_pvalue < p_t) * (nb_acc + 0.04), marker='*', c='k', s=s)
        axs[i].scatter(x + 1 * width / 3, (knn_pvalue < p_t) * (knn_acc + 0.04), marker='*', c='k', s=s)

        p_t = 0.001
        # axs[i].scatter(x - 2 * width / 3, (baseline_pvalue < p_t) * (baseline + 0.06), marker='*', c='k', s=s)
        axs[i].scatter(x - 1 * width / 3, (svm_pvalue < p_t) * (svm_acc + 0.06), marker='*', c='k', s=s)
        axs[i].scatter(x + 0 * width / 3, (nb_pvalue < p_t) * (nb_acc + 0.06), marker='*', c='k', s=s)
        axs[i].scatter(x + 1 * width / 3, (knn_pvalue < p_t) * (knn_acc + 0.06), marker='*', c='k', s=s)

    # if method == 'NB':
    #     fig.suptitle('Classification Using {}'.format('Naive\nBayes'), fontsize=16)
    # else:
    #     fig.suptitle('Classification Using {}'.format(method), fontsize=16)

    fig.tight_layout()
    plt.savefig('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/ML_results/ML_acc'+
                '{}W_batch{}.png'.format(power, batch), dpi=600)
