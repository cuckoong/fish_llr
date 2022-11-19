import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ieee')

wkdir = '/Users/panpan/PycharmProjects/old_project/fish_llr'
if __name__ == '__main__':
    # mode = 'tracking'
    mode = 'quantization'
    # metric = 'all'
    metric = 'subset'
    # df = pd.read_csv(os.path.join(wkdir, 'Processed_data/ML_acc/ML_{}_{}_acc.csv'.format(mode, metric)))
    df = pd.read_csv(os.path.join(wkdir, 'Analysis_Results/ML_results/ML_acc/{}/'
                                         'transgenic_1.2W_batch3_acc.csv'.format(mode)))
    df['SAR'] = df['power'].map({0: 0, 5: 0.009, 1.2: 2, 3: 4.9})
    df['acc'] = df['acc'] * 100
    df['classifier_idx'] = df['classifier'].map({'NB': 2, 'SVM': 0, 'KNN': 1})

    plt.figure(figsize=(3.5, 5))
    idx = 0
    for day in [5, 6, 7, 8]:
        for sar in [0, 0.009, 2, 4.9]:
            df_sel = df[(df['day'] == day) & (df['SAR'] == sar)]
            plt.subplot(4, 4, idx + 1)
            # bar plot in center
            sns.barplot(x='classifier_idx', y='acc', data=df_sel)
            # plt.bar(df_sel['classifier_idx'], df_sel['acc'], width=0.5, color='#1f77b4', align='edge')
            # sns.barplot(x='classifier', y='acc', data=df_sel)
            # add stars to the top of the bars if pvalue less than 0.05
            for i, row in df_sel.iterrows():
                if (row['pvalue'] < 0.05) and (row['acc'] > 60):
                    plt.text(row['classifier_idx'] - 0.1, row['acc'] + 0.5, '*', fontsize=10)
                if (row['pvalue'] < 0.01) and (row['acc'] > 60):
                    plt.text(row['classifier_idx'] - 0.1, row['acc'] + 6, '*', fontsize=10)
                if (row['pvalue'] < 0.001) and (row['acc'] > 60):
                    plt.text(row['classifier_idx'] - 0.1, row['acc'] + 12, '*', fontsize=10)

            plt.ylim(0, 100)
            plt.axhline(y=50, color='grey', linestyle='--')
            # remove top and right border
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            if idx not in [0, 4, 8, 12]:
                # remove y ticks on the right
                plt.yticks([])
                plt.ylabel('')
            else:
                plt.ylabel('{}dpf'.format(day))
            plt.xticks([])
            if idx > 11:
                plt.xlabel('SAR={}'.format(sar))
            else:
                plt.xlabel('')
            idx += 1

    plt.tight_layout()
    # add legend for each classifier for whole figure
    # plt.legend(['SVM', 'KNN', 'NB'], loc='upper center', bbox_to_anchor=(0.5, 1.1))
    # plt.show()
    plt.savefig(os.path.join(wkdir, 'Figures/ML_acc/ML_{}_{}_acc.png'.format(mode, metric)))
