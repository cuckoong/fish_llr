import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ieee')
os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')

if __name__ == '__main__':
    mode = 'quantization'
    fish_type = 'Tg'
    power = 1
    batch_idx = 'all'
    df_all = pd.read_csv(f'Analysis_Results/ML_results/{fish_type}/Quan_Data_Classification/'
                         f'feature_selection_intensity/'
                         f'all_normalized_{batch_idx}.csv')
    df = df_all[df_all['power'] == power].copy()
    # df['SAR'] = df['power'].map({0: 0, 5: 0.009, 1.2: 2, 3: 4.9})
    df['acc'] = df['acc'] * 100
    df['classifier_idx'] = df['case'].map({'NB': 2, 'SVM': 0, 'KNN': 1})

    plt.figure(figsize=(1.5, 5))
    idx = 0
    for day in [5, 6, 7, 8]:
        df_sel = df[(df['day'] == day)]
        plt.subplot(4, 1, idx + 1)
        # bar plot in center
        sns.barplot(x='classifier_idx', y='acc', data=df_sel)
        for i, row in df_sel.iterrows():
            if (row['p-value'] < 0.05) and (row['acc'] > 60):
                plt.text(row['classifier_idx'] - 0.1, row['acc'] + 0.5, '*', fontsize=10)
            if (row['p-value'] < 0.01) and (row['acc'] > 60):
                plt.text(row['classifier_idx'] - 0.1, row['acc'] + 6, '*', fontsize=10)
            if (row['p-value'] < 0.001) and (row['acc'] > 60):
                plt.text(row['classifier_idx'] - 0.1, row['acc'] + 12, '*', fontsize=10)

        plt.ylim(0, 100)
        plt.axhline(y=50, color='grey', linestyle='--')
        # remove top and right border
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.ylabel('{}dpf'.format(day))
        plt.xticks([])
        plt.xlabel('')
        idx += 1

    plt.tight_layout()
    # add legend for each classifier for whole figure
    # plt.legend(['SVM', 'KNN', 'NB'], loc='upper center', bbox_to_anchor=(0.5, 1.1))
    # plt.show()
    plt.savefig('Figures/ML_acc/feature_selection_intensity'
                '/ML_{}_{}_{}W_acc_batch{}.png'.format(fish_type, mode, power, batch_idx), dpi=300)
