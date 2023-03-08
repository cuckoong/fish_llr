import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ieee')
os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr')

if __name__ == '__main__':
    mode = 'quantization'
    fish_type = 'Tg'
    batch = 2
    power = 1
    df = pd.read_csv('Analysis_Results/ML_results/Tg/Quan_Data_Classification/feature_selection/{}W/acc-batch{}.csv'.
                     format(power, batch))
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
        plt.ylabel('{}dpf'.format(day))
        plt.xticks([])
        plt.xlabel('')
        idx += 1

    plt.tight_layout()
    # add legend for each classifier for whole figure
    # plt.legend(['SVM', 'KNN', 'NB'], loc='upper center', bbox_to_anchor=(0.5, 1.1))
    # plt.show()
    plt.savefig('Figures/ML_acc/feature_selection/ML_transgenic_{}_{}W_acc_batch{}.png'.format(mode, power, batch),
                dpi=300)
