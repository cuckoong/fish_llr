import os

import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr/')

if __name__ == '__main__':
    power = 1
    days = [5, 6, 7, 8]
    batches = [1, 2]
    fig, axs = plt.subplots(4, 2, figsize=(4, 8))
    axs = axs.flatten()
    i = 0

    for day in days:
        for batch in batches:
            filename = f'Processed_data/quantization/Tg/batch{batch}/features/' \
                       f'{power}W-60h-{day}dpf-01-30-min.csv'

            df = pd.read_csv(filename)

            df_feature = df.drop(['label'], axis=1)
            pipe = Pipeline(
                [('variance_threshold', VarianceThreshold()), ('scaler', StandardScaler()),
                 ('selection', SelectPercentile(percentile=20)), ('pca', PCA(n_components=2))]
            )

            df_pc = pipe.fit_transform(df_feature, df['label'])

            principalDf = pd.DataFrame(data=df_pc,  columns=['PC1', 'PC2'])
            finalDf = pd.concat([principalDf, df[['label']]], axis=1)

            sns.scatterplot(x='PC1', y='PC2', hue='label', data=finalDf, ax=axs[i])
            axs[i].set_title(f'Day {day} - Batch {batch}')
            # remove legend
            axs[i].get_legend().remove()
            i += 1

    print("??")