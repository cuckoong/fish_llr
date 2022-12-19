import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, permutation_test_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd

import os

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr/')


def clf(data, labels, case='SVM'):
    if case == 'NB':
        steps = [('variance_threshold', VarianceThreshold()), ('scaler', StandardScaler()),
                 ('selection', SelectPercentile(percentile=20)), ('NB', GaussianNB())]
        pip = Pipeline(steps)
        # permutation test
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
        accuracy, _, p = permutation_test_score(pip, data, labels, cv=cv, n_permutations=500, n_jobs=-1,
                                                random_state=1, verbose=0, scoring='balanced_accuracy')
        return accuracy, p

    if case == 'SVM':
        accuracy, p = svm_clf(data, labels)
        return accuracy, p

    elif case == 'KNN':
        accuracy, p = knn_clf(data, labels)
        return accuracy, p


def grid_search(x, y, pip, param_grid):
    gd_sr = GridSearchCV(pip, param_grid=param_grid, scoring='balanced_accuracy', cv=5, n_jobs=-1)
    gd_sr.fit(x, y)
    params = gd_sr.best_params_
    return params


def svm_clf(x, y):
    # grid search pipeline
    steps = [('variance_threshold', VarianceThreshold()), ('scaler', StandardScaler()),
             ('selection', SelectPercentile(percentile=20)), ('SVM', SVC())]
    pip = Pipeline(steps)
    param_grid = [{'SVM__C': np.logspace(-3, 3, 7),
                   'SVM__gamma': np.logspace(-3, 0, 4)}]

    params = grid_search(x, y, pip, param_grid)

    # best model
    best_C = params['SVM__C']
    best_gamma = params['SVM__gamma']
    # best model
    model = SVC(C=best_C, gamma=best_gamma)

    # permutation test
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    accuracy, _, p = permutation_test_score(model, x, y, cv=cv, n_permutations=500, n_jobs=-1,
                                            random_state=1, verbose=0, scoring='balanced_accuracy')
    return accuracy, p


def knn_clf(x, y):
    # grid search pipeline
    steps = [('variance_threshold', VarianceThreshold()), ('scaler', StandardScaler()),
             ('selection', SelectPercentile(percentile=20)), ('knn', KNeighborsClassifier())]
    pip = Pipeline(steps)
    param_grid = [{'knn__n_neighbors': np.arange(1, 10)}]
    params = grid_search(x, y, pip, param_grid)

    # best model
    n_neighbors = params['knn__n_neighbors']
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # permutation test
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    accuracy, _, p = permutation_test_score(model, x, y, cv=cv, n_permutations=500, n_jobs=-1,
                                            random_state=1, verbose=0, scoring='balanced_accuracy')
    return accuracy, p


if __name__ == "__main__":
    cases = ['NB', 'SVM', 'KNN']
    fish_type = 'Tg'
    power_levels = [1.2]  # [0, 1.2, 3]
    time_in_min = 30  # using all 30 minutes data
    days = [5, 6, 7, 8]
    batches = [1]  # [1, 2]
    plates = [1]  # [1, 2]
    hour = 60

    case_list = []
    power_list = []
    day_list = []
    acc_list = []
    p_value_list = []
    time_list = []

    for case_name in cases:  # different classifiers
        for power_level in power_levels:  # different power levels
            for day in days:
                # different batch and plate from
                # the same day,
                # the same level of radiation
                # are grouped together for classification.
                feature_list = []
                label_list = []
                for batch_idx in batches:
                    data_dir = 'Processed_data/quantization/{}/batch{}/features'.format(fish_type, batch_idx)
                    for plate in plates:
                        df = pd.read_csv(os.path.join(data_dir,
                                                      '{}W-60h-{}dpf-0{}-{}-min.csv'.format(power_level, day,
                                                                                            plate, time_in_min)))
                        # get labels
                        label = df['label']
                        label_list.append(label)

                        # get features
                        features = df.drop(['label'], axis=1)
                        feature_list.append(features)

                feature = np.concatenate(feature_list, axis=0)
                label = np.concatenate(label_list, axis=0)

                acc, p_value = clf(feature, label, case=case_name)

                # gathering results
                case_list.append(case_name)
                power_list.append(power_level)
                day_list.append(day)
                acc_list.append(acc)
                p_value_list.append(p_value)

    acc_df = pd.DataFrame(data=dict(case=case_list, power=power_list, day=day_list, acc=acc_list, pvalue=p_value_list))
    acc_df.to_csv('Analysis_Results/ML_results/{}/Quan_Data_Classification/'
                  'acc_with_feature_selection.csv'.format(fish_type), index=False)
