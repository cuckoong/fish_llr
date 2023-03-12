import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, chi2, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, permutation_test_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr/')


def clf(data, labels, case='SVM', cat_features=[], numeric_features=[]):
    if cat_features == [] and numeric_features == []:
        numeric_features = data.columns

    # =============== missing value imputation ===============
    missing_transformer = ColumnTransformer(
        transformers=[
            ('simple_imputer', SimpleImputer(strategy='median'), numeric_features),
            ('indicator', MissingIndicator(), numeric_features)
        ]
    )

    data_clean = missing_transformer.fit_transform(data)
    # add indicator columns for those column that have missing values
    column_names = missing_transformer.get_feature_names_out(input_features=numeric_features)
    data_clean = pd.DataFrame(data_clean, columns=column_names)

    # =============== models ===============

    pre_steps = [
        ('variance_threshold', VarianceThreshold()),
        ('scaler', StandardScaler()),
        ('selection', SelectPercentile(percentile=20))
    ]

    if case == 'NB':
        steps = pre_steps + [('NB', GaussianNB())]
        pip = Pipeline(steps)

    if case == 'SVM':
        pip = svm_clf(data_clean, labels, pre_steps)

    elif case == 'KNN':
        pip = knn_clf(data_clean, labels, pre_steps)

    #  ===============  feature importance ===============
    # x_train, x_test, y_train, y_test = train_test_split(data_clean, labels, test_size=0.2, random_state=42)
    # pip.fit(x_train, y_train)
    #
    # result = permutation_importance(pip, data_clean, labels, n_repeats=10, random_state=42)
    # feature_importances = pd.DataFrame({'feature': x_test.columns, 'importance': result.importances_mean})
    # feature_importances = feature_importances.sort_values('importance', ascending=True, ignore_index=True)
    #
    # # visualize sorted feature importance
    # plt.barh(feature_importances['feature'], feature_importances['importance'])
    # plt.show()
    #
    # sns.displot(data=x_test, x='simple_imputer__total_active_2', hue=labels, shrink=.8, height=5, aspect=2)

    #  =============== permutation test ===============
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    accuracy, _, p = permutation_test_score(pip, data_clean, labels, cv=cv, n_permutations=2000, n_jobs=-1,
                                            random_state=1, verbose=0, scoring='accuracy')
    return accuracy, p


def grid_search(x, y, pip, param_grid):
    gd_sr = GridSearchCV(pip, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    gd_sr.fit(x, y)
    params = gd_sr.best_params_
    return params


def svm_clf(x, y, pre_steps):
    # grid search pipeline
    steps = pre_steps + [('SVM', SVC())]
    pip = Pipeline(steps)

    param_grid = [{'SVM__C': np.logspace(-3, 3, 7),
                   'SVM__gamma': np.logspace(-3, 0, 4)}]

    grid_search = GridSearchCV(pip, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x, y)

    # best model
    best_model = grid_search.best_estimator_

    return best_model


def knn_clf(x, y, pre_steps):
    # grid search pipeline
    steps = pre_steps + [('knn', KNeighborsClassifier())]
    pip = Pipeline(steps)
    param_grid = [{'knn__n_neighbors': np.arange(1, 10)}]
    grid_search = GridSearchCV(pip, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x, y)

    # best model
    best_model = grid_search.best_estimator_
    return best_model


if __name__ == "__main__":
    cases = ['NB', 'SVM', 'KNN']
    fish_type = 'Tg'
    power_levels = [1, 1.2]
    time_in_min = 30  # using all 30 minutes data
    days = [5, 6, 7, 8]
    batches = [1, 2]  # [1, 2]
    plate = 1  # [1, 2]
    hour = 60

    case_list = []
    power_list = []
    day_list = []
    acc_list = []
    pvalue_list = []
    batch_list = []

    for power_level in power_levels:
        for day in days:
            for batch_idx in batches:
                data_dir = 'Processed_data/quantization/{}/batch{}/features'.format(fish_type, batch_idx)
                df = pd.read_csv(os.path.join(data_dir, '{}W-60h-{}dpf-0{}-{}-min.csv'.format(power_level, day,
                                                                                              plate, time_in_min)))
                # get labels
                label = df['label']

                # get features
                features = df.drop(['label'], axis=1)

                for case_name in cases:
                    acc, p_value = clf(features, label, case=case_name)
                    print('case_name: {} Accuracy: {:.3f}%, p-value: {:.3f}'.format(case_name, acc * 100, p_value))

                    case_list.append(case_name)
                    power_list.append(power_level)
                    batch_list.append(batch_idx)
                    day_list.append(day)
                    acc_list.append(acc)
                    pvalue_list.append(p_value)

    res_df = pd.DataFrame({'case': case_list, 'power': power_list, 'day': day_list, 'acc': acc_list,
                           'p-value': pvalue_list, 'batch': batch_list})
    res_df.to_csv('Analysis_Results/ML_results/Tg/Quan_Data_Classification/feature_selection/all.csv', index=False)
