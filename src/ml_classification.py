import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, chi2, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold, permutation_test_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd

import os

os.chdir('/Users/panpan/PycharmProjects/old_project/fish_llr/')


def clf(data, labels, case='SVM', cat_features=[], num_features=[]):
    numeric_transformer = Pipeline(
        steps=[
            ('variance_threshold', VarianceThreshold()),
            ('scaler', StandardScaler()),
            ('selection', SelectPercentile(percentile=20))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ('selection', SelectPercentile(chi2, percentile=20))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ]
    )

    if case == 'NB':
        steps = [('preprocessor', preprocessor),
                 ('NB', GaussianNB())]
        pip = Pipeline(steps)
        # permutation test
        # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        accuracy, _, p = permutation_test_score(pip, data, labels, cv=cv, n_permutations=2000, n_jobs=-1,
                                                random_state=1, verbose=0, scoring='accuracy')
        return accuracy, p

    if case == 'SVM':
        accuracy, p = svm_clf(data, labels, preprocessor)
        return accuracy, p

    elif case == 'KNN':
        accuracy, p = knn_clf(data, labels, preprocessor)
        return accuracy, p


def grid_search(x, y, pip, param_grid):
    gd_sr = GridSearchCV(pip, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    gd_sr.fit(x, y)
    params = gd_sr.best_params_
    return params


def svm_clf(x, y, preprocessor):
    # grid search pipeline
    steps = [('preprocessor', preprocessor),
             ('SVM', SVC())]
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
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    accuracy, _, p = permutation_test_score(model, x, y, cv=cv, n_permutations=2000, n_jobs=-1,
                                            random_state=1, verbose=0, scoring='accuracy')
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
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    accuracy, _, p = permutation_test_score(model, x, y, cv=cv, n_permutations=2000, n_jobs=-1,
                                            random_state=1, verbose=0, scoring='accuracy')
    return accuracy, p


if __name__ == "__main__":
    cases = ['NB', 'SVM', 'KNN']
    fish_type = 'Tg'
    power_levels = [1, 1.2]
    time_in_min = 30  # using all 30 minutes data
    days = [5, 6, 7, 8]
    batches = [1, 2]  # [1, 2]
    plates = [1]  # [1, 2]
    hour = 60

    batch_list = []
    power_list = []
    day_list = []
    feature_list = []
    label_list = []

    for power_level in power_levels:
        for batch_idx in batches:
            for day in days:
                data_dir = 'Processed_data/quantization/{}/batch{}/features'.format(fish_type, batch_idx)

                for plate in plates:
                    df = pd.read_csv(os.path.join(data_dir, '{}W-60h-{}dpf-0{}-{}-min.csv'.format(power_level, day,
                                                                                                  plate, time_in_min)))
                    # get labels
                    label = df['label']
                    label_list.append(label)

                    # get features
                    features = df.drop(['label'], axis=1)
                    feature_list.append(features)

                    # get other info
                    power_list.extend([power_level] * len(label))
                    batch_list.extend([batch_idx] * len(label))
                    day_list.extend([day] * len(label))

    feature = pd.concat(feature_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    power_level = np.array(power_list)
    batch_idx = np.array(batch_list)
    day = np.array(day_list)

    df = pd.DataFrame(data=dict(power=power_level, batch=batch_idx, day=day)).reset_index(drop=True)

    # convert to pandas dummy variables
    feature.reset_index(drop=True, inplace=True)

    # concat features and df
    input = pd.concat([feature, df], axis=1)

    # category and numerical features
    cat_features = ['power', 'batch', 'day']
    num_features = [col for col in input.columns if col not in cat_features]

    for case_name in cases:
        acc, p_value = clf(input, label, case=case_name,
                           cat_features=cat_features, num_features=num_features)
        print('case_name: {} Accuracy: {:.3f}%, p-value: {:.3f}'.format(case_name, acc * 100, p_value))
