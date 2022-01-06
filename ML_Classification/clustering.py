import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, permutation_test_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd


def clf(data, label, case='SVM'):
    if case == 'SVM':
        steps = [('scaler', StandardScaler()), ('SVM', SVC())]
        pip = Pipeline(steps)
        param_grid = [{'SVM__C': np.logspace(-3, 3, 7),
                       'SVM__gamma': np.logspace(-3, 0, 4)}]

    elif case == 'KNN':
        steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
        pip = Pipeline(steps)
        param_grid = [{'knn__n_neighbors': np.arange(1, 10)}]

    elif case == 'NB':
        steps = [('scaler', StandardScaler()), ('NB', GaussianNB())]
        pip = Pipeline(steps)
        # permutation test
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
        acc, _, pvalue = permutation_test_score(pip, data, label, cv=cv, n_permutations=500, n_jobs=-1,
                                                random_state=1, verbose=0, scoring=('balanced_accuracy'))
        return acc, pvalue

    gd_sr = GridSearchCV(pip, param_grid=param_grid, scoring='balanced_accuracy', cv=5, n_jobs=-1)
    gd_sr.fit(data, label)
    params = gd_sr.best_params_

    if case == 'SVM':
        best_C = params['SVM__C']
        best_gamma = params['SVM__gamma']
        # best model
        model = SVC(C=best_C, gamma=best_gamma)
    elif case == 'KNN':
        n_neighbors = params['knn__n_neighbors']
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # permutation test
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    acc, _, pvalue = permutation_test_score(model, data, label, cv=cv, n_permutations=500, n_jobs=-1,
                                            random_state=1, verbose=0, scoring=('balanced_accuracy'))
    return acc, pvalue


if __name__ == "__main__":
    case = 'NB'
    power = 0
    times = ['baseline', 1, 2, 30]
    days = [5, 6, 7, 8]
    batches = [1]

    power_list = []
    time_list = []
    day_list = []
    acc_list = []
    pvalue_list = []

    for time in times:
        for day in days:
            data_list = []
            label_list = []
            for batch in batches:
                dir = '/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/ML_results/burst_4/batch{}/'.format(
                    batch)
                for plate in [1, 2]:
                    data_plate = np.load(
                        dir + str(power) + 'W-60h-' + str(day) + 'dpf-0' + str(plate) + '-' + str(time) +
                        'min-feature.npy')
                    label_plate = np.load(
                        dir + str(power) + 'W-60h-' + str(day) + 'dpf-0' + str(plate) + '-' + 'label.npy')
                    data_list.append(data_plate)
                    label_list.append(label_plate)

            data = np.concatenate(data_list, axis=0)
            label = np.concatenate(label_list, axis=0)

            acc, pvalue = clf(data, label, case=case)
            power_list.append(power)
            time_list.append(time)
            day_list.append(day)
            acc_list.append(acc)
            pvalue_list.append(pvalue)

    df = pd.DataFrame(data=dict(power=power_list, time=time_list, day=day_list, acc=acc_list, pvalue=pvalue_list))
    df.to_csv('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/ML_results/ML_acc/' + case +
              '_{}w_1.csv'.format(power))
