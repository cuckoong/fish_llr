import numpy as np
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt


def get_time_observe(day_num):
    Time = np.concatenate((np.repeat(5, day_num[0]-day_num[1]),
                           np.repeat(6, day_num[1]-day_num[2]),
                           np.repeat(7, day_num[2]-day_num[3]),
                           np.repeat(8, day_num[3])),
                           axis=0)
    Observed = np.concatenate((np.repeat(1, day_num[0]-day_num[1]),
                               np.repeat(1, day_num[1]-day_num[2]),
                               np.repeat(1, day_num[2]-day_num[3]),
                               np.repeat(0, day_num[4]), np.repeat(1, day_num[3]-day_num[4])), axis=0)
    return Time, Observed


day_0w_exp_1 = (96, 95, 90, 90, 90, 90, 90)
day_5w_exp_1 = (120, 117, 117, 111, 111, 111, 111)
day_5w_exp_2 = (120, 119, 110, 101, 101, 101, 101)

day_0w_control = (96, 96, 95, 90, 90, 89, 88)
day_5w_control_1 = (96, 95, 95, 70, 70, 70, 70)
day_5w_control_2 = (96, 96, 87, 84, 84, 84, 84)

day_exp = day_2w_exp
day_control = [day_0w_control[i] +  day_1w_control[i] + day_2w_control[i] for i in range(len(day_0w_control))]

Time0, Observed0 = get_time_observe(day_0w_exp)
Time1, Observed1 = get_time_observe(day_1w_exp)
Time2, Observed2 = get_time_observe(day_2w_exp)
Timec, Observedc = get_time_observe(day_control)

kmf0 = KaplanMeierFitter()
kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()
kmfc = KaplanMeierFitter()

kmf0.fit(Time0, event_observed=Observed0, label = '0W')
kmf1.fit(Time1, event_observed=Observed1, label='1W')
kmf2.fit(Time2, event_observed=Observed2, label = '2.5W')
kmfc.fit(Timec, event_observed=Observedc, label = 'Control')

ax = plt.subplot(111)
kmf0.plot_survival_function(ax=ax)
kmf1.plot_survival_function(ax=ax)
kmf2.plot_survival_function(ax=ax)
kmfc.plot_survival_function(ax=ax)

plt.title('Survival function of zebrafish')
plt.ylabel('Survival Rate')
# plt.xlim(5,9)
# plt.xticks(np.arange(5,10), np.arange(5,10))

Time = np.concatenate((Time0, Time1, Time2, Timec), axis=0)
Observed = np.concatenate((Observed0, Observed1, Observed2, Observedc), axis=0)
Group = np.concatenate((np.repeat(0, day_0w_exp[0]), np.repeat(1, day_1w_exp[0]),
                        np.repeat(2, day_2w_exp[0]), np.repeat(3, day_control[0])),
                       axis=0)


from lifelines.statistics import multivariate_logrank_test
results = multivariate_logrank_test(Time,Group,Observed)
results.print_summary()
print(results.p_value)
print(results.test_statistic)
