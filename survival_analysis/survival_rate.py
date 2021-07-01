import numpy as np
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt

def get_time_observe(day_num):
    # Time = np.concatenate((np.repeat(0, day_num[0]),
    #                        np.repeat(5, day_num[0]),
    #                        np.repeat(6, day_num[0]),
    #                        np.repeat(7, day_num[0]),
    #                        np.repeat(8, day_num[0])),
    #                        axis=0)
    # Observed = np.concatenate((np.repeat(0, day_num[0]),
    #                            np.repeat(0, day_num[1]), np.repeat(1, day_num[0]-day_num[1]),
    #                            np.repeat(0, day_num[2]), np.repeat(1, day_num[0]-day_num[2]),
    #                            np.repeat(0, day_num[3]), np.repeat(1, day_num[0]-day_num[3]),
    #                            np.repeat(0, day_num[4]), np.repeat(1, day_num[0]-day_num[4])), axis=0)
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

day_0w_exp = (120, 96, 72, 70, 68)
day_1w_exp = (120, 111, 111, 111, 109)
day_2w_exp = (108, 96, 91, 92, 91)

day_0w_control = (96, 93, 92, 91, 81)
day_1w_control = (96, 77, 75, 73, 72)
day_2w_control = (96, 67, 58, 52, 51)

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
