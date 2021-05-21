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

def visualize(day_exp, day_control, group, ax):
    Time0, Observed0 = get_time_observe(day_exp)
    Timec, Observedc = get_time_observe(day_control)

    kmf0 = KaplanMeierFitter()
    kmfc = KaplanMeierFitter()

    kmf0.fit(Time0, event_observed=Observed0, label = group)
    kmfc.fit(Timec, event_observed=Observedc, label = 'Control')

    # plt.figure(figsize=(3,3))
    # ax = plt.subplot(111)
    kmf0.plot_survival_function(ax=ax)
    kmfc.plot_survival_function(ax=ax)

    Time = np.concatenate((Time0, Timec), axis=0)
    Observed = np.concatenate((Observed0, Observedc), axis=0)
    Group = np.concatenate((np.repeat(0, day_exp[0]), np.repeat(1, day_control[0])),
                           axis=0)
    #
    # from lifelines.utils import survival_table_from_events
    # table = survival_table_from_events(Time, Observed)
    # print(table.head())

    from lifelines.statistics import logrank_test
    results = logrank_test(Time,Group,Observed)
    print(results.p_value)
    print(results.test_statistic)

    return ax


day_0w_exp = (120, 96, 72, 70, 68)
day_1w_exp = (120, 111, 111, 111, 109)
day_2w_exp = (108, 96, 77, 73, 71)

day_0w_control = (96, 93, 92, 91, 81)
day_1w_control = (96, 77, 75, 73, 72)
day_2w_control = (96, 67, 58, 52, 51)

# day_exp = [day_0w_exp[i] +  day_1w_exp[i] + day_2w_exp[i] for i in range(len(day_0w_exp))]
# day_control = [day_0w_control[i] +  day_1w_control[i] + day_2w_control[i] for i in range(len(day_0w_control))]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(7,2.5), sharey=True)
day_control = day_2w_control
day_exp = day_2w_exp
visualize(day_exp, day_control, '2.5W', axs[2])

day_control = day_1w_control
day_exp = day_1w_exp
visualize(day_exp, day_control, '1W', axs[1])

day_control = day_0w_control
day_exp = day_0w_exp
visualize(day_exp, day_control, '0W', axs[0])


plt.ylabel('Survival Rate')
plt.legend(loc='lower left')
plt.title('')
axs[0].set_ylabel('Survival Rate')
plt.ylim(0.2, 1.2)

# plt.xlim(5,9)
for i in range(3):
    axs[i].set_xticks(np.arange(0,10))
    axs[i].set_xlabel('Days Post Fertilization')

plt.tight_layout()