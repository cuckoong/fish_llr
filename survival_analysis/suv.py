import numpy as np
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt

def get_time_observe(day_num):
    Time = np.concatenate((np.repeat(1, day_num[0]-day_num[1]),
                           np.repeat(2, day_num[1] - day_num[2]),
                           np.repeat(5, day_num[2] - day_num[3]),
                           np.repeat(6, day_num[3]-day_num[4]),
                           np.repeat(7, day_num[4]-day_num[5]),
                           np.repeat(8, day_num[6])),
                           axis=0)
    Observed = np.concatenate((np.repeat(1, day_num[0]-day_num[1]),
                               np.repeat(1, day_num[1]-day_num[2]),
                               np.repeat(1, day_num[2]-day_num[3]),
                               np.repeat(1, day_num[3]-day_num[4]),
                               np.repeat(1, day_num[4]-day_num[5]),
                               np.repeat(0, day_num[6]),
                               np.repeat(1, day_num[5]-day_num[6])),
                              axis=0)
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

day_0w_exp_1 = (96, 95, 90, 90, 90, 90, 90)
day_5w_exp_1 = (120, 117, 117, 111, 111, 111, 111)
day_5w_exp_2 = (120, 119, 110, 101, 101, 101, 101)

day_0w_control_1 = (96, 96, 95, 90, 90, 89, 88)
day_5w_control_1 = (96, 95, 95, 70, 70, 70, 70)
day_5w_control_2 = (96, 96, 87, 84, 84, 84, 84)

# day_exp = [day_0w_exp[i] +  day_1w_exp[i] + day_2w_exp[i] for i in range(len(day_0w_exp))]
# day_control = [day_0w_control[i] +  day_1w_control[i] + day_2w_control[i] for i in range(len(day_0w_control))]

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4, 4), sharey=True)
day_control = day_0w_control_1
day_exp = day_0w_exp_1
visualize(day_exp, day_control, '0W', axs[1])

day_control = [day_5w_control_1[i] + day_5w_control_2[i] for i in range(len(day_5w_control_1))]
day_exp = [day_5w_exp_1[i] + day_5w_exp_2[i] for i in range(len(day_5w_exp_1))]
visualize(day_exp, day_control, '5W', axs[0])


plt.ylabel('Survival Rate')
plt.legend(loc='lower left')
plt.title('')
axs[0].set_ylabel('Survival Rate')
plt.ylim(0.2, 1.2)

# plt.xlim(5,9)
for i in range(2):
    axs[i].set_xticks(np.arange(0,10))
    axs[i].set_xlabel('Days Post Fertilization')

plt.tight_layout()
