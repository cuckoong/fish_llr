a#visualize radiation and control burst activity

set.seed(345)
# load data from files
library(readr)
library(dplyr)
library(plotrix)

minmax_scale <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}

# file - burst data file, integrate every 60s
integrate_df <- function(file, batch_num, selected_day, integrate_duration=60){
  myData <- read_csv(file)
  myData <- myData %>%
    filter(day == selected_day) %>%
    select(activity_sum, animal, end, label, day) %>%
    mutate(animal = as.factor(animal),
       radiation_label = as.factor(label),
       day = as.factor(day)) %>%
    mutate(inte_end = (end-1) %/% integrate_duration + 1) %>%
    group_by(day, radiation_label, animal, inte_end) %>%
    summarise(inte_activity_sum = sum(activity_sum)) %>%
    ungroup() %>%
    group_by(day, radiation_label, animal) %>%
    mutate(scaled_activity_sum = minmax_scale(inte_activity_sum)) %>%
    mutate(batch = batch_num) %>%
    mutate(animal_batch = paste(batch, animal, sep='-')) %>%
    mutate(animal_batch = as.factor(animal_batch),
           batch = as.factor(batch)) %>%
    ungroup() %>%
    select(animal_batch, batch, radiation_label, inte_end, inte_activity_sum, scaled_activity_sum)
  return(myData)
}


setwd('/Users/panpan/PycharmProjects/old_project/fish_llr')

BATCH <- c(1, 2)  # 1 or 2
POWER <- 1.2  # power of the burst activity
ACTIVITY_TYPE <- 'burdur' #'burdur' # 'burdur' or 'all'

file1 <- paste0('Processed_data/quantization/Tg/stat_data/', ACTIVITY_TYPE, '_', POWER, 'w_60h_batch', 1, '_burst4.csv')
file2 <- paste0('Processed_data/quantization/Tg/stat_data/', ACTIVITY_TYPE, '_', POWER, 'w_60h_batch', 2, '_burst4.csv')

selected_day <- 6

# intergrate every 60s
myData1 <- integrate_df(file1, batch_num = 1, selected_day = selected_day)
myData2 <- integrate_df(file2, batch_num = 2, selected_day = selected_day)

myData <- rbind(myData1, myData2)

# remove baseline activity (0 ~ 30s)
rm_baseline_Data <-  myData %>%
  group_by(animal_batch) %>%
  mutate(stim_baseline = case_when(inte_end <= 30 ~ 0,
                                inte_end > 30 ~ 1)) %>%
  # mutate(rm_activity_sum = (inte_activity_sum - mean(inte_activity_sum[stim_baseline==0]))) %>%
   mutate(rm_activity_sum = (scaled_activity_sum - mean(scaled_activity_sum[stim_baseline==0]))) %>%   # minmax scaling
  # filter(stim_baseline == 1) %>%
  select(animal_batch, radiation_label, inte_end, rm_activity_sum)

# group data with same radiation level and same time point
Group_Data <- rm_baseline_Data %>%
  group_by(radiation_label, inte_end) %>%
  summarise(median_activity_sum = median(rm_activity_sum),
            mean_activity_sum = mean(rm_activity_sum),
            sd_activity_sum = sd(rm_activity_sum),
            Q1 = quantile(rm_activity_sum, 0.25),
            Q3 = quantile(rm_activity_sum, 0.75))

# visualize individual fish activity
library(ggplot2)
library(envalysis)

# mean and sd activity
ggplot(data = Group_Data, aes(x=inte_end, y=mean_activity_sum,
                              group = radiation_label, color = radiation_label)) +
  geom_line()+
  geom_point()+
  geom_ribbon(aes(ymin=mean_activity_sum-sd_activity_sum,
                  ymax=mean_activity_sum+sd_activity_sum,
                  fill=radiation_label), alpha=0.1) +
  # geom_pointrange(aes(ymin=Q1, ymax=Q3), alpha=0.5) +
  # geom_pointrange(aes(ymin=mean_activity_sum-sd_activity_sum,
  #                     ymax=mean_activity_sum+sd_activity_sum)) +
  ylab('Acitivty Duration (s)') + xlab('Tracking Time (min)') +
  scale_color_discrete(name='', labels = c('Control', '1.2W'), breaks = c(0, 1)) +
  theme(legend.position = 'right') +
  annotate("text", x = 15, y = 1, label = 'Baseline\nOFF', color = 'Blue') +
  annotate("text", x = c(45,105), y = 1, label = 'ON', color = 'Red') +
  annotate("text", x = c(75,135), y = 1, label = 'OFF', color = 'Blue') +
  geom_vline(xintercept = c(30, 60, 90, 120), linetype = 'dotted') +
  theme_publish()
ggsave(paste0('Figures/Stats/Quantization/Tg/', POWER,
              'W_day', selected_day, '_batches_mean.png'),
       width=8, height=6, units='in', dpi=300)

# median and IQR activity
ggplot(data = Group_Data, aes(x=inte_end, y=median_activity_sum,
                              group = radiation_label, color = radiation_label)) +
  geom_line()+
  geom_point()+
  geom_ribbon(aes(ymin=Q1, ymax=Q3, fill=radiation_label), alpha=0.1) +
  # geom_errorbar(aes(ymin=Q1, ymax=Q3), alpha=0.5) +
  # geom_pointrange(aes(ymin=Q1, ymax=Q3), alpha=1) +
  ylab('Activity Duration (s)') + xlab('Tracking Time (min)') +
  scale_color_discrete(name='', labels = c('Control', '1.2W'), breaks = c(0, 1)) +
  theme(legend.position = 'right') +
  annotate("text", x = 15, y = 1, label = 'Baseline\nOFF', color = 'Blue') +
  annotate("text", x = c(45,105), y = 1, label = 'ON', color = 'Red') +
  annotate("text", x = c(75,135), y = 1, label = 'OFF', color = 'Blue') +
  geom_vline(xintercept = c(30, 60, 90, 120), linetype = 'dotted') +
  theme_publish()
ggsave(paste0('Figures/Stats/Quantization/Tg/', POWER,
              'W_day', selected_day, '_batches_median.png'),
       width=8, height=6, units='in', dpi=300)


