#visualize radiation and control swiming activity (all swiming distance)

set.seed(345)
# load data from files
library(readr)
library(dplyr)
library(plotrix)


minmax_scale <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}

# file - burst data file, no need to re-integrate
# as it is already integrated in software (60s)
get_df <- function(file, batch_num, selected_day, integrate_duration = 60){
  myData <- read_csv(file)
  myData <- myData %>%
    filter(day == selected_day) %>%
    select(activity_sum, animal, end, label, day) %>%
    mutate(animal = as.factor(animal),
       radiation_label = as.factor(label),
       day = as.factor(day)) %>%
    mutate(inte_end = end %/% integrate_duration) %>%
    group_by(day, radiation_label, animal) %>%
    mutate(scaled_activity_sum = minmax_scale(activity_sum)) %>%   # minmax scaling
    mutate(batch = batch_num) %>%
    mutate(animal_batch = paste(batch, animal, sep='-')) %>%
    mutate(animal_batch = as.factor(animal_batch),
           batch = as.factor(batch)) %>%
    ungroup() %>%
    select(animal_batch, batch, radiation_label, inte_end, activity_sum, scaled_activity_sum)
  return(myData)
}


setwd('/Users/panpan/PycharmProjects/old_project/fish_llr')
# file1 <- 'Processed_data/quantization/Tg/stat_data/burdur_1.2w_60h_batch1_burst4.csv'
file1 <- 'Processed_data/quantization/Tg/stat_data/all_1.2w_60h_batch1_burst4.csv'
selected_day <- 5

# intergrate every 60s
myData <- get_df(file1, batch_num = 1, selected_day = selected_day)

# remove baseline activity (0 ~ 30s)
rm_baseline_Data <-  myData %>%
  group_by(animal_batch) %>%
  mutate(stim_baseline = case_when(inte_end <= 30 ~ 0,
                                inte_end > 30 ~ 1)) %>%
  # mutate(rm_activity_sum = (activity_sum - mean(activity_sum[stim_baseline==0]))) %>%
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
  # geom_pointrange(aes(ymin=Q1, ymax=Q3), alpha=0.5) +
  geom_pointrange(aes(ymin=mean_activity_sum-sd_activity_sum,
                      ymax=mean_activity_sum+sd_activity_sum)) +
  ylab('Acitivty Duration (s)') + xlab('Tracking Time (min)') +
  scale_color_discrete(name='', labels = c('Control', '1.2W'), breaks = c(0, 1)) +
  theme(legend.position = 'right') +
  annotate("text", x = 15, y = 0.5, label = 'Baseline\nOFF', color = 'Blue') +
  annotate("text", x = c(45,105), y = 0.5, label = 'ON', color = 'Red') +
  annotate("text", x = c(75,135), y = 0.5, label = 'OFF', color = 'Blue') +
  geom_vline(xintercept = c(30, 60, 90, 120), linetype = 'dotted') +
  theme_publish()
ggsave(paste0('Figures/Stats/Tracking/Tg/', 1.2,
              'W_day', selected_day, '_batch_', 1, 'mean.png'),
       width=8, height=6, units='in', dpi=300)

# median and IQR activity
ggplot(data = Group_Data, aes(x=inte_end, y=median_activity_sum,
                              group = radiation_label, color = radiation_label)) +
  geom_line()+
  geom_pointrange(aes(ymin=Q1, ymax=Q3), alpha=0.5) +
  ylab('Activity Duration (s)') + xlab('Tracking Time (min)') +
  scale_color_discrete(name='', labels = c('Control', '1.2W'), breaks = c(0, 1)) +
  theme(legend.position = 'right') +
  annotate("text", x = 15, y = 0.2, label = 'Baseline\nOFF', color = 'Blue') +
  annotate("text", x = c(45,105), y = 0.2, label = 'ON', color = 'Red') +
  annotate("text", x = c(75,135), y = 0.2, label = 'OFF', color = 'Blue') +
  geom_vline(xintercept = c(30, 60, 90, 120), linetype = 'dotted') +
  theme_publish()
ggsave(paste0('Figures/Stats/Tracking/Tg/', 1.2,
              'W_day', selected_day, '_batch_', 1, 'median.png'),
       width=8, height=6, units='in', dpi=300)


