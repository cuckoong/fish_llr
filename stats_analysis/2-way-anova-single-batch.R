# 2 way anova to analyse the radiation vs control
# seperate day
set.seed(345)
# load data from files
library(readr)
library(dplyr)
library(plotrix)

minmax_scale <- function(x){
  return((x - min(x)) / (max(x) - min(x)))
}

integrate_df <- function(file, batch_num, selected_day, integrate_duration=60){
  myData <- read_csv(file)
  myData <- myData %>%
    filter(day == selected_day) %>%
    select(burdur, animal, end, label, day) %>%
    mutate(animal = as.factor(animal),
       label = as.factor(label),
       day = as.factor(day)) %>%
    mutate(inte_end = (end-1) %/% integrate_duration + 1) %>%
    group_by(day, label, animal, inte_end) %>%
    summarise(inte_burdur = sum(burdur)) %>%
    ungroup() %>%
    group_by(day, label, animal) %>%
    mutate(minmax_bur = minmax_scale(inte_burdur)) %>%
    mutate(batch = batch_num) %>%
    mutate(animal_batch = paste(batch, animal, sep='-')) %>%
    mutate(animal_batch = as.factor(animal_batch),
           batch = as.factor(batch),
           label = as.factor(label)) %>%
    ungroup() %>%
    select(animal_batch, batch, label,inte_burdur, minmax_bur, inte_end)
  return(myData)
}

file1 <- '/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/burdur_0w_batch1_burst4.csv'
selected_day <- 5
myData <- integrate_df(file1, batch_num = 1, selected_day = selected_day)


Group_Data <- myData %>%
  group_by(label, inte_end, batch) %>%
  summarise(mean_burdur = mean(inte_burdur),
            sd_burdur = std.error(inte_burdur))

# visualize individual fish activity
library(ggplot2)
library(envalysis)
ggplot(data = Group_Data, aes(x=inte_end, y=mean_burdur, group = label, color = label)) +
  geom_pointrange(aes(ymin=mean_burdur-sd_burdur, ymax=mean_burdur+sd_burdur)) +
  geom_line() + theme_publish()


myData <-  myData %>%
  mutate(stim_stage = case_when(inte_end <= 90 ~ inte_end,
                                inte_end > 90 ~ (inte_end - 60))) %>%
  mutate(stim_stage = as.factor(stim_stage),
         inte_end = as.factor(inte_end))
  `

# 2 way anova with repeated measure
library(nlme)
# save model
dir <-  paste('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/Statistical_Results/0W/inte_end/day',selected_day, sep='')

# burdur, aname, label, day, stim_stage, stim_group
# baseline <-  lme(inte_burdur~1, random=~1|animal_batch/stim_stage, data = myData, method='ML',
#                  control=lmeControl(opt = "optim"))
baseline <-  lme(inte_burdur~1, random=~1|animal_batch/inte_end, data = myData, method='ML',
                 control=lmeControl(opt = "optim"))

save(baseline, file= paste(dir,'lme_burst4_baseline_integrate.rda', sep="/"))

# load('lme_burst4_baseline.rda')
radiationM <- update(baseline, .~. + label)
save(radiationM, file= paste(dir,'lme_burst4_radiationM.rda', sep="/"))

# stim_stageM <- update(radiationM, .~. + stim_stage)
stim_stageM <- update(radiationM, .~. + inte_end)
save(stim_stageM, file= paste(dir,'lme_burst4_stim_stageM.rda', sep="/"))

# radiation_stim <- update(stim_stageM, .~. + label:stim_stage)
radiation_stim <- update(stim_stageM, .~. + label:inte_end)
save(radiation_stim, file= paste(dir, 'lme_burst4_radiation_stim.rda', sep="/"))

# visualization
anova(baseline,radiationM, stim_stageM, radiation_stim)
