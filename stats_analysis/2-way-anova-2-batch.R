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

file1 <- '/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/burdur_5w_batch1_burst4.csv'
file2 <- '/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/burdur_5w_batch2_burst4.csv'
selected_day <- 8
myData1 <- integrate_df(file1, batch_num = 1, selected_day = selected_day)
myData2 <- integrate_df(file2, batch_num = 2, selected_day = selected_day)

# visualize all inte_end data
myData <- rbind(myData1, myData2)

Group_Data <- myData %>%
  group_by(label, inte_end, batch) %>%
  summarise(mean_burdur = mean(inte_burdur),
            sd_burdur = std.error(inte_burdur))

# visualize individual fish activity
library(ggplot2)
ggplot(data = Group_Data, aes(x=inte_end, y=mean_burdur, group = label, color = label)) +
  geom_pointrange(aes(ymin=mean_burdur-sd_burdur, ymax=mean_burdur+sd_burdur)) +
  geom_line() + facet_grid(.~batch)


myData <-  myData %>%
  mutate(stim_stage = case_when(inte_end <= 90 ~ inte_end,
                                inte_end > 90 ~ (inte_end - 60))) %>%
  mutate(stim_stage = as.factor(stim_stage),
         inte_end = as.factor(inte_end))


# 2 way anova with repeated measure
library(nlme)
# save model
dir <-  paste('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/Statistical_Results/5W/batch/day',selected_day, sep='')

# burdur, aname, label, day, stim_stage, stim_group
# baseline <-  lme(inte_burdur~1, random=~1|animal_batch/stim_stage, data = myData, method='ML',
                 # control=lmeControl(opt = "optim"))
baseline <-  lme(inte_burdur~1, random=~1|animal_batch/inte_end, data = myData, method='ML',
                  control=lmeControl(opt = "optim"))
save(baseline, file= paste(dir,'lme_burst4_baseline_integrate.rda', sep="/"))

# load('lme_burst4_baseline.rda')
radiationM <- update(baseline, .~. + label)
save(radiationM, file= paste(dir,'lme_burst4_radiationM.rda', sep="/"))

batchM <- update(radiationM, .~. + batch)
save(batchM, file= paste(dir,'lme_burst4_batchM.rda', sep="/"))

# stim_stageM <- update(batchM, .~. + stim_stage)
stim_stageM <- update(batchM, .~. + inte_end)
save(stim_stageM, file= paste(dir,'lme_burst4_stim_stageM.rda', sep="/"))

# stim_groupM <- update(stim_stageM, .~. + stim_group)
radiation_batch <- update(stim_stageM, .~. + label:batch)
save(radiation_batch, file= paste(dir, 'lme_burst4_radiation_batch.rda', sep="/"))

# radiation_stim <- update(radiation_batch, .~. + label:stim_stage)
radiation_stim <- update(radiation_batch, .~. + label:inte_end)
save(radiation_stim, file= paste(dir, 'lme_burst4_radiation_stim.rda', sep="/"))

# batch_stim <- update(radiation_stim, .~. + inte_end:stim_stage)
batch_stim <- update(radiation_stim, .~. + batch:inte_end)
save(batch_stim, file= paste(dir, 'lme_burst4_batch_stim.rda', sep="/"))

# all <- update(batch_stim, .~. + inte_end:stim_stage:label)
all <- update(batch_stim, .~. + batch:inte_end:label)
save(all, file= paste(dir, 'lme_burst4_all.rda', sep="/"))

# visualization
anova(baseline,radiationM, batchM, stim_stageM, radiation_batch, radiation_stim, batch_stim, all)
