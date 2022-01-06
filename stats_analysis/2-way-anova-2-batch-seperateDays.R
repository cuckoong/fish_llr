# 2 way anova to analyse the radiation vs control
# seperate day
set.seed(345)
# load data from files
library(readr)
library(dplyr)
selected_day <- 5
myData <- read_csv('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/burdur_5w_burst4.csv')
myData <-  myData %>%
  filter(day==selected_day)  %>%
  mutate(animal_batch = paste(batch, animal, sep='-')) %>%
  mutate(base_stim = case_when(stim_stage < 30 ~ 1,
                          stim_stage >= 30 ~ 2)) %>%
  mutate(animal_batch = as.factor(animal_batch), label = as.factor(label), day = as.factor(day),
         stim_stage = as.factor(stim_stage), batch = as.factor(batch)) %>%
  group_by(batch, label, animal_batch) %>%
  mutate(sd_burdur = sd(inte_burdur[base_stim==1]), mean_burdur = mean(inte_burdur[base_stim==1])) %>%
  mutate(normal_burdur = (inte_burdur - mean(inte_burdur[base_stim==1])))


%>%
  select(normal_burdur, stim_stage, animal_batch, label, batch, base_stim) %>%


baseline_data <- myData %>%
  group_by(batch, label, animal_batch, base_stim) %>%
  summarise(var = var(inte_burdur), mean = mean(inte_burdur)) %>%
  filter(base_stim==1)


# 2 way anova with repeated measure
library(nlme)

# burdur, aname, label, day, stim_stage, stim_group
baseline <-  lme(inte_burdur~1, random=~1|animal_batch/stim_stage, data = myData, method='ML')
# save model
dir <-  '/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/Statistical_Results/5W/day5'
save(baseline, file= paste(dir,'lme_burst4_baseline_day.rda', sep="/"))

# load('lme_burst4_baseline.rda')
radiationM <- update(baseline, .~. + label)
save(radiationM, file= paste(dir,'lme_burst4_radiationM.rda', sep="/"))

batchM <- update(radiationM, .~. + batch)
save(batchM, file= paste(dir,'lme_burst4_stim_batchM.rda', sep="/"))

stim_stageM <- update(batchM, .~. + stim_stage)
save(stim_stageM, file= paste(dir,'lme_burst4_stim_stageM.rda', sep="/"))

radiation_stim <- update(stim_stageM, .~. + label:stim_stage)
save(radiation_stim, file= paste(dir, 'lme_burst4_radiation_stim.rda', sep="/"))

radiation_batch <- update(radiation_stim, .~. + label:batch)
save(radiation_batch, file= paste(dir, 'lme_burst4_radiation_batch.rda', sep="/"))

stim_batch <- update(radiation_batch, .~. + batch:stim_stage)
save(stim_batch, file= paste(dir, 'lme_burst4_stim_batch.rda', sep="/"))

all <- update(day_stim, .~. + day:stim_stage:label)

save(all, file= paste(dir, 'lme_burst4_all.rda', sep="/"))

# visualization
anova(baseline, radiationM, stim_stageM, batchM, radiation_stim, radiation_batch,  )#, day_stim, all)