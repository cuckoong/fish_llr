# 2 way anova to analyse the radiation vs control
set.seed(345)
# load data from files
library(readr)
library(dplyr)
library(RcppRoll)

# load data ------------------------------------------------------------------------------------------------------------
myData <-  read_csv('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/stat_data/burdur_5w_batch1_burst4.csv')

# integraon over one min (60s)
myData <-  myData %>%
  select(burdur, animal, end, label, day) %>%
  mutate(animal = as.factor(animal),
     label = as.factor(label),
     day = as.factor(day)) %>%
  mutate(inte_end = end %/% 60) %>%
  group_by(day, label, animal, inte_end) %>%
  summarise(inte_burdur = sum(burdur)) %>%
  mutate(stim_stage = case_when(inte_end <= 30 ~ inte_end,
                              inte_end > 30 & inte_end <= 90 ~ inte_end,
                              inte_end > 90 & inte_end <= 150 ~ inte_end - 60)) %>%
  mutate(stim_stage = as.factor(stim_stage)) %>%
  select(inte_burdur, stim_stage, animal, label, day)


library(ggplot2)
ggplot(data = myData, aes(x = stim_stage, y = inte_burdur,color = label))+geom_line()


myData <-  myData %>%
  select(burdur, animal, end, label, day) %>%
  filter((end > 1770 & end <= 1830) | (end > 3570 & end <= 3630) |
  (end > 5370 & end <= 5430) |(end > 7170 & end <= 7230)) %>%
  mutate(stim_stage = case_when(end <= 1830 ~ end-1770,
                                end > 3570 & end <= 3630 ~ end - 3570 + 60,
                                end > 5370 & end <= 5430 ~ end - 5370,
                                end > 7170 & end <= 7230 ~ end - 7170 + 60)) %>%
  mutate(stim_group = case_when(end <= 1830 ~ 1,
                                end > 3570 & end <= 3630 ~ 2,
                                end > 5370 & end <= 5430 ~ 3,
                                end > 7170 & end <= 7230 ~ 4)) %>%
  select(burdur, animal, label, day, stim_stage, stim_group) %>%
  mutate(animal = as.factor(animal),
         label = as.factor(label),
         day = as.factor(day),
         stim_stage = as.factor(stim_stage),
         stim_group = as.factor(stim_group))



# 2 way anova with repeated measure
library(nlme)

# burdur, aname, label, day, stim_stage, stim_group
baseline <-  lme(inte_burdur~1, random=~1|animal/stim_stage/day, data = myData, method='ML')
# save model
dir <-  'Analysis_Results/Statistical_Results/linear_mixed_model_burst4_5W_batch1'

save(baseline, file= paste(dir,'lme_burst4_baseline_integrate.rda', sep="/"))

# load('lme_burst4_baseline.rda')
radiationM <- update(baseline, .~. + label)
save(radiationM, file= paste(dir,'lme_burst4_radiationM.rda', sep="/"))

dayM <- update(radiationM, .~. + day)
save(dayM, file= paste(dir,'lme_burst4_dayM.rda', sep="/"))

stim_stageM <- update(dayM, .~. + stim_stage)
save(stim_stageM, file= paste(dir,'lme_burst4_stim_stageM.rda', sep="/"))

# stim_groupM <- update(stim_stageM, .~. + stim_group)
radiation_day <- update(stim_stageM, .~. + label:day)
save(radiation_day, file= paste(dir, 'lme_burst4_radiation_day.rda', sep="/"))

radiation_stim <- update(radiation_day, .~. + label:stim_stage)
save(radiation_stim, file= paste(dir, 'lme_burst4_radiation_stim.rda', sep="/"))

day_stim <- update(radiation_stim, .~. + day:stim_stage)
save(day_stim, file= paste(dir, 'lme_burst4_day_stim.rda', sep="/"))

all <- update(day_stim, .~. + day:stim_stage:label)

save(all, file= paste(dir, 'lme_burst4_all.rda', sep="/"))

# visualization
anova(baseline, radiationM, dayM, stim_stageM, radiation_day)#, radiation_stim, day_stim, all)