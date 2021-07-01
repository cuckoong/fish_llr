# 2 way anova to analyse the radiation vs control
set.seed(35)
# load data from files
library(readr)
library(dplyr)

# load data ------------------------------------------------------------------------------------------------------------
myData <-  read_csv('/linear_mixed_model_burst0_5W_batch1/burdur_5w_all_burst0.csv')
myData <-  myData %>%
  select(burdur, animal, end, label, radiation, day) %>%
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
baseline <-  lme(burdur~1, random=~1|animal/stim_stage/day, data = myData, method='ML')
# save model
save(baseline, file= 'linear_mixed_model_burst0_5W_batch1/lme_burst0_baseline.rda')
# load('lme_burst4_baseline.rda')
radiationM <- update(baseline, .~. + label)
save(radiationM, file= 'linear_mixed_model_burst0_5W_batch1/lme_burst0_radiationM.rda')

dayM <- update(radiationM, .~. + day)
save(dayM, file= 'linear_mixed_model_burst0_5W_batch1/lme_burst0_dayM.rda')

stim_stageM <- update(dayM, .~. + stim_stage)
save(stim_stageM, file= 'linear_mixed_model_burst0_5W_batch1/lme_burst0_stim_stageM.rda')


# stim_groupM <- update(stim_stageM, .~. + stim_group)
radiation_day <- update(stim_stageM, .~. + label:day)
save(radiation_day, file= 'linear_mixed_model_burst0_5W_batch1/lme_burst0_radiation_day.rda')

radiation_stim <- update(radiation_day, .~. + label:stim_stage)
save(radiation_stim, file= 'linear_mixed_model_burst0_5W_batch1/lme_burst0_radiation_stim.rda')

day_stim <- update(radiation_stim, .~. + day:stim_stage)
save(day_stim, file= 'linear_mixed_model_burst0_5W_batch1/lme_burst0_day_stim.rda')

all <- update(day_stim, .~. + day:stim_stage:label)

save(all, file= 'linear_mixed_model_burst0_5W_batch1/lme_burst0_all.rda')

save(all, file= 'linear_mixed_model_burst4_5W_batch1/lme_burst4_all.rda')

# visualization
anova(baseline, radiationM, dayM, stim_stageM, radiation_day, radiation_stim, day_stim, all)