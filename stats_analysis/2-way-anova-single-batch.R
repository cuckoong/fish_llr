# 2 way anova to analyse the radiation vs control
# seperate day
set.seed(345)
# load data from files
library(readr)
library(dplyr)
library(plotrix)

# minmax_scale <- function(x){
#   return((x - min(x)) / (max(x) - min(x)))
# }

integrate_df <- function(file, batch_num, selected_day, integrate_duration=60){
  myData <- read_csv(file)
  myData <- myData %>%
    filter(day == selected_day) %>%
    select(burdur, animal, end, label, day) %>%
    mutate(animal = as.factor(animal),
       radiation_label = as.factor(label),
       day = as.factor(day)) %>%
    mutate(inte_end = (end-1) %/% integrate_duration + 1) %>%
    group_by(day, radiation_label, animal, inte_end) %>%
    summarise(inte_burdur = sum(burdur)) %>%
    ungroup() %>%
    group_by(day, radiation_label, animal) %>%
    # mutate(minmax_bur = minmax_scale(inte_burdur)) %>%
    mutate(batch = batch_num) %>%
    mutate(animal_batch = paste(batch, animal, sep='-')) %>%
    mutate(animal_batch = as.factor(animal_batch),
           batch = as.factor(batch)) %>%
    ungroup() %>%
    select(animal_batch, batch, radiation_label, inte_end, inte_burdur)
  return(myData)
}

file1 <- '/Users/panpan/PycharmProjects/old_project/fish_llr/Analysis_Results/stat_data/burdur_1.2w_60h_batch3_burst4.csv'
selected_day <- 5

myData <- integrate_df(file1, batch_num = 1, selected_day = selected_day)


myData <-  myData %>%
  group_by(animal_batch) %>%
  mutate(stim_baseline = case_when(inte_end <= 30 ~ 0,
                                inte_end > 30 ~ 1)) %>%
  mutate(rm_burdur = (inte_burdur - mean(inte_burdur[stim_baseline==0]))) %>%
  filter(stim_baseline == 1) %>%
  select(animal_batch, radiation_label, inte_end, rm_burdur)


Group_Data <- myData %>%
  group_by(radiation_label, inte_end) %>%
  summarise(median_burdur = median(rm_burdur),
            mean_burdur = mean(rm_burdur),
            sd_burdur = sd(rm_burdur),
            Q1 = quantile(rm_burdur, 0.25),
            Q3 = quantile(rm_burdur, 0.75))

# visualize individual fish activity
library(ggplot2)
library(envalysis)
ggplot(data = Group_Data, aes(x=inte_end, y=mean_burdur, group = radiation_label, color = radiation_label)) +
  geom_line()+
  # geom_pointrange(aes(ymin=Q1, ymax=Q3), alpha=0.5) +
  geom_pointrange(aes(ymin=mean_burdur-sd_burdur, ymax=mean_burdur+sd_burdur)) +
  ylab('Burst Duration (s)') + xlab('Tracking Time (min)') +
  scale_color_discrete(name='', labels = c('Control', '0W'), breaks = c(0, 1)) +
  theme(legend.position = 'right') +
  # annotate("text", x = 15, y = 12, label = 'Baseline\nOFF', color = 'Blue') +
  # annotate("text", x = c(45,105), y = 12, label = 'ON', color = 'Red') +
  # annotate("text", x = c(75,135), y = 12, label = 'OFF', color = 'Blue') +
  # geom_vline(xintercept = c(30, 60, 90, 120), linetype = 'dotted') +
  theme_publish()
ggsave(paste0('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/Visualization/Visual_', 0,
              'W_day', selected_day, '_', 1, '.png'), width=8, height=6, units='in', dpi=300)

# myData <-  myData %>%
#   mutate(stim_stage = case_when(inte_end <= 90 ~ inte_end,
#                                 inte_end > 90 ~ (inte_end - 60))) %>%
#   mutate(stim_stage = as.factor(stim_stage),
#          inte_end = as.factor(inte_end))


# 2 way anova with repeated measure
library(nlme)
# save model
dir <- paste0('/home/tmp2/PycharmProjects/fish_llr/Analysis_Results/Statistical_Results/0W/inte_end/day', selected_day)

# anova
attach(myData)
nestinginfo <- groupedData(rm_burdur ~ radiation_label | animal_batch, data= myData)
fit.compsym <- gls(rm_burdur ~ factor(radiation_label)*factor(inte_end), data=nestinginfo,
                   corr=corCompSymm(, form= ~ 1 | animal_batch))

# fit.nostruct <- gls(inte_burdur ~ factor(radiation_label)*factor(inte_end), data=nestinginfo,
#                     corr=corSymm(, form= ~ 1 | animal_batch), weights = varIdent(form = ~ 1 | inte_end))
fit.ar1 <- gls(rm_burdur ~ factor(radiation_label)*factor(inte_end), data=nestinginfo,
               corr=corAR1(, form= ~ 1 | animal_batch))

anova(fit.compsym, fit.ar1)#, fit.ar1het) #compares the models
save(fit.ar1, file= paste(dir, 'ar1.rda', sep="/"))
anova(fit.ar1)
tmp = summary(fit.ar1)

fit.ar1het <- gls(rm_burdur ~ factor(radiation_label)*factor(inte_end), data=nestinginfo,
                  corr=corAR1(, form= ~ 1 | animal_batch), weights=varIdent(form = ~ 1 | inte_end))

anova(fit.compsym, fit.ar1, fit.ar1het) #compares the models

fit.ar1polytime <- gls(inte_burdur ~ factor(radiation_label)*poly(inte_end, degree = 3), data=nestinginfo,
                       corr=corAR1(, form= ~ 1 | inte_end))

summary(fit.ar1polytime)

anova(fit.compsym)
anova(fit.ar1)
anova(fit.ar1polytime)
anova(fit.ar1polytime, fit.ar1)


# burdur, aname, label, day, stim_stage, stim_group
# baseline <-  lme(inte_burdur~1, random=~1|animal_batch/stim_stage, data = myData, method='ML',
#                  control=lmeControl(opt = "optim"))
baseline <-  lme(inte_burdur~1, random=~1|inte_end, data = myData, method='ML',
                 control=lmeControl(opt = "optim"))
save(baseline, file= paste(dir,'lme_burst4_baseline_integrate.rda', sep="/"))

# load('lme_burst4_baseline.rda')
radiationM <- update(baseline, .~. + radiation_label, control=lmeControl(opt = "optim"))
save(radiationM, file= paste(dir,'lme_burst4_radiationM.rda', sep="/"))

# stim_stageM <- update(radiationM, .~. + stim_stage)
stim_stageM <- update(radiationM, .~. + inte_end, correlation = corAR1(),
                      control=lmeControl(opt = "optim"))
save(stim_stageM, file= paste(dir,'lme_burst4_stim_stageM.rda', sep="/"))

# radiation_stim <- update(stim_stageM, .~. + label:stim_stage)
radiation_stim <- update(stim_stageM, .~. + radiation_label:inte_end, correlation = corAR1(),
                         control=lmeControl(opt = "optim"))
save(radiation_stim, file= paste(dir, 'lme_burst4_radiation_stim.rda', sep="/"))

# visualization
anova(baseline,radiationM, stim_stageM, radiation_stim)
