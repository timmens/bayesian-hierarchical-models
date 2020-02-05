#Data Analysis term paper

# Load required packages
# Required Packages
library(mlmRev)
library(lme4)
library(rstanarm)
library(ggplot2)
library(stargazer)

# Use dataset from the mlmRev package: GCSE exam scores
data(Gcsemv, package = "mlmRev")
table <- summary(Gcsemv)

# Make Male the reference category and rename variable
Gcsemv$female <- relevel(Gcsemv$gender, "M")


# Use only total score on coursework paper 
GCSE <- subset(x = Gcsemv, 
               select = c(school, student, female, course))

# Count unique schools and students
J <- length(unique(GCSE$school))
N <- nrow(GCSE)

# visualise the data of the data
hist(GCSE$course) 


#------------------------------------------------------------------------------------------------#
#Estimation of multilevle models using a partilal maximum likelihood implementation

## Model 1
M1 <- lmer(formula = course ~ 1 + female + (1 + female | school), 
           data = GCSE, 
           REML = FALSE)

Summ_M1 = summary(M1) 
capture.output(Summ_M1, file = "M1_ML.txt")

# Get shrinkage factor
head(ranef(M1)$school)

#--------------------------------------------------------------------------------------------------#
#Estimation of multilevel models using a full bayesian framework

## Model 1
M1_stanlmer <- stan_lmer(formula = course ~ female + (1 + female | school), 
                         data = GCSE,
                         seed = 349)

# Obtain a summary of priors used
prior_summary(object = M1_stanlmer)

# Display posterior medians and posterior median absolute deviations
BaySumm_M1 = print(M1_stanlmer, digits = 2)
capture.output(BaySumm_M1, file = "M1_bay.txt")

# Get shrinkage factor
head(ranef(M1_stanlmer)$school)

## Plot convergence plot
plot(M1_stanlmer, "rhat")
dev.copy(png,'rhat.png')
dev.off()

## Plot convergence plot
plot(M1_stanlmer, "rhat")
dev.copy(png,'ess.png')
dev.off()
#_____________________________________________________________________________________________
# Complete-pooling regression
pooled <- lm(formula = course ~ female,
             data = GCSE)
a_pooled <- coef(pooled)[1]   # complete-pooling intercept
b_pooled <- coef(pooled)[2]   # complete-pooling slope

# No-pooling regression: fitting separate regression for each school
a_nopooled <- numeric()
b_nopooled <- numeric()
for(j in GCSE$school){
  if(j == "84707"){           # only 1 student in this school
    a_nopooled[j] <- NA
    b_nopooled[j] <- NA
  } else {
    sch_lm <- lm(formula = course ~ female, 
                 data = subset(GCSE, school == j))
    a_nopooled[j] <- coef(sch_lm)[1]
    b_nopooled[j] <- coef(sch_lm)[2]
  }
}

# Partial pooling (multilevel) regression
a_part_pooled <- coef(M1)$school[, 1]
b_part_pooled <- coef(M1)$school[, 2]



# (1) Subset 8 of the schools; generate data frame
df <- data.frame(y, x, schid)
df8 <- subset(df, schid %in% sel.sch)

y <- GCSE$course
x <- as.numeric(GCSE$female) - 1 + runif(N, -.05, .05)
schid <- GCSE$school
sel.sch <- c("65385",
             "68207",
             "60729",
             "67051",
             "50631",
             "60427",
             "64321",
             "68137")

# (2) Assign complete-pooling, no-pooling, partial pooling estimates
df8$a_pooled <- a_pooled 
df8$b_pooled <- b_pooled
df8$a_nopooled <- a_nopooled[df8$schid]
df8$b_nopooled <- b_nopooled[df8$schid]
df8$a_part_pooled <- a_part_pooled[df8$schid]
df8$b_part_pooled <- b_part_pooled[df8$schid]

# (3) Plot regression fits for the 8 schools
ggplot(data = df8, 
       aes(x = x, y = y)) + 
  facet_wrap(facets = ~ schid, 
             ncol = 4) + 
  theme_bw() +
  geom_jitter(position = position_jitter(width = .05, 
                                         height = 0)) +
  geom_abline(aes(intercept = a_pooled, 
                  slope = b_pooled), 
              linetype = "solid", 
              color = "blue", 
              size = 0.5) +
  geom_abline(aes(intercept = a_nopooled, 
                  slope = b_nopooled), 
              linetype = "longdash", 
              color = "red", 
              size = 0.5) + 
  geom_abline(aes(intercept = a_part_pooled, 
                  slope = b_part_pooled), 
              linetype = "dotted", 
              color = "purple", 
              size = 0.7) + 
  scale_x_continuous(breaks = c(0, 1), 
                     labels = c("male", "female")) + 
  labs(title = "Complete-pooling (blue), No-pooling (red), and Partial pooling (purple) estimates",
       x = "", 
       y = "Total score on coursework paper")+theme_bw(base_family = "serif")

dev.copy(png,'pooling.png')
dev.off()

# ***********************************************************************************************
## Ranking plot
sims <- as.matrix(M1_stanlmer)
dim(sims)
para_name <- colnames(sims)
para_name

# Obtain school-level varying intercept a_j
# draws for overall mean
mu_a_sims <- as.matrix(M1_stanlmer, 
                       pars = "(Intercept)")
# draws for 73 schools' school-level error
u_sims <- as.matrix(M1_stanlmer, 
                    regex_pars = "b\\[\\(Intercept\\) school\\:")
# draws for 73 schools' varying intercepts               
a_sims <- as.numeric(mu_a_sims) + u_sims          

# Obtain sigma_y and sigma_alpha^2
# draws for sigma_y
s_y_sims <- as.matrix(M1_stanlmer, 
                      pars = "sigma")
# draws for sigma_alpha^2
s__alpha_sims <- as.matrix(M1_stanlmer, 
                           pars = "Sigma[school:(Intercept),(Intercept)]")

# Compute mean, SD, median, and 95% credible interval of varying intercepts

# Posterior mean and SD of each alpha
a_mean <- apply(X = a_sims,     # posterior mean
                MARGIN = 2,
                FUN = mean)
a_sd <- apply(X = a_sims,       # posterior SD
              MARGIN = 2,
              FUN = sd)

# Posterior median and 95% credible interval
a_quant <- apply(X = a_sims, 
                 MARGIN = 2, 
                 FUN = quantile, 
                 probs = c(0.025, 0.50, 0.975))
a_quant <- data.frame(t(a_quant))
names(a_quant) <- c("Q2.5", "Q50", "Q97.5")

# Combine summary statistics of posterior simulation draws
a_df <- data.frame(a_mean, a_sd, a_quant)
round(head(a_df), 2)

# Sort dataframe containing an estimated alpha's mean and sd for every school
a_df <- a_df[order(a_df$a_mean), ]
a_df$a_rank <- c(1 : dim(a_df)[1])  # a vector of school rank 

# Combine summary statistics of posterior simulation draws
#a_df <- data.frame(a_mean, a_sd, a_quant)
#round(head(a_df), 2)

ggplot(data = a_df, 
       aes(x = a_rank, 
           y = a_mean)) +
  geom_pointrange(aes(ymin = Q2.5, 
                      ymax = Q97.5),
                  position = position_jitter(width = 0.1, 
                                             height = 0)) + 
  geom_hline(yintercept = mean(a_df$a_mean), 
             size = 0.5, 
             col = "red") + 
  scale_x_continuous("Rank", 
                     breaks = seq(from = 0, 
                                  to = 80, 
                                  by = 5)) + 
  scale_y_continuous(expression(paste("varying intercept, ", alpha[j]))) + 
  theme_bw( base_family = "serif")

dev.copy(png,'ranking.png')
dev.off()

mean(a_df$a_mean)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Histogram for school differences
# The difference between the two school averages (school #21 and #51)
school_diff <- a_sims[, 21] - a_sims[, 51]

# Investigate differences of two distributions
mean <- mean(school_diff)
sd <- sd(school_diff)
quantile <- quantile(school_diff, probs = c(0.025, 0.50, 0.975))
quantile <- data.frame(t(quantile))
names(quantile) <- c("Q2.5", "Q50", "Q97.5")
diff_df <- data.frame(mean, sd, quantile)
round(diff_df, 2)

# Histogram of the differences
ggplot(data = data.frame(school_diff), 
       aes(x = school_diff)) + 
  geom_histogram(color = "black", 
                 fill = "gray", 
                 binwidth = 0.75) + 
  scale_x_continuous("Score diffence between two schools: #21, #51",
                     breaks = seq(from = -20, 
                                  to = 20, 
                                  by = 10)) + 
  geom_vline(xintercept = c(mean(school_diff),
                            quantile(school_diff, 
                                     probs = c(0.025, 0.975))),
             colour = "red", 
             linetype = "longdash") + 
  geom_text(aes(25, 200, label = "MEAN = 5.747"), 
            color = "black", 
            size = 4) + 
  geom_text(aes(25, 175, label = "SD = 6.094"), 
            color = "black", 
            size = 4) + 
  theme_bw( base_family = "serif") 

dev.copy(png,'differences.png')
dev.off()

# get count of how much school 21 is better than school 51
prop.table(table(a_sims[, 21] > a_sims[, 51]))

# 创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创创
## Robustness checks for the bayesian model

# Estimate a bayesian model with informative priors
M1_good <- stan_lmer(formula = course ~ female + (1 + female | school), 
                         data = GCSE, 
                         prior = normal(location = 0, 
                                        scale = 1,
                                        autoscale = FALSE),
                         prior_intercept = normal(location = 0, 
                                                  scale = 1, 
                                                  autoscale = FALSE),
                         seed = 349)


## Get mean, sd and credible intervals for good prior
sims_good <- as.matrix(M1_good)
dim(sims_good)
para_name_good <- colnames(sims_good)
para_name_good

# Obtain school-level varying intercept a_j
# draws for overall mean
mu_a_sims_good <- as.matrix(M1_good, 
                            pars = "(Intercept)")
# draws for 73 schools' school-level error
u_sims_good <- as.matrix(M1_good, 
                         regex_pars = "b\\[\\(Intercept\\) school\\:")
# draws for 73 schools' varying intercepts               
a_sims_good <- as.numeric(mu_a_sims_good) + u_sims_good         

# Obtain sigma_y and sigma_alpha^2
# draws for sigma_y
s_y_sims_good <- as.matrix(M1_good, 
                           pars = "sigma")
# draws for sigma_alpha^2
s__alpha_sims_good <- as.matrix(M1_good, 
                                pars = "Sigma[school:(Intercept),(Intercept)]")

# Compute mean, SD, median, and 95% credible interval of varying intercepts

# Posterior mean and SD of each alpha
a_mean_good <- apply(X = a_sims_good,     # posterior mean
                     MARGIN = 2,
                     FUN = mean)
a_sd_good <- apply(X = a_sims_good,       # posterior SD
                   MARGIN = 2,
                   FUN = sd)

# Posterior median and 95% credible interval
a_quant_good <- apply(X = a_sims_good, 
                      MARGIN = 2, 
                      FUN = quantile, 
                      probs = c(0.025, 0.50, 0.975))
a_quant_good <- data.frame(t(a_quant_good))
names(a_quant_good) <- c("Q2.5", "Q50", "Q97.5")

# Combine summary statistics of posterior simulation draws
a_df_good <- data.frame(a_mean_good, a_sd_good, a_quant_good)

# Sort dataframe containing an estimated alpha's mean and sd for every school
a_df_good <- a_df_good[order(a_df_good$a_mean_good), ]
a_df_good$a_rank_good <- c(1 : dim(a_df_good)[1])  # a vector of school rank 

colMeans(a_df_good)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# estimate model with weakly informative priors
M1_weak <- stan_lmer(formula = course ~ female + (1 + female | school), 
                     data = GCSE, 
                     prior = normal(location = 0, 
                                    scale = 10,
                                    autoscale = FALSE),
                     prior_intercept = normal(location = 0, 
                                              scale = 10, 
                                              autoscale = FALSE),
                     seed = 349)

## Get mean, sd and credible intervals for weak prior
sims_weak <- as.matrix(M1_weak)
dim(sims_weak)
para_name_weak <- colnames(sims_weak)
para_name_weak

# Obtain school-level varying intercept a_j
# draws for overall mean
mu_a_sims_weak <- as.matrix(M1_weak, 
                            pars = "(Intercept)")
# draws for 73 schools' school-level error
u_sims_weak <- as.matrix(M1_weak, 
                         regex_pars = "b\\[\\(Intercept\\) school\\:")
# draws for 73 schools' varying intercepts               
a_sims_weak <- as.numeric(mu_a_sims_weak) + u_sims_weak         

# Obtain sigma_y and sigma_alpha^2
# draws for sigma_y
s_y_sims_weak <- as.matrix(M1_weak, 
                           pars = "sigma")
# draws for sigma_alpha^2
s__alpha_sims_weak <- as.matrix(M1_weak, 
                                pars = "Sigma[school:(Intercept),(Intercept)]")

# Compute mean, SD, median, and 95% credible interval of varying intercepts

# Posterior mean and SD of each alpha
a_mean_weak <- apply(X = a_sims_weak,     # posterior mean
                     MARGIN = 2,
                     FUN = mean)
a_sd_weak <- apply(X = a_sims_weak,       # posterior SD
                   MARGIN = 2,
                   FUN = sd)

# Posterior median and 95% credible interval
a_quant_weak <- apply(X = a_sims_weak, 
                      MARGIN = 2, 
                      FUN = quantile, 
                      probs = c(0.025, 0.50, 0.975))
a_quant_weak <- data.frame(t(a_quant_weak))
names(a_quant_weak) <- c("Q2.5", "Q50", "Q97.5")

# Combine summary statistics of posterior simulation draws
a_df_weak <- data.frame(a_mean_weak, a_sd_weak, a_quant_weak)

# Sort dataframe containing an estimated alpha's mean and sd for every school
a_df_weak <- a_df_weak[order(a_df_weak$a_mean_weak), ]
a_df_weak$a_rank_weak <- c(1 : dim(a_df_weak)[1])  # a vector of school rank 

colMeans(a_df_weak)


#**************************************************************************************************

# estimate model with uninformative priors
M1_uninformative <- stan_lmer(formula = course ~ female + (1 + female | school), 
                              data = GCSE, 
                              prior = normal(location = 0, 
                                             scale = 100,
                                             autoscale = FALSE),
                              prior_intercept = normal(location = 0, 
                                                       scale = 100, 
                                                       autoscale = FALSE),
                              seed = 349)

## Get mean, sd and credible intervals for uninformative prior
sims_uninformative <- as.matrix(M1_uninformative)
dim(sims_uninformative)
para_name_uninformative <- colnames(sims_uninformative)
para_name_uninformative

# Obtain school-level varying intercept a_j
# draws for overall mean
mu_a_sims_uninformative <- as.matrix(M1_uninformative, 
                            pars = "(Intercept)")
# draws for 73 schools' school-level error
u_sims_uninformative <- as.matrix(M1_uninformative, 
                         regex_pars = "b\\[\\(Intercept\\) school\\:")
# draws for 73 schools' varying intercepts               
a_sims_uninformative <- as.numeric(mu_a_sims_uninformative) + u_sims_uninformative         

# Obtain sigma_y and sigma_alpha^2
# draws for sigma_y
s_y_sims_uninformative <- as.matrix(M1_uninformative, 
                           pars = "sigma")
# draws for sigma_alpha^2
s__alpha_sims_uninformative <- as.matrix(M1_uninformative, 
                                pars = "Sigma[school:(Intercept),(Intercept)]")

# Compute mean, SD, median, and 95% credible interval of varying intercepts

# Posterior mean and SD of each alpha
a_mean_uninformative <- apply(X = a_sims_uninformative,     # posterior mean
                     MARGIN = 2,
                     FUN = mean)
a_sd_uninformative <- apply(X = a_sims_uninformative,       # posterior SD
                   MARGIN = 2,
                   FUN = sd)

# Posterior median and 95% credible interval
a_quant_uninformative <- apply(X = a_sims_uninformative, 
                      MARGIN = 2, 
                      FUN = quantile, 
                      probs = c(0.025, 0.50, 0.975))
a_quant_uninformative <- data.frame(t(a_quant_uninformative))
names(a_quant_uninformative) <- c("Q2.5", "Q50", "Q97.5")

# Combine summary statistics of posterior simulation draws
a_df_uninformative <- data.frame(a_mean_uninformative, a_sd_uninformative, a_quant_uninformative)

# Sort dataframe containing an estimated alpha's mean and sd for every school
a_df_uninformative <- a_df_uninformative[order(a_df_uninformative$a_mean_uninformative), ]
a_df_uninformative$a_rank_uninformative <- c(1 : dim(a_df_uninformative)[1])  # a vector of school rank 

colMeans(a_df_uninformative)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

