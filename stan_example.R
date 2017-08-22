# ---- 0. Setup ----
library(tidyverse)
library(rstan)

# setwd("/Users/amodig/git/R/stan")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# ---- 1. Legacy analysis ----
data <- read.csv("data_sales_grouped.csv")

model_data <- list(
  N = nrow(data),
  y = data$revenue,
  x = data$label,
  count = data$count)

fit <- stan(file = "legacy_updated.stan",
            data = model_data, iter = 10000, chains = 8)

saveRDS(fit, "fitted_model_legacy.rds")

thetas <- extract(fit, pars='theta')$theta  # prob. of negative outcome
convs <- 1 - thetas  # conversions
mean_ctrl <- extract(fit, pars='pctrl')$pctrl
mean_test <- extract(fit, pars='test')$test

df1 <- data.frame(cond = factor(rep(c("ctrl","test"), each=nrow(convs))),
                  conv = c(convs[,1], convs[,2]),
                  mean = c(mean_ctrl, mean_test))

ggplot(df1, aes(x=mean, fill=cond)) + geom_density(alpha=.3) + ggtitle("Revenue mean v1")
ggplot(df1, aes(x=conv, fill=cond)) + geom_density(alpha=.3) + ggtitle("Conversion v1")

# ---- 2. Partially pooled analysis ----
rev <- data$revenue
rev <- rev[rev > 0]
mean_log_y <- mean(log(rev))
sd_log_y <- sd(log(rev))

model_data <- list(
  N = nrow(data),
  J = length(unique(data$label)),
  y = data$revenue,
  id = data$label,
  count = data$count,
  mean_log_y = mean_log_y,
  sd_log_y = sd_log_y)

fit <- stan(file = "hier2.stan",
            data = model_data, iter = 10000, chains = 8,
            control = list(adapt_delta = 0.99,  # default: 0.8
                           max_treedepth = 10))  # default: 10

saveRDS(fit, "fitted_model_hier.rds")

thetas <- extract(fit, pars='theta')$theta  # prob. of negative outcome
convs <- 1 - thetas  # conversions
means <- extract(fit, pars='revenue_mean')$revenue_mean
stds <- extract(fit, pars='revenue_std')$revenue_std
diff <- means[,1] - means[,2]

df2 <- data.frame(cond = factor(rep(c("ctrl","test"), each=nrow(means))),
                  mean = c(means[,1], means[,2]),
                  conv = c(convs[,1], convs[,2]))

ggplot(df2, aes(x=mean, fill=cond)) + geom_density(alpha=.3) + ggtitle("Revenue mean v2")
ggplot(df2, aes(x=conv, fill=cond)) + geom_density(alpha=.3) + ggtitle("Conversion v2")

# ---- 3. Comparison ------
df_compare <- data.frame(cond = factor(rep(c("ctrl_v1","test_v1","ctrl_v2","test_v2"), each=nrow(means))),
                         mean = c(mean_ctrl, mean_test, means[,1], means[,2]),
                         conv = c(df1$conv, df2$conv))

ggplot(df_compare, aes(x=mean, fill=cond)) + geom_density(alpha=.3) + ggtitle("Revenue mean compare")
ggplot(df_compare, aes(x=conv, fill=cond)) + geom_density(alpha=.3) + ggtitle("Conversion compare")
