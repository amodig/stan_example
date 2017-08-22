functions {
  real lognormal_mean(real mu, real sigma) {
    return exp(mu + 0.5 * sigma * sigma);
  }
  real lognormal_sd(real mu, real sigma) {
    real a;
    real b;
    a = exp(sigma * sigma) - 1;
    b = exp(2 * mu + sigma * sigma);
    return sqrt(a * b); // sqrt of variance
  }
}
data {
  // notice we use grouped data per outcome value
  int<lower=0> N; // sum of unique outcomes
  int<lower=0> J; // number of test groups
  real<lower=0> y[N]; // observed outcome i.e. revenue
  int<lower=1> id[N]; // label for each unique outcome
  int<lower=1> count[N]; // count of each unique outcome
  real<lower=0> mean_log_y; // mean of log-positive outcomes
  real<lower=0> sd_log_y; // sd of log-positive outcomes
}
parameters {
  vector<lower=0, upper=1>[J] theta; // chance of success per test group, i.e. conversion rates per test group
  real<lower=0, upper=1> phi; // overall population chance of success
  real<lower=1> kappa; // population concentration
  vector<lower=0>[J] mu; // mu for lognormal distribution per test group
  vector<lower=0>[J] sigma; // sigma for lognormal distribution per test group
  real<lower=0> mu_sigma; // scale for mu ~ Normal
  real<lower=0> sigma_sigma; // scale for sigma ~ Normal
}
model {
  // Priors for Bernoulli(theta):
  // phi ~ beta(1, 1); // hyperprior (uniform, could drop, change to beta(2,2)?)
  phi ~ beta(2, 2);
  kappa ~ pareto(1, 1.5); // hyperprior (requires that kappa > 1st Pareto parameter)
  theta ~ beta(kappa * phi, kappa * (1 - phi)); // prior

  // Empirical priors for Log-normal(mu, sigma):
  mu_sigma ~ student_t(4, 0, 1); // hyperprior; param #3 = mean of half-t-dist.
  sigma_sigma ~ student_t(4, 0, 0.05); // hyperprior
  mu ~ normal(mean_log_y, mu_sigma);
  sigma ~ normal(sd_log_y, sigma_sigma);

  // Likelihood log-probability increment:
  for (n in 1:N) {
    if (y[n] > 0) {
      target += count[n] * (bernoulli_lpmf(0 | theta[id[n]]) +
        lognormal_lpdf(y[n] | mu[id[n]], sigma[id[n]]));
    } else {
      target += count[n] * bernoulli_lpmf(1 | theta[id[n]]);
    }
  }
}
generated quantities {
  vector[J] revenue_mean;
  vector[J] revenue_std;

  for (j in 1:J) {
    revenue_mean[j] = lognormal_mean(mu[j], sigma[j]);
    revenue_std[j] = lognormal_sd(mu[j], sigma[j]);
  }
}
