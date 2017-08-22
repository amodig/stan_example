data {
  int<lower=0> N;
  vector[N] y;
  int<lower=0, upper=2> x[N];  // bernoulli outcome
  vector[N] count;
}
transformed data {
}
parameters {
  vector[2] mu;
  vector<lower=0, upper=10>[2] sigma;  // lognormal sigma
  vector<lower=0, upper=1>[2] theta;  // bernoulli prob
}
model {
  mu ~ normal(2, 10);

  for (n in 1:N) {
    if (y[n] > 0)
      target += count[n] * (bernoulli_lpmf(0 | theta[x[n]])
                        + lognormal_lpdf(y[n] | mu[x[n]], sigma[x[n]]));
    else
      target += count[n] * bernoulli_lpmf(1 | theta[x[n]]);
  }
}
generated quantities {
  real pctrl ;
  real test ;

  // mean of log-normal distribution
  pctrl = exp(mu[1] + 0.5 * sigma[1] * sigma[1]);
  test = exp(mu[2] + 0.5 * sigma[2] * sigma[2]);
}
