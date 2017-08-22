= A/B/C/D/.../N testing with Stan

An example project of generalized A/B testing using Bayesian modeling with
Stan. Project includes implementations in PyStan and RStan.

The hierarchical model is based on the idea of partial pooling similar to model
number 3 here: http://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html

For performance reasons, the example sales data has been grouped
(i.e. transactions with the same revenue and test group are counted together).
