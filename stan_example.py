
# coding: utf-8

# In[ ]:


get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Arttu Modig' -v -m -d -t -p numpy,pandas,seaborn,matplotlib,pystan,notebook")


# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pystan
import seaborn as sns

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
os.environ["CXXFLAGS"] = "-O3 -mtune=native -march=native -Wno-unused-variable -Wno-unused-function"


# In[ ]:


def plot_AB_test_mean_posteriors(ctrl_mean, test_mean, ctrl_conversion, test_conversion,
                                 N, iter_total, title="Posterior samples of log-normal revenue mean",
                                 xlabel="revenue mean"):
    """Plot A/B test mean posteriors"""
    fig, ax = plt.subplots()
    ax.hist(test_mean, bins=100, alpha=0.85, label="test")
    ax.hist(ctrl_mean, bins=100, alpha=0.65, label="ctrl")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    # fine-tune
    y_max_orig = ax.get_ylim()[1]
    ax.set_ylim([0, 1.25 * y_max_orig])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_visible(False)
    # info text
    ax.text(0, 0.99, 'Test group conversion: {k1:0.1f} %\nControl group conversion: {k0:0.1f} %\n'
            'Test sample size: {N}\nMCMC iterations: {iter}'\
            .format(k1=100*test_conversion, k0=100*ctrl_conversion, N=N, iter=iterations_total),
            fontsize=11, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    # mean lines
    ax.axvline(x=test_mean.mean(), ymax=1/1.3, color='black', linestyle='dashed', linewidth=1)
    ax.axvline(x=ctrl_mean.mean(), ymax=1/1.3, color='black', linestyle='dashed', linewidth=1)
    height = ax.get_ylim()[1] - ax.get_ylim()[0]
    width = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.text(test_mean.mean() + 0.02 * width, 0.1 * height, '{0:.2f}'.format(test_mean.mean()),
            color='black', fontsize=11)
    ax.text(ctrl_mean.mean() + 0.02 * width, 0.1 * height, '{0:.2f}'.format(ctrl_mean.mean()),
            color='black', fontsize=11)


# In[ ]:


get_ipython().magic('env CC')


# In[ ]:


get_ipython().magic('pwd')


# In[ ]:


data = pd.read_csv("data_sales_grouped.csv")


# In[ ]:


data.head()


# ### Legacy model for A/B testing

# In[ ]:


get_ipython().system('cat legacy_updated.stan')


# In[ ]:


iterations = 10000
chains = 8
iterations_total = iterations * chains

model_data = {"N": data.shape[0],
              "y": data['revenue'].astype(float).values,
              "x": data['label'].astype(int).values,
              "count": data['count'].astype(int).values}

# fit model
fit_legacy = pystan.stan(file="legacy_updated.stan", model_name="legacy",
                  data=model_data, iter=iterations, chains=chains,
                  n_jobs=-1, verbose=True)

# extract results
theta = fit_legacy.extract()['theta']  # theta is no-purchase rate in legacy model
conversion = 1 - theta
ctrl_conversion = np.mean(conversion[:,0])
test_conversion = np.mean(conversion[:,1])
ctrl_revenue = fit.extract()['pctrl'] #* conversion_0
test_revenue = fit.extract()['test'] #* conversion_1

# plot results
plot_AB_test_mean_posteriors(ctrl_mean=ctrl_revenue, test_mean=test_revenue,
                             ctrl_conversion = ctrl_conversion,
                             test_conversion = test_conversion,
                             N=data['count'].sum(), iter_total=iterations_total)


# ### New model for A/B/C/D/etc. testing

# In[ ]:


get_ipython().system('cat hier2.stan')


# In[ ]:


iterations = 50000
chains = 8
iterations_total = iterations * chains

rev = data['revenue']
rev = rev[rev > 0]
mean_log_y = np.mean(np.log(rev))
sd_log_y = np.std(np.log(rev))

model_data = {"N": data.shape[0],
            "J": data['label'].nunique(),
            "y": data['revenue'].astype(float).values,
            "id": data['label'].astype(int).values,  # id should start from 1!
            "count": data['count'].astype(int).values,
            "mean_log_y": mean_log_y,
            "sd_log_y": sd_log_y}

sm = pystan.StanModel(file="hier2.stan", verbose=True)


# In[ ]:


fit_hier = sm.sampling(data=model_data, iter=iterations, chains=1, n_jobs=1, verbose=True,
                  control={'adapt_delta': 0.99, 'max_treedepth': 10})


# In[ ]:


revenue_mean = fit_hier.extract()['revenue_mean'] 
revenue_std = fit_hier.extract()['revenue_std']
conversion = fit_hier.extract()['theta']


# In[ ]:


plot_AB_test_mean_posteriors(ctrl_mean=revenue_mean[:,0], test_mean=revenue_mean[:,1],
                             ctrl_conversion = np.mean(conversion[:,0]),
                             test_conversion = np.mean(conversion[:,1]),
                             N=data['count'].sum(), iter_total=iterations_total)


# In[ ]:




