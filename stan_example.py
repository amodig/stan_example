
# coding: utf-8

# In[ ]:


# get_ipython().magic('matplotlib inline')
# get_ipython().magic('load_ext watermark')
# get_ipython().magic("watermark -a 'Arttu Modig' -v -m -d -t -p numpy,pandas,seaborn,matplotlib,pystan,notebook")


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pystan


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

fit = pystan.stan(file="legacy_updated.stan", model_name="legacy",
                  data=model_data, iter=iterations, chains=chains,
                  n_jobs=-1, verbose=True)

theta_0 = fit.extract()['theta'][:,0]
theta_1 = fit.extract()['theta'][:,1]
conversion_0 = 1 - theta_0
conversion_1 = 1 - theta_1
k0 = np.round(np.average(conversion_0) * 100, decimals=1)
k1 = np.round(np.average(conversion_1) * 100, decimals=1)
ctrl_revenue = fit.extract()['pctrl'] * conversion_0
test_revenue = fit.extract()['test'] * conversion_1

# collect results
results_all = {}
results_all['n'] = data['count'].sum()
results_all['test_revenue'] = test_revenue
results_all['ctrl_revenue'] = ctrl_revenue
results_all['k0'] = k0
results_all['k1'] = k1


# In[ ]:


fig, ax = plt.subplots()

ax.hist((test_revenue), bins=100, alpha=0.85, label="test")
ax.hist((ctrl_revenue), bins=100, alpha=0.65, label="ctrl")

ax.set_title("Posterior samples of log-normal mean")
plt.axvline(test_revenue.mean(), color='#FF8400', linestyle='dashed', linewidth=2)
plt.axvline(ctrl_revenue.mean(), color='#FF8400', linestyle='dashed', linewidth=2)
ax.set_yticklabels([])
ax.set_xlabel("revenue")
ax.legend()
x_axis = ax.get_xlim()[1] - ax.get_xlim()[0]
y_axis = ax.get_ylim()[1] - ax.get_ylim()[0]
width = plt.xlim()[0] + (x_axis * 0.02)
height = ax.get_ylim()[1]
font_height = y_axis * 0.04
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

ax.text(width, height - font_height*1.3, 'Testiryhmän konversio: {} %'.format(k1).replace('.', ','), fontsize=11)
ax.text(width, height - font_height*2.3, 'Kontrolliryhmän konversio: {} %'.format(k0).replace('.', ','), fontsize=11)
ax.text(width, height - font_height*3.3, 'Otantakoko = {0:,}'.format(data['count'].sum()).replace(',', ' '), fontsize=11)
ax.text(width, height - font_height*4.3, 'Iteraatioita = {0:,}'.format(iterations_total).replace(',', ' '), fontsize=11)
ax.text(test_revenue.mean()+0.02*x_axis, 0.05*height, '{0:.2f}'.format(test_revenue.mean()), color='#ff7700', fontsize=12)
ax.text(ctrl_revenue.mean()+0.02*x_axis, 0.05*height, '{0:.2f}'.format(ctrl_revenue.mean()), color='#ff7700', fontsize=12)


# ### New model for A/B/C/D/etc. testing

# In[ ]:


get_ipython().system('cat hier2.stan')


# In[ ]:


iterations = 10000
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


# WTF Y U NOT WORK??
fit = sm.sampling(data=model_data, iter=iterations, chains=1, n_jobs=1, verbose=True,
                  control={'adapt_delta': 0.99, 'max_treedepth': 10})
