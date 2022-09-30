import numpy as np 
import pandas as pd 
import pymc3 as pm 
import matplotlib.pyplot as plt 
import arviz 
import pickle

bd = pd.read_csv('./numpyro/birthdays.txt')

lam_date_join = lambda row: f"{row['year']}-{row['month']}-{row['day']}"
bd["date"] = pd.to_datetime(bd.apply(lam_date_join, axis=1), format="%Y-%m-%d")

lam_get_day = lambda date: date.days

bd_sm = bd[bd["year"] <= 1970]
bd_sm['date_indexed'] = (bd_sm['date'] - bd_sm['date'].min()).apply(lam_get_day)
bd_sm['date_normed'] = (bd_sm['date_indexed'] - bd_sm['date_indexed'].mean()) / bd_sm['date_indexed'].std()
bd_sm['births_normed'] = (bd_sm['births'] - bd_sm['births'].mean()) / bd_sm['births'].std()


X = bd_sm['date_normed'].values[:300,None]
y = bd_sm['births_normed'].values[:300,None]

with pm.Model() as model:
    # GP1: long-term trends. Squared-exponential covariance function
    # parameter priors
    length_long = pm.Uniform('length_long',0,3)
    sigma_long = pm.HalfCauchy('sigma_long', 1)
    # kernel
    kernel_long = sigma_long**2 * pm.gp.cov.ExpQuad(1, length_long) # need to multiply by the variance
    # Gaussian process prior 1
    gp_1 = pm.gp.Marginal(cov_func=kernel_long) # mean 0 by default

    # GP2: short-term trends. Squared-exponential covariance function
    # parameter priors
    length_short = pm.Uniform('length_short',0,0.5) # smaller length scale
    sigma_short = pm.HalfCauchy('sigma_short', 2)
    # kernel
    kernel_short = sigma_short**2 * pm.gp.cov.ExpQuad(1, length_short)
    # Gaussian process prior 1
    gp_2 = pm.gp.Marginal(cov_func=kernel_short)

    # GP3: weekly trend squared exp * periodic
    #
    length_week = pm.Uniform('length_week',0,.5)
    ls_period_week = pm.Uniform('ls_period_week',0,0.5)
    period_week = pm.Uniform('period_week',0,1)
    sigma_week = pm.HalfCauchy('sigma_week', 2)
    kernel_week = sigma_week**2 * pm.gp.cov.ExpQuad(1, length_week) * pm.gp.cov.Periodic(1, period=period_week, ls=ls_period_week)
    gp_3 = pm.gp.Marginal(cov_func=kernel_week)

    # GP4: yearly trend
    length_year = pm.Uniform('length_year',0,.5)
    ls_period_year = pm.Uniform('ls_period_year',0,0.5)
    period_year = pm.Uniform('period_year',0,1)
    sigma_year = pm.HalfCauchy('sigma_year', 2)
    kernel_year = sigma_year**2 * pm.gp.cov.ExpQuad(1, length_year) * pm.gp.cov.Periodic(1, period=period_year, ls=ls_period_year)
    gp_4 = pm.gp.Marginal(cov_func=kernel_year)

    # observation noise
    sigma_noise = pm.HalfCauchy('sigma_noise', 2)

    # total Gaussian process
    gp = gp_1 + gp_2 + gp_3 + gp_4

    # marginal likelihood: where the observations come into play
    y_ = gp.marginal_likelihood('y_obs', X=X, y=y.squeeze(), noise=sigma_noise)



with model:
    mp = pm.find_MAP()
    trace = pm.sample(3000, start=mp, njobs=4)

summary(trace)

with open('trace.pkl','wb') as fout:
    pickle.dump((model,trace), fout)