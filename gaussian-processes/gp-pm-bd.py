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
# bd_sm['date_normed'] = (bd_sm['date_indexed'] - bd_sm['date_indexed'].mean()) / bd_sm['date_indexed'].std()
bd_sm['births_normed'] = (bd_sm['births'] - bd_sm['births'].mean()) / bd_sm['births'].std()

# date_scale = 1/bd_sm['date_indexed'].std() # one day corresps to date_scale normed days

bd_sm['date_normed'] = bd_sm['date_indexed']
# bd_sm['births_normed'] = bd_sm['births']


# add in the special days

# get the day of the week (used for some holidays)
n_weeks = int(np.ceil(bd_sm.shape[0] / 7))
bd_sm["weekdays"] = pd.Series(["Wed","Thr","Fri","Sat","Sun","Mon","Tue"] * n_weeks)

def find_holiday(df: pd.DataFrame, month: int, weekday: str, n: int):
    """
        Get indicator column for holiday which occurs on
        n-th weekday of month. (n=-1 for last)
    """
    indic = pd.Series(0, df.index) 
    hols = df.query(f"month == {month} and weekdays == '{weekday}'").groupby(by=['year']).nth(n=n)
    for j in range(hols.shape[0]): # for each one...
        j_date = hols.iloc[j]["date"] # ...get the date...
        indic[df["date"]==j_date] = 1 # ...and record it

    return indic

bd_sm["spd_newyears"] = ((bd_sm["month"] == 1) & (bd_sm["day"] == 1)).astype(int)
bd_sm["spd_val"] = ((bd_sm["month"] == 2) & (bd_sm["day"] == 14)).astype(int)
bd_sm["spd_leap"] = ((bd_sm["month"] == 2) & (bd_sm["day"] == 29)).astype(int)
bd_sm["spd_aprilf"] = ((bd_sm["month"] == 4) & (bd_sm["day"] == 1)).astype(int)
bd_sm["spd_memorial"] = find_holiday(bd_sm, month=5, weekday='Mon', n=-1) 
bd_sm["spd_indep"] = ((bd_sm["month"] == 7) & (bd_sm["day"] == 4)).astype(int)
bd_sm["spd_labor"] = find_holiday(bd_sm, month=9, weekday='Mon', n=0) 
bd_sm["spd_halloween"] = ((bd_sm["month"] == 10) & (bd_sm["day"] == 31)).astype(int)
bd_sm["spd_thanks"] = find_holiday(bd_sm, month=11, weekday='Thr', n=3)
bd_sm["spd_xmas"] = ((bd_sm["month"] == 12) & (bd_sm["day"] == 25)).astype(int)


# X = bd_sm['date_normed'].values[:300,None]
# y = bd_sm['births_normed'].values[:300,None]

X = bd_sm[['date_normed', 'spd_newyears', 'spd_val', 'spd_leap', 'spd_aprilf',
    'spd_memorial','spd_indep','spd_labor','spd_halloween','spd_thanks',
    'spd_xmas']].values[:,:]
y = bd_sm['births_normed'].values[:,None]

with pm.Model() as model:
    # GP1: long-term trends. Squared-exponential covariance function
    length_long = pm.Normal('length_long',365,1.)
    sigma_long = pm.HalfCauchy('sigma_long', 5)
    # kernel
    kernel_long = sigma_long**2 * pm.gp.cov.ExpQuad(1, length_long, active_dims=[0]) # need to multiply by the variance
    # Gaussian process prior 1
    f_1 = pm.gp.Marginal(cov_func=kernel_long) # mean 0 by default

    # GP2: short-term trends. Squared-exponential covariance function
    # parameter priors
    length_short = pm.Normal('length_short',10,.1) # smaller length scale
    sigma_short = pm.HalfCauchy('sigma_short', 5)
    # kernel
    kernel_short = sigma_short**2 * pm.gp.cov.ExpQuad(1, length_short, active_dims=[0])
    # Gaussian process prior 1
    f_2 = pm.gp.Marginal(cov_func=kernel_short)

    # GP3: weekly trend squared exp * periodic ("local periodic kernel")
    length_week = pm.Normal('length_week', 20, .5) # length scale of exp envelope
    ls_period_week = pm.Normal('ls_period_week', 2, .2) # 1/ls is amplitude of the periodic part
    sigma_week = pm.HalfCauchy('sigma_week', 5)
    kernel_week = sigma_week**2 * pm.gp.cov.ExpQuad(1, length_week, active_dims=[0]) \
        * pm.gp.cov.Periodic(1, period=7, ls=ls_period_week, active_dims=[0])
    f_3 = pm.gp.Marginal(cov_func=kernel_week)

    # GP4: yearly trend
    length_year = pm.Normal('length_year',1000,100)
    ls_period_year = pm.Normal('ls_period_year',100,10)
    sigma_year = pm.HalfCauchy('sigma_year', 5)
    kernel_year = sigma_year**2 * pm.gp.cov.ExpQuad(1, length_year, active_dims=[0]) \
        * pm.gp.cov.Periodic(1, period=365.25, ls=ls_period_year, active_dims=[0])
    f_4 = pm.gp.Marginal(cov_func=kernel_year)


    # GP5: special days (weekend not implemented)
    # there are 10 special days, so 10 indicator variables for each sample => 10 parameters needed (coefficients)
    
    # beta = pm.Normal('beta_1',0, 3.,)
    # k1 = beta * pm.gp.cov.Linear(1,c=0, active_dims=[1]) # remaining columns of X
    # f_5 = pm.gp.Marginal(cov_func=k1)

    sigma_spd = pm.HalfCauchy('sigma_spec', 5)
    k_5 = sigma_spd**2 * pm.gp.cov.Linear(
        input_dim=11, # total number of columns of `X`
        c=0,
        active_dims=[1,2,3,4,5,6,7,8,9,10], # which columns we will act on
        )
    f_5 = pm.gp.Marginal(cov_func=k_5)


    # observation noise
    sigma_noise = pm.HalfCauchy('sigma_noise', .1)

    # total Gaussian process
    gp = f_1 + f_2 + f_3 + f_4 + f_5

    # marginal likelihood: where the observations come into play
    y_ = gp.marginal_likelihood('y_obs', X=X, y=y.squeeze(), noise=sigma_noise, is_observed=True)




# sample from the model posteriors to get posterior distributions ("plausibility") for the kernel parameters
with model:
    # mp = pm.find_MAP()
    trace = pm.sample(draws=2000, tune=1800) 
    # advi_fit = pm.fit(method='svgd', n=2000, obj_optimizer=pm.adagrad(learning_rate=1e-1))

with open('model_trace.pkl', 'wb') as fout:
    pickle.dump((model,trace), fout)