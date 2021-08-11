
from dataset import ApAvDataset, significance_test

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing

from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets

import arviz as az

import jax
import jax.numpy as jnp

import numpyro
import numpyro.infer
# import render
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam, TransformReparam
from numpyro.contrib.control_flow import scan

import pickle

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import plotly.express as px


# from models import latent_utility_model_nostim, latent_utility_model_stim, regularized_random_walk_model, constant_model, drift_perseverance_model#, generate_model
from models import fit
from models import generate_onpolicy_model

# %%
with open('ApAvDataset_behavior.pkl', 'rb') as f:
    dataset = pickle.load(f)


# %%
all_data = dataset.get_data(monkey_id=2, full_sessions_only=True)
all_data = significance_test(all_data)


# %%



# %%
dates = [datum['metadata']['datetime'] for datum in all_data]


# %%
# data = [datum for datum in all_data if (datum['metadata']['significant_diff'] and datum['metadata']['stim_increases_avoidance'])][0:NUM_SESSIONS]
data = [datum for datum in all_data if (datum['metadata']['significant_diff'] and datum['metadata']['stim_increases_avoidance'])]#[0:NUM_SESSIONS]
NUM_SESSIONS=len(data)
# %%
session_lengths = [sum(datum['metadata']['datanums'].values()) for datum in data]

# %%
concatenated_data = pd.concat([pd.concat((data[i]['stim0'], data[i]['stim1'], data[i]['resid'])) for i in range(len(data))])
data_sep = [pd.concat((data[i]['stim0'], data[i]['stim1'], data[i]['resid'])) for i in range(len(data))]
stage = np.concatenate([np.array([0] * len(data[i]['stim0']) + [1] * len(data[i]['stim1']) + [2] * len(data[i]['resid'])) for i in range(len(data))], 0)
stage_sep = [np.array([0] * len(data[i]['stim0']) + [1] * len(data[i]['stim1']) + [2] * len(data[i]['resid'])) for i in range(len(data))]

stim_indicator = np.concatenate([np.array([0] * len(data[i]['stim0']) + [1] * len(data[i]['stim1']) + [0] * len(data[i]['resid'])) for i in range(len(data))], 0)
stim_indicator_sep = [np.array([0] * len(data[i]['stim0']) + [1] * len(data[i]['stim1']) + [0] * len(data[i]['resid'])) for i in range(len(data))]
# two_state_data = pd.concat((data[0]['stim0'], data[0]['stim1'], data[0]['resid']))
# two_state_data['stim1nostim0'] = [0] * len(data[0]['stim0']) + [1] * len(data[0]['stim1']) + [0] * len(data[0]['resid'])
# concatenated_data['num_stim1_trials'] = [0] * len(data[0]['stim0']) + list(range(1, len(data[0]['stim1'])+1)) + [len(data[0]['stim1'])] * len(data[0]['resid'])
# concatenated_data['num_resid_trials'] = [0] * len(data[0]['stim0']) + [0] * len(data[0]['stim1']) + list(range(1, len(data[0]['resid'])+1))

scaler = preprocessing.StandardScaler().fit(concatenated_data[['reward_amount', 'aversi_amount']])
# concatenated_data[['reward_amount', 'aversi_amount']] *= 0.01
# scaler.mean_
concatenated_data[['reward_amount', 'aversi_amount']] = scaler.transform(concatenated_data[['reward_amount', 'aversi_amount']])
for datum in data_sep:
    datum[['reward_amount', 'aversi_amount']] = scaler.transform(datum[['reward_amount', 'aversi_amount']])# for datum in data_sep
# print(data_sep)
# X_data = np.array(concatenated_data[['reward_amount', 'aversi_amount', 'num_stim1_trials', 'num_resid_trials']])
X = np.array(concatenated_data[['reward_amount', 'aversi_amount']])
X_sep = [np.array(datum[['reward_amount', 'aversi_amount']]) for datum in data_sep]
# stim_durations = np.array(concatenated_data[['num_stim1_trials', 'num_resid_trials']])
# X_scaled = scaler.transform(X)
y = np.array(concatenated_data[['appro1_avoid0']]).flatten().astype(int)
y_sep = [np.array(datum[['appro1_avoid0']]).flatten().astype(int) for datum in data_sep]
def switchpoints(stage):
    array = np.zeros_like(stage)
    array[np.argmax(stage==1)] = 1
    array[np.argmax(stage==2)] = 1
    return array
    
switch_sep = [switchpoints(stage) for stage in stage_sep]
# num_stim = np.array(concatenated_data['num_stim1_trials'])
# num_resid = np.array(concatenated_data['num_resid_trials'])
# num_trials_per_stage = [len(data[0]['stim0']), len(data[0]['stim1']), len(data[0]['resid'])]
# y_prev = np.roll(y, 1)
# y_prev[0] = 0
# y_prev_indicator = y_prev.astype(int)
# y_prev_indicator[y_prev_indicator==0] = -1
# y_prev_indicator[0] = 0
y_prev_indicator_sep = []
y_prev_sep = []
for _y in y_sep:
    _y_prev = np.roll(_y, 1)
    _y_prev[0] = 0
    _y_prev_indicator = _y_prev.astype(int)
    y_prev_sep += [_y_prev.astype(int)]
    _y_prev_indicator[_y_prev_indicator==0] = -1
    _y_prev_indicator[0] = 0
    y_prev_indicator_sep += [_y_prev_indicator]
print(X.shape)
# print(X_data.shape)
print(y.shape)
# print(num_stim.shape)
# print(num_trials_per_stage)
# repetition_indicator_sep = []
# for _y_prev, _y in zip(y_prev_sep, y_sep):
#     rep_0_2 = _y_prev + _y
    


drift_scale = 0.1
repetition_kernel_scale = 10
stimulation_immediate_scale = 1
rw_scale = 0


all_results_ema_model = []
for i, (y_, y_prev_indicator_, stage_, X_) in enumerate(zip(y_sep, y_prev_indicator_sep, stage_sep, X_sep)):
#     if i != 4:
#         continue
    results = {}
    model = generate_onpolicy_model(drift_scale=drift_scale, repetition_kernel_scale=repetition_kernel_scale, stimulation_immediate_scale=stimulation_immediate_scale, random_walk_scale=rw_scale, saturating_ema=False, log_diminishing_perseverance=True, lapse=True, switch_scale=0.1)
    mcmc, idata = fit(model, 1, X=X_, stage=stage_, y=y_)
#     display(az.summary(idata, round_to=3, var_names=['~stimulated_weights', '~autocorr', '~stimulated_weights_base', '~logits']))
    results[f'ema_model | drift: {drift_scale}, rep: {repetition_kernel_scale}, stim_IE: {stimulation_immediate_scale}, RW: {rw_scale}'] = mcmc
    all_results_ema_model += [results]
#     break


for i, result in enumerate(all_results_ema_model):
    
    mcmc = list(result.values())[0]
    idata = az.from_numpyro(mcmc)
    mean_pred = np.array(np.mean(idata.posterior.probs_with_lapse, axis=(0,1)))
    std_pred = np.array(np.std(idata.posterior.probs_with_lapse, axis=(0,1)))
#     loo = az.loo(idata, pointwise=True)
    loo = compute_reloo(model, mcmc, X=X_sep[i], stage=stage_sep[i])
    fig = go.Figure(go.Scatter(
                x=np.arange(len(loo.pareto_k)), 
                y=loo.pareto_k,
                text=[f'reward: {d[0]}, aversi: {d[1]}, decision: {d[2]}, pred: {d[3]}, std: {d[4]}' for d in list(zip(X_sep[i][:,0], X_sep[i][:,1], y_sep[i], mean_pred, std_pred))],
                mode='markers'
            ))
    fig.add_trace(go.Scatter(
        x = np.arange(len(loo.pareto_k)),
        y = np.ones(len(loo.pareto_k))*0.7,
        mode='lines'
    ))
    fig.show()
