
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
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam, TransformReparam
from numpyro.contrib.control_flow import scan

import dill as pickle

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import plotly.express as px


from models import fit
from models import generate_hierarchical_model
from lfo_cv import compute_reloo

import os
import argparse

os.environ["PYTHONHASHSEED"] = "0"


with open('ApAvDataset_behavior.pkl', 'rb') as f:
    dataset = pickle.load(f)

numpyro.set_host_device_count(12)

# %%
all_data = dataset.get_data(monkey_id=2, full_sessions_only=True)
# all_data = significance_test(all_data)
# %%



# %%
#str_dates = list(map(str, dates))
#idx = str_dates.index('2011-04-04 16:44:35')
#print(idx)
# %%
data = [datum for datum in all_data if (datum['metadata']['significant_diff'] and datum['metadata']['stim_increases_avoidance'])]
data = data[:20]
dates = [datum['metadata']['datetime'] for datum in data]
str_dates = list(map(str, dates))

# data = [datum for datum in all_data if (datum['metadata']['significant_diff'] and datum['metadata']['stim_increases_avoidance'])]#[0:NUM_SESSIONS]
#data = all_data[idx:idx+1]
NUM_SESSIONS=len(data)
# assert NUM_SESSIONS ==
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



base_config = {
    "volatility": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 0.1},
    },
    "volatility_hyper_scale": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 1},
    },
    "repetition_kernel": {
        "shape": (3, 2),
        "dist_type": dist.Normal,
        "params": {"loc": 0.0, "scale": 10},
    },
    "repetition_kernel_hyper_mean": {
        "shape": (3, 2),
        "dist_type": dist.Normal,
        "params": {"loc": 0.0, "scale": 10},
    },
    "repetition_kernel_hyper_scale": {
        "shape": (3, 2),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "drift": {
        "shape": (3, 3),
        "dist_type": dist.Normal,
        "params": {"loc": 0.0, "scale": 0.1},
    },
    "drift_hyper_mean": {
        "shape": (3, 3),
        "dist_type": dist.Normal,
        "params": {"loc": 0.0, "scale": 0.1},
    },
    "drift_hyper_scale": {
        "shape": (3, 3),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 0.1},
    },
    "perseverance_growth_rate": {
        "shape": (3, 2),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "perseverance_growth_rate_hyper_scale": {
        "shape": (3, 2),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "forget_rate": {
        "shape": (2,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "forget_rate_hyper": {
        "shape": (2,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "mean_reversion_hyper": {
        "shape": (2,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "mean_reversion": {
        "shape": (3,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "lapse_prob": {
        "shape": (3,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "approach_given_lapse": {
        "shape": (3,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "switch_scale": 1.0,
    "saturating_ema": False,
    "poisson_cdf_diminishing_perseverance": True
}


def run(config, name='', reuse_models=True, smoke_test=False):

#     config_hash = abs(hash(str(config)))
#     use_saved_mcmc = False

    if not os.path.exists(f"output/{name}/pareto_ks"):
        os.makedirs(f"output/{name}/pareto_ks")
    # %%
#     if not os.path.exists(f"output/{name}/mcmcs"):
#         os.makedirs(f"output/{name}/mcmcs")

#     if not os.path.exists(f"output/{name}/idatas"):
#         os.makedirs(f"output/{name}/idatas")

    if not os.path.exists(f"output/{name}/loos"):
        os.makedirs(f"output/{name}/loos")


    with open(f'output/{name}/configdict.pkl', 'wb') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

    with open(f'output/dates.pkl', 'wb') as f:
        pickle.dump(dates, f, pickle.HIGHEST_PROTOCOL)

    hierarchical_model = generate_hierarchical_model(base_config)
    params_with_scale = ['repetition_kernel', 'drift']
    reparam_config = {key: LocScaleReparam(0) for key in [f"{date}_{param}" for date in str_dates for param in params_with_scale]}
    hierarchical_model = reparam(hierarchical_model, config=reparam_config)

    if smoke_test:
        mcmc, idata = fit(hierarchical_model, 8, X_sep=X_sep[:2], y_sep=y_sep[:2], stage_sep=stage_sep[:2], str_dates=str_dates[:2])
    else:
        mcmc, idata = fit(hierarchical_model, 8, X_sep=X_sep, y_sep=y_sep, stage_sep=stage_sep, str_dates=str_dates)

    az.to_netcdf(idata, f'output/{name}/idata.nc')
    with open(f'output/{name}/mcmc.pkl', 'wb') as f:
        pickle.dump(mcmc, f, pickle.HIGHEST_PROTOCOL)


#     for i, mcmc in enumerate(all_results_ema_model):

#         idata = az.from_numpyro(mcmc)
#         mean_pred = np.array(np.mean(idata.posterior.probs_with_lapse, axis=(0,1)))
#         std_pred = np.array(np.std(idata.posterior.probs_with_lapse, axis=(0,1)))
#     #     loo = az.loo(idata, pointwise=True)
#         loo = compute_reloo(model, mcmc, X=X_sep[i], stage=stage_sep[i])
#         fig = go.Figure(go.Scatter(
#                     x=np.arange(len(loo.pareto_k)),
#                     y=loo.pareto_k,
#                     text=[f'reward: {d[0]}, aversi: {d[1]}, decision: {d[2]}, pred: {d[3]}, std: {d[4]}' for d in list(zip(X_sep[i][:,0], X_sep[i][:,1], y_sep[i], mean_pred, std_pred))],
#                     mode='markers'
#                 ))
#         fig.add_trace(go.Scatter(
#             x = np.arange(len(loo.pareto_k)),
#             y = np.ones(len(loo.pareto_k))*0.7,
#             mode='lines'
#         ))
#     #     fig.show()
#         loo.to_csv(f'output/{name}/loos/session{dates[i]}.csv')

#         fig.write_image(f'output/{name}/pareto_ks/session{dates[i]}.svg')
#         if smoke_test:
#             break

if __name__ == "__main__":
    import copy

    parser = argparse.ArgumentParser()
    parser.add_argument('--smoketest', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.smoketest:
        print('Running smoketest!')
    #
    config = copy.deepcopy(base_config)
    # config['poisson_cdf_diminishing_perseverance'] = True
    # config['perseverance_growth_rate']['shape'] = (1,)
    # config['repetition_kernel']['shape'] = (1,)
    # config['forget_rate']['shape'] = (1,)
    # config['switch_scale'] = 0.0
    run(config, 'baseline_standard', smoke_test=args.smoketest)
    #
    # config = copy.deepcopy(base_config)
    # config['poisson_cdf_diminishing_perseverance'] = False
    # config['perseverance_growth_rate']['shape'] = ()
    # config['repetition_kernel']['shape'] = ()
    # config['forget_rate']['shape'] = ()
    # config['switch_scale'] = 0.0
    # run(config, 'baseline_no_repetition', smoke_test=args.smoketest)

#     config = copy.deepcopy(base_config)
#     config['poisson_cdf_diminishing_perseverance'] = True
#     config['perseverance_growth_rate']['shape'] = (2,)
#     config['repetition_kernel']['shape'] = (2,)
#     config['forget_rate']['shape'] = (2,)
#     config['switch_scale'] = 0.0
#     run(config, 'repetition_posneg', smoke_test=args.smoketest)
#     #
#     config = copy.deepcopy(base_config)
#     config['poisson_cdf_diminishing_perseverance'] = True
#     config['perseverance_growth_rate']['shape'] = (3,)
#     config['repetition_kernel']['shape'] = (3,)
#     config['forget_rate']['shape'] = (3,)
#     config['switch_scale'] = 0.0
#     run(config, 'repetition_stimdependent', smoke_test=args.smoketest)

#     config = copy.deepcopy(base_config)
#     config['poisson_cdf_diminishing_perseverance'] = True
#     config['perseverance_growth_rate']['shape'] = (3,2)
#     config['repetition_kernel']['shape'] = (3,2)
#     config['forget_rate']['shape'] = (3,2)
#     config['switch_scale'] = 0.0
#     run(config, 'repetition_posneg_stimdependent', smoke_test=args.smoketest)

    # config = copy.deepcopy(base_config)
    # config['perseverance_growth_rate']['shape'] = (2,)
    # run(config, 'pgr2', smoke_test=args.smoketest)
    # config = copy.deepcopy(base_config)
    # config['perseverance_growth_rate']['shape'] = (3,)
    # run(config, 'pgr3', smoke_test=args.smoketest)
    # config = copy.deepcopy(base_config)
    # config['perseverance_growth_rate']['shape'] = (3,2)
    # run(config, 'pgr32', smoke_test=args.smoketest)

    # config = copy.deepcopy(base_config)
    # config['perseverance_growth_rate']['shape'] = (2,)
    # config['repetition_kernel']['shape'] = (1,)
    # run(config, 'repkernel1_pgr2', smoke_test=args.smoketest)
    # config = copy.deepcopy(base_config)
    # config['perseverance_growth_rate']['shape'] = (2,)
    # config['repetition_kernel']['shape'] = ()
    # run(config, 'repkernel0_pgr2', smoke_test=args.smoketest)
    # config = copy.deepcopy(base_config)
    # config['perseverance_growth_rate']['shape'] = (2,)
    # config['repetition_kernel']['shape'] = (2,)
    # run(config, 'repkernel2_pgr2', smoke_test=args.smoketest)
    # config = copy.deepcopy(base_config)
    # config['perseverance_growth_rate']['shape'] = (2,)
    # config['repetition_kernel']['shape'] = (3,)
    # run(config, 'repkernel3_pgr2', smoke_test=args.smoketest)
    # config = copy.deepcopy(base_config)
    # config['perseverance_growth_rate']['shape'] = (2,)
    # config['repetition_kernel']['shape'] = (3,2)
    # run(config, 'repkernel32_pgr2', smoke_test=args.smoketest)
