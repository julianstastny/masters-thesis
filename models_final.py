import jax
import jax.numpy as jnp

import numpyro
import numpyro.infer

# import render
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.handlers import reparam
from numpyro.infer.reparam import LocScaleReparam, TransformReparam
from numpyro.contrib.control_flow import scan, cond
from numpyro.distributions.util import clamp_probs

from jax.scipy.special import betainc
from jax.scipy.special import gammaincc

import arviz as az

import numpy as np

HN_SIGMA = 1.0 # To make mean=1 for HalfNormal



def fit(model, num_chains, num_warmup=1000, num_samples=1000, rng_seed=0, **kwargs):
    #     assert set(kwargs.keys()) <= set(model.__code__.co_varnames), model.__code__.co_varnames
    #     assert (('X' in kwargs.keys() or 'Xs' in kwargs.keys()) and ('y' in kwargs.keys()))
    nuts_kernel = NUTS(model, adapt_step_size=True, init_strategy=numpyro.infer.init_to_sample)
    mcmc = MCMC(
        nuts_kernel,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
    rng_key = jax.random.PRNGKey(rng_seed)
#     with numpyro.validation_enabled():
    mcmc.run(rng_key, **kwargs)
    return mcmc, az.from_numpyro(mcmc)


def poisson_cdf(value, rate):
    #     k = jnp.floor(value) + 1
    k = value + 1
    return gammaincc(k, rate)




def numpyro_sample(name, dist_cls, ind_shape, target_shape, **kwargs):
    return jnp.resize(
        numpyro.sample(
            name, dist_cls(**kwargs).expand(ind_shape).to_event(len(ind_shape))
        ),
        target_shape,
    )


def maybe_numpyro_sample(name, dist_cls, ind_shape, target_shape, **kwargs):
    if not ind_shape:
        return jnp.zeros(target_shape)
    return numpyro_sample(name, dist_cls, ind_shape, target_shape, **kwargs)


def _numpyro_config_sample(name, config, target_shape, date=None, **kwargs):
    if date is not None:
        final_name = f"{date}_{name}"
    else:
        final_name = name
    dist_cls = config[name]["dist_type"]
    ind_shape = config[name]["shape"]
    if not kwargs:
        return maybe_numpyro_sample(
            final_name, dist_cls, ind_shape, target_shape, **config[name]["params"]
        )
    return maybe_numpyro_sample(
            final_name, dist_cls, ind_shape, target_shape, **kwargs
        )

def generate_switch_indicator(stage):
    array = np.zeros_like(stage)
    array[np.argmax(stage == 1)] = 1
    array[np.argmax(stage == 2)] = 1
    return array

config_descriptive = {
    "volatility": {
        "shape": (),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 0.1},
    },
    "repetition_kernel": {
        "shape": (),
        "dist_type": dist.Normal,
        "params": {"loc": 0.0, "scale": 10},
    },
    "drift": {
        "shape": (3, 3),
        "dist_type": dist.Normal,
        "params": {"loc": 0.0, "scale": 0.1},
    },
    "perseverance_growth_rate": {
        "shape": (),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "forget_rate": {
        "shape": (),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "mean_reversion": {
        "shape": (3,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "lapse_prob": {
        "shape": (1,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "approach_given_lapse": {
        "shape": (1,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "switch_scale": 0.0,
    "saturating_ema": False,
    "poisson_cdf_diminishing_perseverance": True
}

config_descriptive_perseveration = {
    "volatility": {
        "shape": (),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 0.1},
    },
    "repetition_kernel": {
        "shape": (3, 2),
        "dist_type": dist.Normal,
        "params": {"loc": 0.0, "scale": 10},
    },
    "drift": {
        "shape": (3, 3),
        "dist_type": dist.Normal,
        "params": {"loc": 0.0, "scale": 0.1},
    },
    "perseverance_growth_rate": {
        "shape": (3, 2),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "forget_rate": {
        "shape": (2,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "mean_reversion": {
        "shape": (3,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "lapse_prob": {
        "shape": (1,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "approach_given_lapse": {
        "shape": (1,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "switch_scale": 0.0,
    "saturating_ema": False,
    "poisson_cdf_diminishing_perseverance": True
}

def generate_descriptive_model(config):
    assert not (config['saturating_ema'] and config['poisson_cdf_diminishing_perseverance'])
    switch_scale = config['switch_scale']
    saturating_ema = int(config['saturating_ema'])

    def numpyro_config_sample(name, target_shape):
        return _numpyro_config_sample(name, config, target_shape)

    def ema(ema_prev, value, alpha):
        return (1 - alpha) * ema_prev + alpha ** saturating_ema * value

    if config['poisson_cdf_diminishing_perseverance']:

        def process_ema(x, rate):
            return poisson_cdf(x, rate)

    else:

        def process_ema(x, *args):
            return x

    def model(X, stage, y=None, str_date=None):
        volatility = numpyro_config_sample("volatility", target_shape=3)
        with numpyro.handlers.reparam(config={"initial_weight": TransformReparam()}):
            init_weight = numpyro.sample(
                "initial_weight",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(3), 1).to_event(1),
                    dist.transforms.AffineTransform(
                        0,
                        1.0,
                    ),
                ),
            )
#         init_weight = numpyro.sample("initial_weight", dist.Normal(jnp.zeros(3), 1).to_event(1))
        repetition_kernel = numpyro_config_sample(
            "repetition_kernel", target_shape=(3, 2)
        )
        perseverance_growth_rate = numpyro_config_sample(
            "perseverance_growth_rate",
            target_shape=(3, 2),
        )
        drift = numpyro_config_sample(
            "drift",
            target_shape=(3, 3),
        )
        switch_indicator = generate_switch_indicator(stage)
        forget_rate = numpyro_config_sample(
            "forget_rate",
            target_shape=(3,2),
        )
        lapse_prob = numpyro_config_sample(
            "lapse_prob",
            target_shape=(3,),
        )
        approach_given_lapse = numpyro_config_sample(
            "approach_given_lapse",
            target_shape=(3,),
        )
        mean_reversion = numpyro_config_sample(
            "mean_reversion",
            target_shape=3,
        )
        if y is not None:
            for i in np.flatnonzero(y == -1):
                y = jax.ops.index_update(y, i, numpyro.sample(f"y_null_{i}", dist.Bernoulli(0.5).mask(False)))

        def transition(carry, xs):
            weight_curr, ema_pos_prev, ema_neg_prev, y_prev = carry
            stage_curr, switch, x_curr, y_curr = xs 
            weight_curr = numpyro.deterministic("AR(1)", weight_curr)
            weights_with_offset = numpyro.deterministic(
                "stimulated_weights", init_weight + weight_curr
            )
            ema_pos_curr = numpyro.deterministic(
                "ema_positive", ema(ema_pos_prev, y_prev == 1, forget_rate[stage_curr, 0])
            )
            ema_neg_curr = numpyro.deterministic(
                "ema_negative", ema(ema_pos_prev, y_prev == 0, forget_rate[stage_curr, 1])
            )
            lema_pos_curr = process_ema(
                ema_pos_curr, perseverance_growth_rate[stage_curr, 0]
            )
            lema_neg_curr = process_ema(
                ema_neg_curr, perseverance_growth_rate[stage_curr, 1]
            )
            autocorr = numpyro.deterministic(
                "autocorr",
                lema_pos_curr * repetition_kernel[stage_curr, 0]
                + lema_neg_curr * repetition_kernel[stage_curr, 1],
            )
            logit = numpyro.deterministic(
                "logits",
                x_curr[0] * weights_with_offset[0]
                + x_curr[1] * weights_with_offset[1]
                + weights_with_offset[2]
                + autocorr,
            )
            prob_with_lapse = numpyro.deterministic(
                "probs_with_lapse",
                (1 - lapse_prob[stage_curr]) * jax.nn.sigmoid(logit)
                + lapse_prob[stage_curr] * approach_given_lapse[stage_curr],
            )
            obs = numpyro.sample(
                "y", dist.Bernoulli(probs=clamp_probs(prob_with_lapse)), obs=y_curr
            )
            weight_next_base = 0 #numpyro.sample("AR(1)_next_base", dist.Normal(jnp.zeros(3), 1).to_event(1))
            weight_next = (1 - mean_reversion[stage_curr]) * weight_curr + weight_next_base * volatility + drift[stage_curr]
            return (weight_next, ema_pos_curr, ema_neg_curr, obs), (obs)

        _, (obs) = scan(
            transition,
            (jnp.zeros(3), 0.5, 0.5, -1),
            (stage, switch_indicator, X, y),
            length=len(X),
        )

    return model

config_descriptive_perseveration_alt = {
    "volatility": {
        "shape": (),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 0.1},
    },
    "drift": {
        "shape": (3, 3),
        "dist_type": dist.Normal,
        "params": {"loc": 0.0, "scale": 0.1},
    },
    "perseverance_growth_rate": {
        "shape": (3, 2),
        "dist_type": dist.Normal,
        "params": {"loc": 0.0, "scale": 1.0},
    },
    "forget_rate": {
        "shape": (3,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
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
        "shape": (1,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "switch_scale": 0.0,
    "saturating_ema": False,
    "poisson_cdf_diminishing_perseverance": True
}



def generate_descriptive_model_alt(config):
    assert not (config['saturating_ema'] and config['poisson_cdf_diminishing_perseverance'])
    switch_scale = config['switch_scale']
    saturating_ema = int(config['saturating_ema'])

    def numpyro_config_sample(name, target_shape):
        return _numpyro_config_sample(name, config, target_shape)

    def ema(ema_prev, value, pgr, rate):
        return poisson_cdf(jnp.abs(ema_prev), rate) * ema_prev + pgr[0] * value + pgr[1] * (1 - value)

    def model(X, stage, y=None, str_date=None):
        volatility = numpyro_config_sample("volatility", target_shape=3)
        with numpyro.handlers.reparam(config={"initial_weight": TransformReparam()}):
            init_weight = numpyro.sample(
                "initial_weight",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(3), 1).to_event(1),
                    dist.transforms.AffineTransform(
                        0,
                        1.0,
                    ),
                ),
            )
#         init_weight = numpyro.sample("initial_weight", dist.Normal(jnp.zeros(3), 1).to_event(1))
#         repetition_kernel = numpyro_config_sample(
#             "repetition_kernel", target_shape=(3, 2)
#         )
        perseverance_growth_rate = numpyro_config_sample(
            "perseverance_growth_rate",
            target_shape=(3, 2),
        )
        drift = numpyro_config_sample(
            "drift",
            target_shape=(3, 3),
        )
        switch_indicator = generate_switch_indicator(stage)
#         forget_rate = numpyro_config_sample(
#             "forget_rate",
#             target_shape=(3,),
#         )
        lapse_prob = numpyro_config_sample(
            "lapse_prob",
            target_shape=(3,),
        )
#         approach_given_lapse = numpyro_config_sample(
#             "approach_given_lapse",
#             target_shape=(3,),
#         )
        mean_reversion = numpyro_config_sample(
            "mean_reversion",
            target_shape=3,
        )
        mean_reversion_growth_rate = numpyro.sample("mean_reversion_growth_rate", dist.HalfNormal(10))
        if y is not None:
            for i in np.flatnonzero(y == -1):
                y = jax.ops.index_update(y, i, numpyro.sample(f"y_null_{i}", dist.Bernoulli(0.5).mask(False)))
                
        ema_init = numpyro.sample("ema_init", dist.Normal(0, 1))

        def transition(carry, xs):
            weight_curr, ema_pos_prev, y_prev = carry
            stage_curr, switch, x_curr, y_curr = xs 
            weight_curr = numpyro.deterministic("AR(1)", weight_curr)
            weights_with_offset = numpyro.deterministic(
                "stimulated_weights", init_weight + weight_curr
            )

            logit = numpyro.deterministic(
                "logits",
                x_curr[0] * weights_with_offset[0]
                + x_curr[1] * weights_with_offset[1]
                + weights_with_offset[2]
            )
            prob_with_lapse = numpyro.deterministic(
                "probs_with_lapse",
                (1 - lapse_prob[stage_curr]) * jax.nn.sigmoid(logit)
                + lapse_prob[stage_curr] * jax.nn.sigmoid(ema_pos_prev),
            )
            obs = numpyro.sample(
                "y", dist.Bernoulli(probs=clamp_probs(prob_with_lapse)), obs=y_curr
            )
            weight_next_base = 0 #numpyro.sample("AR(1)_next_base", dist.Normal(jnp.zeros(3), 1).to_event(1))
            weight_next = (1 - mean_reversion[stage_curr]) * weight_curr + weight_next_base * volatility + drift[stage_curr]
            
            ema_pos_curr = numpyro.deterministic(
                "ema", ema(ema_pos_prev, y_prev == 1, perseverance_growth_rate[stage_curr], mean_reversion_growth_rate)
            )
#             lema_pos_curr = numpyro.deterministic(
#                 "ap_process_ema(
#                 ema_pos_curr, perseverance_growth_rate[stage_curr, 0]
#             )
            return (weight_next, ema_pos_curr, obs), (obs)

        _, (obs) = scan(
            transition,
            (jnp.zeros(3), ema_init, -1),
            (stage, switch_indicator, X, y),
            length=len(X),
        )

    return model

config_mechanistic = {
    "learning_rate": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {'scale': 1.0},
    },
#     "bias_learning_rate": {
#         "shape": (3,),
#         "dist_type": dist.HalfNormal,
#         "params": {'scale': 0.1},
#     },
    "lapse_prob": {
        "shape": (1,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 10.0, "concentration1": 1.0},
    },
    "approach_given_lapse": {
        "shape": (1,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "mean_reversion": {
        "shape": (3,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "baselined_pers": True
}


def generate_mechanistic_model(config):
#     switch_scale = config['switch_scale']

    bp = int(config['baselined_pers'])

    def numpyro_config_sample(name, target_shape, **kwargs):
        return _numpyro_config_sample(name, config, target_shape, **kwargs)

    def model(X, stage, y=None, str_date=None):
        X = np.concatenate((X, np.ones((X.shape[0],1))),axis=1)


        date = None
        global_intercept = 0

        true_weight_nostim = numpyro.sample(f"true_weight_nostim", dist.Normal(0, 1).expand((1,3)).to_event(1))
    
        true_weight_stim = numpyro.sample(f"true_weight_stim_base", dist.Normal(0, 1).expand((1,3)).to_event(1))
        true_weight_stim = numpyro.deterministic("true_weight_stim", true_weight_nostim + true_weight_stim)
        
        true_weight_resid = numpyro.sample(f"true_weight_resid_base", dist.Normal(0, 1).expand((1,3)).to_event(1))
        true_weight_resid = numpyro.deterministic("true_weight_resid", true_weight_stim + true_weight_resid)

        true_weight_mean = jnp.vstack([true_weight_nostim, true_weight_stim, true_weight_resid])

        init_weight = numpyro.sample("initial_weight_base", dist.Normal(0, 1).expand((3,)).to_event(1))
        init_weight = numpyro.deterministic("initial_weight", true_weight_nostim[0] + init_weight)

        
#         repetition_kernel = numpyro_config_sample(
#             "repetition_kernel", target_shape=3, date=date,
#         )

        switch_indicator = generate_switch_indicator(stage)

        learning_rate = numpyro_config_sample(
            "learning_rate",
            target_shape=(3,), date=date, scale=HN_SIGMA
        )
#         learning_rate = numpyro.deterministic(f"{date}_learning_rate_scaled", learning_rate * learning_rate_hyper)
#         bias_learning_rate = numpyro_config_sample(
#             "bias_learning_rate",
#             target_shape=3, date=date, scale=HN_SIGMA * 0.1
#         )
#         bias_learning_rate = numpyro.deterministic(f"{date}_bias_learning_rate_scaled", bias_learning_rate * bias_learning_rate_hyper)

#             perseverance_weight = numpyro_config_sample(
#                 "perseverance_weight",
#                 target_shape=3, date=date
#             )
        lapse_prob = numpyro_config_sample(
            "lapse_prob",
            target_shape=3,
            date=date
        )
        approach_given_lapse = numpyro_config_sample(
            "approach_given_lapse",
            target_shape=3,
            date=date
        )
        mean_reversion = numpyro_config_sample(
            "mean_reversion",
            target_shape=3,
        )
        bias_init = 0.0
#         bias_init = numpyro.sample(f"bias_init", dist.Normal(0,1))

#         local_intercept = numpyro.sample(f"local_intercept", dist.Normal(0,10))
        # local_intercept = 0

#         intercept = numpyro.deterministic(f"net_intercept", global_intercept + local_intercept)

        def transition(carry, xs):
            weight_prev, x_prev, bias_curr, true_utility_prev, y_prev = carry
            stage_curr, switch, x_curr, y_curr = xs
            weight_with_offset = numpyro.deterministic(f"stimulated_weights", init_weight + weight_prev)
            utility = numpyro.deterministic(f"predicted_utility", x_curr[0] * weight_with_offset[0] + x_curr[1] * weight_with_offset[1] + weight_with_offset[2])
#             bias_weighted = numpyro.deterministic(f"bias_weighted", repetition_kernel[stage_curr] * bias_curr)
#             weighted_utility = numpyro.deterministic(f"weighted_utility", utility + bias_weighted)
            probs = numpyro.deterministic(
                "probs_with_lapse",
                (1 - lapse_prob[stage_curr]) * jax.nn.sigmoid(utility)
                + lapse_prob[stage_curr] * approach_given_lapse[stage_curr],
            )
            obs = numpyro.sample(
                f"y", dist.Bernoulli(probs=clamp_probs(probs)), obs=y_curr
            )
            # ====Mechanistic part====
            true_utility = x_curr[0] * true_weight_nostim[stage_curr][0] + x_curr[1] * true_weight_nostim[stage_curr][1] + true_weight_nostim[stage_curr][2]
            delta = numpyro.deterministic(f"delta", (true_utility - utility) * x_curr * obs) # * obs because this update can only happen if approach happened
            new_weights = numpyro.deterministic(
                f"AR(1) with learning",
                (1 - mean_reversion[stage_curr]) * weight_prev + learning_rate[stage_curr] * delta) #TODO: Maybe weigh this update with perseverance_weight?

#             bias_next = numpyro.deterministic(
#                 f"bias_base",
#                 bias_curr + obs * (true_utility - bp * weighted_utility) * bias_learning_rate[stage_curr]
#             )
            bias_next = bias_init
            return (new_weights, x_curr, bias_next, true_utility, obs), (obs)

        _, (obs) = scan(
            transition,
            (np.zeros(3), np.zeros(3), bias_init, 0.0, -1),
            (stage, switch_indicator, X, y),
            length=len(X),
        )

    return model

config_mechanistic_perseveration = {
    "learning_rate": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {'scale': 0.1},
    },
    "bias_learning_rate": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {'scale': 0.1},
    },
    "lapse_prob": {
        "shape": (1,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 10.0, "concentration1": 1.0},
    },
    "approach_given_lapse": {
        "shape": (1,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "mean_reversion": {
        "shape": (3,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "bias_mean_reversion": {
        "shape": (3,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "baselined_pers": True
}


def generate_mechanistic_perseveration_model(config):
#     switch_scale = config['switch_scale']

#     bp = int(config['baselined_pers'])

    def numpyro_config_sample(name, target_shape, **kwargs):
        return _numpyro_config_sample(name, config, target_shape, **kwargs)

    def model(X, stage, y=None, str_date=None):

        date = None
        global_intercept = 0

        true_weight_nostim = numpyro.sample(f"true_weight_nostim", dist.Normal(0, 1).expand((1,3)).to_event(1))
    
        true_weight_stim = numpyro.sample(f"true_weight_stim_base", dist.Normal(0, 1).expand((1,3)).to_event(1))
        true_weight_stim = numpyro.deterministic("true_weight_stim", true_weight_nostim + true_weight_stim)
        
        true_weight_resid = numpyro.sample(f"true_weight_resid_base", dist.Normal(0, 1).expand((1,3)).to_event(1))
        true_weight_resid = numpyro.deterministic("true_weight_resid", true_weight_stim + true_weight_resid)

        true_weight_mean = jnp.vstack([true_weight_nostim, true_weight_stim, true_weight_resid])

#         init_weight = numpyro.sample("initial_weight_base", dist.Normal(0, 1).expand((3,)).to_event(1))
#         init_weight = numpyro.deterministic("initial_weight", true_weight_nostim[0] + init_weight)
        init_weight = numpyro.sample("initial_weight_base", dist.Normal(0, 1).expand((3,)).to_event(1))
        init_weight = numpyro.deterministic("initial_weight", true_weight_nostim[0] + init_weight)
        
#         repetition_kernel = numpyro_config_sample(
#             "repetition_kernel", target_shape=3, date=date,
#         )

        switch_indicator = generate_switch_indicator(stage)

        learning_rate = numpyro_config_sample(
            "learning_rate",
            target_shape=3, date=date, scale=HN_SIGMA * 0.1
        )
#         learning_rate = numpyro.deterministic(f"{date}_learning_rate_scaled", learning_rate * learning_rate_hyper)
        bias_learning_rate = numpyro_config_sample(
            "bias_learning_rate",
            target_shape=3, date=date, scale=HN_SIGMA
        )
#         bias_learning_rate = numpyro.deterministic(f"{date}_bias_learning_rate_scaled", bias_learning_rate * bias_learning_rate_hyper)

#             perseverance_weight = numpyro_config_sample(
#                 "perseverance_weight",
#                 target_shape=3, date=date
#             )
        lapse_prob = numpyro_config_sample(
            "lapse_prob",
            target_shape=3,
            date=date
        )
        approach_given_lapse = numpyro_config_sample(
            "approach_given_lapse",
            target_shape=3,
            date=date
        )
        mean_reversion = numpyro_config_sample(
            "mean_reversion",
            target_shape=3,
        )
        bias_mean_reversion = numpyro_config_sample(
            "bias_mean_reversion",
            target_shape=3,
        )
#         bias_init = 0.0
        bias_init = numpyro.sample(f"bias_init", dist.Normal(0,10))

#         local_intercept = numpyro.sample(f"local_intercept", dist.Normal(0,1))
        # local_intercept = 0

#         intercept = numpyro.deterministic(f"net_intercept", global_intercept + local_intercept)

        def transition(carry, xs):
            weight_prev, x_prev, bias_curr, true_utility_prev, y_prev = carry
            stage_curr, switch, x_curr, y_curr = xs
            weight_with_offset = numpyro.deterministic(f"stimulated_weights", init_weight[:2] + weight_prev)
            utility = numpyro.deterministic(f"predicted_utility", x_curr[0] * weight_with_offset[0] + x_curr[1] * weight_with_offset[1])# + init_weight[2])
#             bias_weighted = numpyro.deterministic(f"bias_weighted", repetition_kernel[stage_curr] * bias_curr)
            utility_with_bias = numpyro.deterministic(f"utility_with_bias", utility + bias_curr)
            probs = numpyro.deterministic(
                "probs_with_lapse",
                (1 - lapse_prob[stage_curr]) * jax.nn.sigmoid(utility_with_bias)
                + lapse_prob[stage_curr] * approach_given_lapse[stage_curr],
            )
            obs = numpyro.sample(
                f"y", dist.Bernoulli(probs=clamp_probs(probs)), obs=y_curr
            )
            # ====Mechanistic part====
            true_utility = x_curr[0] * true_weight_nostim[stage_curr][0] + x_curr[1] * true_weight_nostim[stage_curr][1] + true_weight_nostim[stage_curr][2]
#             error = true_utility - utility_with_bias
            error = true_utility - utility
            delta = numpyro.deterministic(f"delta", error * x_curr * obs) # * obs because this update can only happen if approach happened
            new_weights = numpyro.deterministic(
                f"AR(1) with learning",
                (1 - mean_reversion[stage_curr]) * weight_prev + learning_rate[stage_curr] * delta) #TODO: Maybe weigh this update with perseverance_weight?
            
            error_bias = true_utility - utility#_with_bias
            bias_next = numpyro.deterministic(
                f"bias_base",
                (1 - bias_mean_reversion[stage_curr]) * bias_curr + obs * error_bias * bias_learning_rate[stage_curr]
            )
            return (new_weights, x_curr, bias_next, true_utility, obs), (obs)

        _, (obs) = scan(
            transition,
            (np.zeros(2), np.zeros(2), bias_init, 0.0, -1),
            (stage, switch_indicator, X, y),
            length=len(X),
        )

    return model

config_mechanistic_with_descriptive_perseveration = {
    "learning_rate": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {'scale': 0.1},
    },
#     "bias_learning_rate": {
#         "shape": (3,),
#         "dist_type": dist.HalfNormal,
#         "params": {'scale': 0.1},
#     },
    "lapse_prob": {
        "shape": (1,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 10.0, "concentration1": 1.0},
    },
    "approach_given_lapse": {
        "shape": (1,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "mean_reversion": {
        "shape": (3,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "baselined_pers": True,
} | config_descriptive_perseveration


def generate_mechanistic_model_with_descriptive_perseveration(config):
#     switch_scale = config['switch_scale']

    bp = int(config['baselined_pers'])

    def numpyro_config_sample(name, target_shape, **kwargs):
        return _numpyro_config_sample(name, config, target_shape, **kwargs)
    
    def ema(ema_prev, value, alpha):
        return (1 - alpha) * ema_prev + value

    if config['poisson_cdf_diminishing_perseverance']:

        def process_ema(x, rate):
            return poisson_cdf(x, rate)

    else:

        def process_ema(x, *args):
            return x

    def model(X, stage, y=None, str_date=None):
        X = np.concatenate((X, np.ones((X.shape[0],1))),axis=1)


        date = None
        global_intercept = 0

        true_weight_nostim = numpyro.sample(f"true_weight_nostim", dist.Normal(0, 1).expand((1,3)).to_event(1))
    
        true_weight_stim = numpyro.sample(f"true_weight_stim_base", dist.Normal(0, 1).expand((1,3)).to_event(1))
        true_weight_stim = numpyro.deterministic("true_weight_stim", true_weight_nostim + true_weight_stim * 0.1)
        
        true_weight_resid = numpyro.sample(f"true_weight_resid_base", dist.Normal(0, 1).expand((1,3)).to_event(1))
        true_weight_resid = numpyro.deterministic("true_weight_resid", true_weight_stim + true_weight_resid * 0.1)

        true_weight_mean = jnp.vstack([true_weight_nostim, true_weight_stim, true_weight_resid])

        init_weight = numpyro.sample("initial_weight_base", dist.Normal(0, 1).expand((3,)).to_event(1))
        init_weight = numpyro.deterministic("initial_weight", true_weight_nostim[0] + init_weight * 0.1)

        
#         repetition_kernel = numpyro_config_sample(
#             "repetition_kernel", target_shape=3, date=date,
#         )
        repetition_kernel = numpyro_config_sample(
            "repetition_kernel", target_shape=(3, 2)
        )
        perseverance_growth_rate = numpyro_config_sample(
            "perseverance_growth_rate",
            target_shape=(3, 2),
        )
        forget_rate = numpyro_config_sample(
            "forget_rate",
            target_shape=(3,2),
        )

        switch_indicator = generate_switch_indicator(stage)

        learning_rate = numpyro_config_sample(
            "learning_rate",
            target_shape=3, date=date, scale=HN_SIGMA * 0.1
        )
#         learning_rate = numpyro.deterministic(f"{date}_learning_rate_scaled", learning_rate * learning_rate_hyper)
#         bias_learning_rate = numpyro_config_sample(
#             "bias_learning_rate",
#             target_shape=3, date=date, scale=HN_SIGMA * 0.1
#         )
#         bias_learning_rate = numpyro.deterministic(f"{date}_bias_learning_rate_scaled", bias_learning_rate * bias_learning_rate_hyper)

#             perseverance_weight = numpyro_config_sample(
#                 "perseverance_weight",
#                 target_shape=3, date=date
#             )
        lapse_prob = numpyro_config_sample(
            "lapse_prob",
            target_shape=3,
            date=date
        )
        approach_given_lapse = numpyro_config_sample(
            "approach_given_lapse",
            target_shape=3,
            date=date
        )
        mean_reversion = numpyro_config_sample(
            "mean_reversion",
            target_shape=3,
        )
        bias_init = 0.0
#         bias_init = numpyro.sample(f"bias_init", dist.Normal(0,1))

#         local_intercept = numpyro.sample(f"local_intercept", dist.Normal(0,10))
        # local_intercept = 0

#         intercept = numpyro.deterministic(f"net_intercept", global_intercept + local_intercept)

        def transition(carry, xs):
            weight_prev, x_prev, bias_curr, true_utility_prev, y_prev, ema_pos_prev, ema_neg_prev = carry
            stage_curr, switch, x_curr, y_curr = xs
            
            ema_pos_curr = numpyro.deterministic(
                "ema_positive", ema(ema_pos_prev, y_prev == 1, forget_rate[stage_curr, 0])
            )
            ema_neg_curr = numpyro.deterministic(
                "ema_negative", ema(ema_pos_prev, y_prev == 0, forget_rate[stage_curr, 1])
            )
            lema_pos_curr = process_ema(
                ema_pos_curr, perseverance_growth_rate[stage_curr, 0]
            )
            lema_neg_curr = process_ema(
                ema_neg_curr, perseverance_growth_rate[stage_curr, 1]
            )
            
            autocorr = numpyro.deterministic(
                "autocorr",
                lema_pos_curr * repetition_kernel[stage_curr, 0]
                + lema_neg_curr * repetition_kernel[stage_curr, 1],
            )
            weight_with_offset = numpyro.deterministic(f"stimulated_weights", init_weight + weight_prev)
            utility = numpyro.deterministic(f"predicted_utility", x_curr[0] * weight_with_offset[0] + x_curr[1] * weight_with_offset[1] + weight_with_offset[2])
#             bias_weighted = numpyro.deterministic(f"bias_weighted", repetition_kernel[stage_curr] * bias_curr)
            utility_with_autocorr = numpyro.deterministic(f"weighted_utility", utility + autocorr)
            probs = numpyro.deterministic(
                "probs_with_lapse",
                (1 - lapse_prob[stage_curr]) * jax.nn.sigmoid(utility_with_autocorr)
                + lapse_prob[stage_curr] * approach_given_lapse[stage_curr],
            )
            obs = numpyro.sample(
                f"y", dist.Bernoulli(probs=clamp_probs(probs)), obs=y_curr
            )
            # ====Mechanistic part====
            true_utility = x_curr[0] * true_weight_nostim[stage_curr][0] + x_curr[1] * true_weight_nostim[stage_curr][1] + true_weight_nostim[stage_curr][2]
            delta = numpyro.deterministic(f"delta", (true_utility - utility) * x_curr * obs) # * obs because this update can only happen if approach happened
            new_weights = numpyro.deterministic(
                f"AR(1) with learning",
                (1 - mean_reversion[stage_curr]) * weight_prev + learning_rate[stage_curr] * delta) #TODO: Maybe weigh this update with perseverance_weight?

#             bias_next = numpyro.deterministic(
#                 f"bias_base",
#                 bias_curr + obs * (true_utility - bp * weighted_utility) * bias_learning_rate[stage_curr]
#             )
            bias_next = bias_init
            return (new_weights, x_curr, bias_next, true_utility, obs, ema_pos_curr, ema_neg_curr), (obs)

        _, (obs) = scan(
            transition,
            (np.zeros(3), np.zeros(3), bias_init, 0.0, -1, 0.5, 0.5),
            (stage, switch_indicator, X, y),
            length=len(X),
        )

    return model