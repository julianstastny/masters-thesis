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
    with numpyro.validation_enabled():
        mcmc.run(rng_key, **kwargs)
    return mcmc, az.from_numpyro(mcmc)


def poisson_cdf(value, rate):
    #     k = jnp.floor(value) + 1
    k = value + 1
    return gammaincc(k, rate)


# def generate_onpolicy_model(
#     drift_scale,
#     repetition_kernel_scale,
#     stimulation_immediate_scale,
#     random_walk_scale,
#     switch_scale=None,
#     saturating_ema=False,
#     stage_dependence=True,
#     log_diminishing_perseverance=True,
#     lapse=True,
# ):
#     assert not (saturating_ema and log_diminishing_perseverance)
#     stage_dependence = int(stage_dependence)
#     switch_scale = 1 if switch_scale is None else switch_scale
#     if drift_scale:

#         def drift_fn():
#             return numpyro.sample(
#                 "drift",
#                 dist.Normal(jnp.zeros((3, 3)), drift_scale).to_event(
#                     1 + stage_dependence
#                 ),
#             )

#     else:

#         def drift_fn():
#             return jnp.zeros((3, 3))

#     if repetition_kernel_scale:

#         def repetition_kernel_fn():
#             return numpyro.sample(
#                 "repetition_kernel",
#                 dist.Normal(jnp.zeros((3, 2)), repetition_kernel_scale).to_event(
#                     1 + stage_dependence
#                 ),
#             )

#     else:

#         def repetition_kernel_fn():
#             return jnp.zeros((3, 2))

#     if stimulation_immediate_scale:

#         def stimulation_immediate_fn():
#             return numpyro.sample(
#                 "stimulation_immediate",
#                 dist.Normal(jnp.zeros((3, 3)), 10).to_event(1 + stage_dependence),
#             )

#     else:

#         def stimulation_immediate_fn():
#             return jnp.zeros((3, 3))

#     def generate_switch_indicator(stage):
#         array = np.zeros_like(stage)
#         array[np.argmax(stage == 1)] = 1
#         array[np.argmax(stage == 2)] = 1
#         return array

#     if lapse:

#         def lapse_fn():
#             lapse_prob = numpyro.sample("lapse_prob", dist.Beta(np.ones(3), np.ones(3)).to_event(1))
#             approach_given_lapse = numpyro.sample(
#                 "approach_given_lapse", dist.Beta(np.ones(3), np.ones(3)).to_event(1)
#             )
#             return lapse_prob, approach_given_lapse
#     else:

#         def lapse_fn():
#             return (0, 0)

#     if saturating_ema:

#         def ema(ema_prev, value, alpha):
#             return (1 - alpha) * ema_prev + alpha * value

#     else:

#         def ema(ema_prev, value, alpha):
#             return (1 - alpha) * ema_prev + value

#     if log_diminishing_perseverance:

# #         def process_ema(x):
# #             return jnp.log(x + 1)
#         def process_ema(x, rate):
#             return poisson_cdf(x, rate)
#     else:

#         def process_ema(x, *args):
#             return x

#     if stage_dependent_perseverance_growth_rate:
#         def get_perseverance_growth_rate():
#             return numpyro.sample('perseverance_growth_rate', dist.HalfNormal(np.ones(2)*10))
#     else:
#         def get_perseverance_growth_rate():
#             return numpyro.sample('perseverance_growth_rate', dist.HalfNormal(np.ones(2)*10))
#         perseverance_growth_rate = numpyro.sample('perseverance_growth_rate', dist.HalfNormal(np.ones(2)*10))


#     def model(X, stage, y=None):

#         true_weight_mean = numpyro.sample(
#             "true_weight_mean", dist.Normal(jnp.zeros(3), 1).to_event(1)
#         ).flatten()
#         true_weight_scale = numpyro.sample(
#             "true_weight_scale", dist.HalfNormal(jnp.ones(3)).to_event(1)
#         ).flatten()
#         volatility = numpyro.sample(
#             "volatility", dist.HalfNormal(jnp.ones(3)*0.1).to_event(1)
#         ).flatten()
#         with numpyro.handlers.reparam(config={"initial_weight": TransformReparam()}):
#             init_weight = numpyro.sample(
#                 "initial_weight",
#                 dist.TransformedDistribution(
#                     dist.Normal(jnp.zeros(3), 1).to_event(1),
#                     dist.transforms.AffineTransform(
#                         true_weight_mean,
#                         true_weight_scale,
#                     ),
#                 )
#             )
#         repetition_kernel = repetition_kernel_fn()
#         repetition_kernel_positive = repetition_kernel[:, 0]
#         repetition_kernel_negative = repetition_kernel[:, 1]
#         perseverance_growth_rate = numpyro.sample('perseverance_growth_rate', dist.HalfNormal(np.ones(2)*10))
#         drift = drift_fn()
#         stimulation_immediate = stimulation_immediate_fn()
#         switch_indicator = generate_switch_indicator(stage)
#         alpha = numpyro.sample(
#             "alpha", dist.Beta(np.ones(2), np.ones(2)).to_event(1)
#         ).flatten()
#         lapse_prob, approach_given_lapse = lapse_fn()
#         mean_reversion = numpyro.sample(
#             "mean_reversion", dist.Uniform(np.zeros(3), np.ones(3)).to_event(1)
#         )
#         if y is not None:
#             for i in np.flatnonzero(y==-1):
#                 y[i] = numpyro.sample(f"y_null_{i}", dist.Bernoulli(0.5).mask(False))
#         def transition(carry, xs):
#             weight_prev, ema_pos_prev, ema_neg_prev, y_prev = carry
#             stage_curr, switch, x_curr, y_curr = xs
#             with numpyro.handlers.reparam(config={"AR(1)": TransformReparam()}):
#                 weight_curr = numpyro.sample(
#                     "AR(1)",
#                     dist.TransformedDistribution(
#                         dist.Normal(jnp.zeros(3), 1).to_event(1),
#                         dist.transforms.AffineTransform(
#                             mean_reversion[stage_curr] * weight_prev
#                             + drift[stage_curr],
#                             volatility * (1 - switch) + switch_scale * switch,
#                         ),
#                     ),
#                 )
#             weights_with_offset = numpyro.deterministic(
#                 "stimulated_weights", init_weight + weight_curr
#             )
#             ema_pos_curr = numpyro.deterministic(
#                 "ema_positive", ema(ema_pos_prev, y_prev == 1, alpha[0])
#             )
#             ema_neg_curr = numpyro.deterministic(
#                 "ema_negative", ema(ema_pos_prev, y_prev == 0, alpha[1])
#             )
#             lema_pos_curr = process_ema(ema_pos_curr, perseverance_growth_rate[0])
#             lema_neg_curr = process_ema(ema_neg_curr, perseverance_growth_rate[1])
#             autocorr = numpyro.deterministic(
#                 "autocorr",
#                 lema_pos_curr * repetition_kernel_positive[stage_curr]
#                 + lema_neg_curr * repetition_kernel_negative[stage_curr],
#             )
#             logit = numpyro.deterministic(
#                 "logits",
#                 x_curr[0] * weights_with_offset[0]
#                 + x_curr[1] * weights_with_offset[1]
#                 + weights_with_offset[2]
#                 + autocorr,
#             )
#             prob_with_lapse = numpyro.deterministic(
#                 "probs_with_lapse",
#                 (1 - lapse_prob[stage_curr]) * jax.nn.sigmoid(logit)
#                 + lapse_prob[stage_curr] * approach_given_lapse[stage_curr],
#             )
#             obs = numpyro.sample(
#                 "y", dist.Bernoulli(probs=clamp_probs(prob_with_lapse)), obs=y_curr
#             )
#             return (weight_curr, ema_pos_curr, ema_neg_curr, obs), (obs)
#         _, (obs) = scan(
#             transition,
# #             (init_weight, np.array([ema_pos_init]), np.array([ema_neg_init]), np.array([y_prev_init])),
#             (init_weight, 0.5, 0.5, -1),
#             (stage, switch_indicator, X, y),
#             length=len(X),
#         )

#     return model


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


config = {
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


def generate_onpolicy_model(config):
    assert not (config['saturating_ema'] and config['poisson_cdf_diminishing_perseverance'])
    switch_scale = config['switch_scale']
    saturating_ema = int(config['saturating_ema'])

    def numpyro_config_sample(name, target_shape):
        return _numpyro_config_sample(name, config, target_shape)

    def generate_switch_indicator(stage):
        array = np.zeros_like(stage)
        array[np.argmax(stage == 1)] = 1
        array[np.argmax(stage == 2)] = 1
        return array

    def ema(ema_prev, value, alpha):
        return (1 - alpha) * ema_prev + alpha ** saturating_ema * value

    if config['poisson_cdf_diminishing_perseverance']:

        def process_ema(x, rate):
            return poisson_cdf(x, rate)

    else:

        def process_ema(x, *args):
            return x

    def model(X, stage, y=None):
        true_weight_mean = numpyro_sample(
            "true_weight_mean",
            dist.Normal,
            ind_shape=(3,),
            target_shape=3,
            loc=0.0,
            scale=1.0,
        )
        true_weight_scale = numpyro_sample(
            "true_weight_scale",
            dist.HalfNormal,
            ind_shape=(3,),
            target_shape=3,
            scale=1.0,
        )
        #         volatility = numpyro_sample(
        #             "volatility", dist.HalfNormal, ind_shape=(3,), target_shape=3, scale=0.1
        #         )
        volatility = numpyro_config_sample("volatility", target_shape=3)
        with numpyro.handlers.reparam(config={"initial_weight": TransformReparam()}):
            init_weight = numpyro.sample(
                "initial_weight",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(3), 1).to_event(1),
                    dist.transforms.AffineTransform(
                        true_weight_mean,
                        true_weight_scale,
                    ),
                ),
            )
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
        stimulation_immediate = numpyro_config_sample(
            "stimulation_immediate",
            target_shape=(3, 3),
        )
        switch_indicator = generate_switch_indicator(stage)
        forget_rate = numpyro_config_sample(
            "forget_rate",
            target_shape=2,
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
        approach_given_lapse = numpyro_config_sample(
            "approach_given_lapse",
            target_shape=(3,),
        )

#         if (y is None) or sum(y==-1)==0:
#             y_imputed = y
#         else:
#             for j in np.flatnonzero(y == -1):
#                 concat_list += [y[i:j]]
#                 y_j = numpyro.sample(f"y_null_{j}", dist.Bernoulli(0.5).mask(False))
#                 concat_list += [y_j]
#                 i = j+1
#             concat_list += [y[i:]]
#             y_imputed = jnp.concatenate(concat_list)
#         if y is not None and (sum(y==-1)==1):
#             j = np.flatnonzero(y==-1).item()
#             y_j = numpyro.sample(f"y_null_{j}", dist.Bernoulli(0.5).mask(False))
#             y = jax.ops.index_update(y, j, y_j)
#             y_imputed = jnp.concatenate([y[:j], y_j, y[j+1:]])
        if y is not None:
            for i in np.flatnonzero(y == -1):
                y = jax.ops.index_update(y, i, numpyro.sample(f"y_null_{i}", dist.Bernoulli(0.5).mask(False)))

        def transition(carry, xs):
            weight_prev, ema_pos_prev, ema_neg_prev, y_prev = carry
            stage_curr, switch, x_curr, y_curr = xs
            with numpyro.handlers.reparam(config={"AR(1)": TransformReparam()}):
                weight_curr = numpyro.sample(
                    "AR(1)",
                    dist.TransformedDistribution(
                        dist.Normal(jnp.zeros(3), 1).to_event(1),
                        dist.transforms.AffineTransform(
                            mean_reversion[stage_curr] * weight_prev
                            + drift[stage_curr],
                            volatility * (1 - switch) + switch_scale * switch,
                        ),
                    ),
                )
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
            return (weight_curr, ema_pos_curr, ema_neg_curr, obs), (obs)

        _, (obs) = scan(
            transition,
            #             (init_weight, np.array([ema_pos_init]), np.array([ema_neg_init]), np.array([y_prev_init])),
            (jnp.zeros(3), 0.5, 0.5, -1),
            (stage, switch_indicator, X, y),
            length=len(X),
        )

    return model

# def numpyro_reparameterized_sample

def generate_hierarchical_model(config):
    assert not (config['saturating_ema'] and config['poisson_cdf_diminishing_perseverance'])
    switch_scale = config['switch_scale']
    saturating_ema = int(config['saturating_ema'])

    def numpyro_config_sample(name, target_shape, **kwargs):
        return _numpyro_config_sample(name, config, target_shape, **kwargs)

    def generate_switch_indicator(stage):
        array = np.zeros_like(stage)
        array[np.argmax(stage == 1)] = 1
        array[np.argmax(stage == 2)] = 1
        return array

    def ema(ema_prev, value, alpha):
        return (1 - alpha) * ema_prev + alpha ** saturating_ema * value

    if config['poisson_cdf_diminishing_perseverance']:

        def process_ema(x, rate):
            return poisson_cdf(x, rate)

    else:

        def process_ema(x, *args):
            return x

    def model(X_sep, stage_sep, y_sep=None, str_dates=None):
        true_weight_hyper_mean = numpyro_sample(
            "true_weight_hyper_mean",
            dist.Normal,
            ind_shape=(3,),
            target_shape=3,
            loc=0.0,
            scale=1.0,
        )
        true_weight_hyper_scale = numpyro_sample(
            "true_weight_hyper_scale",
            dist.HalfNormal,
            ind_shape=(3,),
            target_shape=3,
            scale=1.0,
        )
#         true_weight_deviation_hyper = numpyro_sample(
#             "true_weight_deviation_hyper",
#             dist.HalfNormal,
#             ind_shape=(3,),
#             target_shape=3,
#             scale=1.0,
#         )
        volatility_hyper_scale = numpyro_config_sample("volatility_hyper_scale", target_shape=3)

        repetition_kernel_hyper_mean = numpyro_config_sample(
            "repetition_kernel_hyper_mean", target_shape=(3, 2)
        )
        repetition_kernel_hyper_scale = numpyro_config_sample(
            "repetition_kernel_hyper_scale", target_shape=(3, 2)
        )
        perseverance_growth_rate_hyper_scale = numpyro_config_sample(
            "perseverance_growth_rate_hyper_scale",
            target_shape=(3, 2),
        )
        drift_hyper_mean = numpyro_config_sample(
            "drift_hyper_mean",
            target_shape=(3, 3),
        )
        drift_hyper_scale = numpyro_config_sample(
            "drift_hyper_scale",
            target_shape=(3, 3),
        )
        forget_rate_hyper = numpyro_config_sample(
            "forget_rate_hyper",
            target_shape=2,
        )
        mean_reversion_hyper = numpyro_config_sample(
            "mean_reversion_hyper",
            target_shape=2,
        )
        lapse_prob = numpyro_config_sample(
            "lapse_prob",
            target_shape=(3,),
        )
        approach_given_lapse = numpyro_config_sample(
            "approach_given_lapse",
            target_shape=(3,),
        )


        for X, stage, y, date in zip(X_sep, stage_sep, y_sep, str_dates):
            switch_indicator = generate_switch_indicator(stage)
#             true_weight_mean = numpyro_sample(
#                 f"{date}_true_weight_mean",
#                 dist.Normal,
#                 ind_shape=(3,),
#                 target_shape=3,
#                 loc=true_weight_hyper_mean,
#                 scale=true_weight_hyper_scale,
#             )
#             true_weight_scale = numpyro_sample(
#                 "true_weight_scale",
#                 dist.HalfNormal,
#                 ind_shape=(3,),
#                 target_shape=3,
#                 scale=1.0,
#             )
            #         volatility = numpyro_sample(
            #             "volatility", dist.HalfNormal, ind_shape=(3,), target_shape=3, scale=0.1
            #         )
            volatility = numpyro_config_sample("volatility", target_shape=3, date=date, scale=volatility_hyper_scale)
            with numpyro.handlers.reparam(config={f"{date}_initial_weight": TransformReparam()}):
                init_weight = numpyro.sample(
                    f"{date}_initial_weight",
                    dist.TransformedDistribution(
                        dist.Normal(jnp.zeros(3), 1).to_event(1),
                        dist.transforms.AffineTransform(
                            true_weight_hyper_mean,
                            true_weight_hyper_scale,
                        ),
                    ),
                )
            repetition_kernel = numpyro_config_sample(
                "repetition_kernel", target_shape=(3, 2), date=date, loc=repetition_kernel_hyper_mean, scale=repetition_kernel_hyper_scale
            )
#             perseverance_growth_rate = numpyro_config_sample(
#                 "perseverance_growth_rate",
#                 target_shape=(3, 2), date=date, scale=perseverance_growth_rate_hyper_scale
#             )
            perseverance_growth_rate = numpyro_config_sample(
                "perseverance_growth_rate",
                target_shape=(3, 2), date=date, scale=1
            )
            perseverance_growth_rate = numpyro.deterministic(f"{date}_perseverance_growth_rate_scaled", perseverance_growth_rate * perseverance_growth_rate_hyper_scale)
            drift = numpyro_config_sample(
                "drift",
                target_shape=(3, 3), date=date, loc=drift_hyper_mean, scale=drift_hyper_scale
            )
#             stimulation_immediate = numpyro_config_sample(
#                 f"{date}_stimulation_immediate",
#                 target_shape=(3, 3), loc=stimulation_immediate_hyper_mean, scale=stimulation_immediate_hyper_scale
#             )
            switch_indicator = generate_switch_indicator(stage)
            forget_rate = numpyro_config_sample(
                "forget_rate",
                target_shape=2, date=date, concentration0=forget_rate_hyper[0], concentration1=forget_rate_hyper[1]
            )
            mean_reversion = numpyro_config_sample(
                "mean_reversion",
                target_shape=3, date=date, concentration0=mean_reversion_hyper[0], concentration1=mean_reversion_hyper[1]
            )


        #         if (y is None) or sum(y==-1)==0:
        #             y_imputed = y
        #         else:
        #             for j in np.flatnonzero(y == -1):
        #                 concat_list += [y[i:j]]
        #                 y_j = numpyro.sample(f"y_null_{j}", dist.Bernoulli(0.5).mask(False))
        #                 concat_list += [y_j]
        #                 i = j+1
        #             concat_list += [y[i:]]
        #             y_imputed = jnp.concatenate(concat_list)
        #         if y is not None and (sum(y==-1)==1):
        #             j = np.flatnonzero(y==-1).item()
        #             y_j = numpyro.sample(f"y_null_{j}", dist.Bernoulli(0.5).mask(False))
        #             y = jax.ops.index_update(y, j, y_j)
        #             y_imputed = jnp.concatenate([y[:j], y_j, y[j+1:]])
#             if y is not None:
#                 for i in np.flatnonzero(y == -1):
#                     y = jax.ops.index_update(y, i, numpyro.sample(f"y_null_{i}", dist.Bernoulli(0.5).mask(False)))

            def transition(carry, xs):
                weight_prev, ema_pos_prev, ema_neg_prev, y_prev = carry
                stage_curr, switch, x_curr, y_curr = xs
#                 with numpyro.handlers.reparam(config={"AR(1)": TransformReparam()}):
#                     weight_curr = numpyro.sample(
#                         f"{date}_AR(1)",
#                         dist.TransformedDistribution(
#                             dist.Normal(jnp.zeros(3), 1).to_event(1),
#                             dist.transforms.AffineTransform(
#                                 mean_reversion[stage_curr] * weight_prev
#                                 + drift[stage_curr],
#                                 volatility * (1 - switch) + switch_scale * switch,
#                             ),
#                         ),
#                     )
#                 weight_curr = numpyro.sample(f"{date}_AR(1)",
#                     dist.Normal(
#                         mean_reversion[stage_curr] * weight_prev + drift[stage_curr],
#                         volatility * (1 - switch) + switch_scale * switch).to_event(1))
                weight_curr = numpyro.sample(f"{date}_AR(1)_decentered", dist.Normal(jnp.zeros(3), 1).to_event(1))
                weight_curr = numpyro.deterministic(
                    f"{date}_AR(1)",
                    weight_curr * (volatility * (1 - switch) + switch_scale * switch) + mean_reversion[stage_curr] * weight_prev + drift[stage_curr]
                )

                weights_with_offset = numpyro.deterministic(
                    f"{date}_stimulated_weights", init_weight + weight_curr
                )
                ema_pos_curr = numpyro.deterministic(
                    f"{date}_ema_positive", ema(ema_pos_prev, y_prev == 1, forget_rate[0])
                )
                ema_neg_curr = numpyro.deterministic(
                    f"{date}_ema_negative", ema(ema_pos_prev, y_prev == 0, forget_rate[1])
                )
                lema_pos_curr = process_ema(
                    ema_pos_curr, perseverance_growth_rate[stage_curr, 0]
                )
                lema_neg_curr = process_ema(
                    ema_neg_curr, perseverance_growth_rate[stage_curr, 1]
                )
                autocorr = numpyro.deterministic(
                    f"{date}_autocorr",
                    lema_pos_curr * repetition_kernel[stage_curr, 0]
                    + lema_neg_curr * repetition_kernel[stage_curr, 1],
                )
                logit = numpyro.deterministic(
                    f"{date}_logits",
                    x_curr[0] * weights_with_offset[0]
                    + x_curr[1] * weights_with_offset[1]
                    + weights_with_offset[2]
                    + autocorr,
                )
                prob_with_lapse = numpyro.deterministic(
                    f"{date}_probs_with_lapse",
                    (1 - lapse_prob[stage_curr]) * jax.nn.sigmoid(logit)
                    + lapse_prob[stage_curr] * approach_given_lapse[stage_curr],
                )
                obs = numpyro.sample(
                    f"{date}_y", dist.Bernoulli(probs=clamp_probs(prob_with_lapse)), obs=y_curr
                )
                return (weight_curr, ema_pos_curr, ema_neg_curr, obs), (obs)

            _, (obs) = scan(
                transition,
                #             (init_weight, np.array([ema_pos_init]), np.array([ema_neg_init]), np.array([y_prev_init])),
                (jnp.zeros(3), 0.5, 0.5, -1),
                (stage, switch_indicator, X, y),
                length=len(X),
            )

    return model

# def generate_hierarchical_model(config):
#     assert not (config['saturating_ema'] and config['poisson_cdf_diminishing_perseverance'])
#     switch_scale = config['switch_scale']
#     saturating_ema = int(config['saturating_ema'])

#     def numpyro_config_sample(name, target_shape, params=None):
#         return _numpyro_config_sample(name, config, target_shape, params)

#     def generate_switch_indicator(stage):
#         array = np.zeros_like(stage)
#         array[np.argmax(stage == 1)] = 1
#         array[np.argmax(stage == 2)] = 1
#         return array

#     def ema(ema_prev, value, alpha):
#         return (1 - alpha) * ema_prev + alpha ** saturating_ema * value

#     if config['poisson_cdf_diminishing_perseverance']:

#         def process_ema(x, rate):
#             return poisson_cdf(x, rate)

#     else:

#         def process_ema(x, *args):
#             return x

#     def model(X_sep, stage_sep, y_sep=None, str_dates=None):
#         true_weight_mean_hyper = numpyro_sample(
#             "true_weight_hyper_mean",
#             dist.Normal,
#             ind_shape=(3,),
#             target_shape=3,
#             loc=0.0,
#             scale=1.0,
#         )
#         true_weight_scale_hyper = numpyro_sample(
#             "true_weight_hyper_scale",
#             dist.HalfNormal,
#             ind_shape=(3,),
#             target_shape=3,
#             scale=1.0,
#         )
#         repetition_kernel_hyper_mean = numpyro_config_sample(
#             "repetition_kernel_hyper_mean", target_shape=(3, 2)
#         )
#         repetition_kernel_hyper_scale = numpyro_config_sample(
#             "repetition_kernel_hyper_scale", target_shape=(3, 2)
#         )
#         perseverance_growth_rate_hyper_mean = numpyro_config_sample(
#             "perseverance_growth_rate_hyper_mean",
#             target_shape=(3, 2),
#         )
#         perseverance_growth_rate_hyper_scale = numpyro_config_sample(
#             "perseverance_growth_rate_hyper_scale",
#             target_shape=(3, 2),
#         )
#         drift_hyper_mean = numpyro_config_sample(
#             "drift_hyper_mean",
#             target_shape=(3, 3),
#         )
#         drift_hyper_scale = numpyro_config_sample(
#             "drift_hyper_scale",
#             target_shape=(3, 3),
#         )
#         stimulation_immediate_hyper_mean = numpyro_config_sample(
#             "stimulation_immediate_hyper_mean",
#             target_shape=(3, 3),
#         )
#         stimulation_immediate_hyper_scale = numpyro_config_sample(
#             "stimulation_immediate_hyper_scale",
#             target_shape=(3, 3),
#         )
#         forget_rate_hyper_mean = numpyro_config_sample(
#             "forget_rate_hyper_mean",
#             target_shape=2,
#         )
#         forget_rate_hyper_scale = numpyro_config_sample(
#             "forget_rate_hyper_scale",
#             target_shape=2,
#         )
#         mean_reversion_hyper_mean = numpyro_config_sample(
#             "mean_reversion_hyper_mean",
#             target_shape=3,
#         )
#         mean_reversion_hyper_scale = numpyro_config_sample(
#             "mean_reversion_hyper_scale",
#             target_shape=3,
#         )
#         lapse_prob = numpyro_config_sample(
#             "lapse_prob",
#             target_shape=(3,),
#         )
#         approach_given_lapse = numpyro_config_sample(
#             "approach_given_lapse",
#             target_shape=(3,),
#         )


#         for X, stage, y, date in zip(X_sep, stage_sep, y_sep, str_dates):
#             switch_indicator = generate_switch_indicator(stage)
#             true_weight_mean = numpyro_sample(
#                 f"{date}_true_weight_mean",
#                 dist.Normal,
#                 ind_shape=(3,),
#                 target_shape=3,
#                 loc=true_weight_mean_hyper,
#                 scale=true_weight_scale_hyper,
#             )
# #             true_weight_scale = numpyro_sample(
# #                 "true_weight_scale",
# #                 dist.HalfNormal,
# #                 ind_shape=(3,),
# #                 target_shape=3,
# #                 scale=1.0,
# #             )
#             #         volatility = numpyro_sample(
#             #             "volatility", dist.HalfNormal, ind_shape=(3,), target_shape=3, scale=0.1
#             #         )
#             volatility = numpyro_config_sample(f"{date}_volatility", target_shape=3)
#             with numpyro.handlers.reparam(config={f"{date}_initial_weight": TransformReparam()}):
#                 init_weight = numpyro.sample(
#                     f"{date}_initial_weight",
#                     dist.TransformedDistribution(
#                         dist.Normal(jnp.zeros(3), 1).to_event(1),
#                         dist.transforms.AffineTransform(
#                             true_weight_mean,
#                             true_weight_scale,
#                         ),
#                     ),
#                 )
#             repetition_kernel = numpyro_config_sample(
#                 f"{date}_repetition_kernel", target_shape=(3, 2)
#             )
#             perseverance_growth_rate = numpyro_config_sample(
#                 f"{date}_perseverance_growth_rate",
#                 target_shape=(3, 2),
#             )
#             drift = numpyro_config_sample(
#                 f"{date}_drift",
#                 target_shape=(3, 3),
#             )
#             stimulation_immediate = numpyro_config_sample(
#                 f"{date}_stimulation_immediate",
#                 target_shape=(3, 3),
#             )
#             switch_indicator = generate_switch_indicator(stage)
#             forget_rate = numpyro_config_sample(
#                 f"{date}_forget_rate",
#                 target_shape=2,
#             )
#             lapse_prob = numpyro_config_sample(
#                 f"{date}_lapse_prob",
#                 target_shape=(3,),
#             )
#             approach_given_lapse = numpyro_config_sample(
#                 f"{date}_approach_given_lapse",
#                 target_shape=(3,),
#             )
#             mean_reversion = numpyro_config_sample(
#                 f"{date}_mean_reversion",
#                 target_shape=3,
#             )
#             approach_given_lapse = numpyro_config_sample(
#                 f"{date}_approach_given_lapse",
#                 target_shape=(3,),
#             )

#         #         if (y is None) or sum(y==-1)==0:
#         #             y_imputed = y
#         #         else:
#         #             for j in np.flatnonzero(y == -1):
#         #                 concat_list += [y[i:j]]
#         #                 y_j = numpyro.sample(f"y_null_{j}", dist.Bernoulli(0.5).mask(False))
#         #                 concat_list += [y_j]
#         #                 i = j+1
#         #             concat_list += [y[i:]]
#         #             y_imputed = jnp.concatenate(concat_list)
#         #         if y is not None and (sum(y==-1)==1):
#         #             j = np.flatnonzero(y==-1).item()
#         #             y_j = numpyro.sample(f"y_null_{j}", dist.Bernoulli(0.5).mask(False))
#         #             y = jax.ops.index_update(y, j, y_j)
#         #             y_imputed = jnp.concatenate([y[:j], y_j, y[j+1:]])
#             if y is not None:
#                 for i in np.flatnonzero(y == -1):
#                     y = jax.ops.index_update(y, i, numpyro.sample(f"y_null_{i}", dist.Bernoulli(0.5).mask(False)))

#             def transition(carry, xs):
#                 weight_prev, ema_pos_prev, ema_neg_prev, y_prev = carry
#                 stage_curr, switch, x_curr, y_curr = xs
#                 with numpyro.handlers.reparam(config={"AR(1)": TransformReparam()}):
#                     weight_curr = numpyro.sample(
#                         "AR(1)",
#                         dist.TransformedDistribution(
#                             dist.Normal(jnp.zeros(3), 1).to_event(1),
#                             dist.transforms.AffineTransform(
#                                 mean_reversion[stage_curr] * weight_prev
#                                 + drift[stage_curr],
#                                 volatility * (1 - switch) + switch_scale * switch,
#                             ),
#                         ),
#                     )
#                 weights_with_offset = numpyro.deterministic(
#                     "stimulated_weights", init_weight + weight_curr
#                 )
#                 ema_pos_curr = numpyro.deterministic(
#                     "ema_positive", ema(ema_pos_prev, y_prev == 1, forget_rate[0])
#                 )
#                 ema_neg_curr = numpyro.deterministic(
#                     "ema_negative", ema(ema_pos_prev, y_prev == 0, forget_rate[1])
#                 )
#                 lema_pos_curr = process_ema(
#                     ema_pos_curr, perseverance_growth_rate[stage_curr, 0]
#                 )
#                 lema_neg_curr = process_ema(
#                     ema_neg_curr, perseverance_growth_rate[stage_curr, 1]
#                 )
#                 autocorr = numpyro.deterministic(
#                     "autocorr",
#                     lema_pos_curr * repetition_kernel[stage_curr, 0]
#                     + lema_neg_curr * repetition_kernel[stage_curr, 1],
#                 )
#                 logit = numpyro.deterministic(
#                     "logits",
#                     x_curr[0] * weights_with_offset[0]
#                     + x_curr[1] * weights_with_offset[1]
#                     + weights_with_offset[2]
#                     + autocorr,
#                 )
#                 prob_with_lapse = numpyro.deterministic(
#                     "probs_with_lapse",
#                     (1 - lapse_prob[stage_curr]) * jax.nn.sigmoid(logit)
#                     + lapse_prob[stage_curr] * approach_given_lapse[stage_curr],
#                 )
#                 obs = numpyro.sample(
#                     "y", dist.Bernoulli(probs=clamp_probs(prob_with_lapse)), obs=y_curr
#                 )
#                 return (weight_curr, ema_pos_curr, ema_neg_curr, obs), (obs)

#             _, (obs) = scan(
#                 transition,
#                 #             (init_weight, np.array([ema_pos_init]), np.array([ema_neg_init]), np.array([y_prev_init])),
#                 (jnp.zeros(3), 0.5, 0.5, -1),
#                 (stage, switch_indicator, X, y),
#                 length=len(X),
#             )

#     return model
