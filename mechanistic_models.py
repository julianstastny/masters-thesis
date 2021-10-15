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
    nuts_kernel = NUTS(model, adapt_step_size=True)
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


config = {
    "volatility": {
        "shape": (3,),
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
    "stimulation_immediate": {
        "shape": (3, 3),
        "dist_type": dist.Normal,
        "params": {"scale": 10},
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
    "lapse_prob": {
        "shape": (3,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "approach_given_lapse": {
        "shape": (3,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "inverse_mean_reversion": {
        "shape": (3,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "learning_rate": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 0.1},
    },
    "switch_scale": 1.0,
    "saturating_ema": False,
    "poisson_cdf_diminishing_perseverance": True
}

hierarchical_mechanistic_base_config = {
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
    "learning_rate_hyper": {
        "shape": (2, 3, 3),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "learning_rate": {
        "shape": (3, 3),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "init_weight_deviation_hyper": {
        "shape": (3),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 1},
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
def generate_mechanistic_model(config):
    assert not (config['saturating_ema'] and config['poisson_cdf_diminishing_perseverance'])
#     switch_scale = config['switch_scale']
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
        X = np.concatenate((X, np.ones((X.shape[0],1))),axis=1)
        print(X.shape)
#         true_weight_mean = numpyro_sample(
#             "true_weight_mean",
#             dist.Normal,
#             ind_shape=(3,),
#             target_shape=3,
#             loc=0.0,
#             scale=1.0,
#         )
#         true_weight_mean_prior_mean = numpyro_sample(
#             "true_weight_mean_prior_mean",
#             dist.Normal,
#             ind_shape=(3,),
#             target_shape=3,
#             loc=0.0,
#             scale=1.0,
#         )
#         true_weight_mean_prior_scale = numpyro_sample(
#             "true_weight_mean_prior_scale",
#             dist.HalfNormal,
#             ind_shape=(3,),
#             target_shape=3,
#             scale=1.0,
#         )
        # true_weight_mean = []
        true_weight_mean_0 = numpyro_sample('true_weight_mean_nostim', dist.Normal, (3,), (1, 3))
        with numpyro.handlers.reparam(config={"true_weight_mean_stim": TransformReparam()}):
            true_weight_mean_1 = numpyro.sample(
                "true_weight_mean_stim",
                dist.TransformedDistribution(
                    dist.Normal(0, 1).expand((1,3)).to_event(1),
                    dist.transforms.AffineTransform(
                        true_weight_mean_0,
                        1
                    ),
                ),
            )
        if True:
            true_weight_mean_2 = numpyro.deterministic("true_weight_mean_resid", true_weight_mean_0)
        else:
            with numpyro.handlers.reparam(config={"true_weight_mean_resid": TransformReparam()}):
                true_weight_mean_2 = numpyro.sample(
                    "true_weight_mean_resid",
                    dist.TransformedDistribution(
                        dist.Normal(0, 1).expand((1,3)).to_event(1),
                        dist.transforms.AffineTransform(
                            true_weight_mean_1,
                            1
                        ),
                    ),
                )
        # true_weight_mean += [jnp.expand_dims(true_weight_mean_i, 0)]
        true_weight_mean = jnp.concatenate([true_weight_mean_0, true_weight_mean_1, true_weight_mean_2], axis=0)
        #         volatility = numpyro_sample(
        #             "volatility", dist.HalfNormal, ind_shape=(3,), target_shape=3, scale=0.1
        #         )
        # true_weight_mean = jnp.concatenate(true_weight_mean, axis=0)
        print(true_weight_mean)
#         volatility = numpyro_config_sample("volatility", target_shape=3)
        learning_rate = numpyro_config_sample("learning_rate", target_shape=(3, 3))
        with numpyro.handlers.reparam(config={"initial_weight": TransformReparam()}):
            init_weight = numpyro.sample(
                "initial_weight",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(3), 1).to_event(1),
                    dist.transforms.AffineTransform(
#                         true_weight_mean,
                        true_weight_mean[0],
                        1
#                         true_weight_scale,
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
#         drift = numpyro_config_sample(
#             "drift",
#             target_shape=(3, 3),
#         )
#         stimulation_immediate = numpyro_config_sample(
#             "stimulation_immediate",
#             target_shape=(3, 3),
#         )
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
#         inverse_mean_reversion = numpyro_config_sample(
#             "inverse_mean_reversion",
#             target_shape=3,
#         )
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
            for j in np.flatnonzero(y==-1):
                y_j = numpyro.sample(f"y_null_{j}", dist.Bernoulli(0.5).mask(False))
                y = jax.ops.index_update(y, j, y_j)

        def transition(carry, xs):
#             weight_prev, ema_pos_prev, ema_neg_prev, y_prev, ar_1_prev = carry
            weight_prev, ema_pos_prev, ema_neg_prev, y_prev = carry
            stage_curr, switch, x_curr, y_curr = xs
            weight_with_offset = numpyro.deterministic('stimulated_weights', init_weight + weight_prev)
            ema_pos_curr = numpyro.deterministic(
                "ema_positive", ema(ema_pos_prev, y_prev == 1, forget_rate[0])
            )
            ema_neg_curr = numpyro.deterministic(
                "ema_negative", ema(ema_pos_prev, y_prev == 0, forget_rate[1])
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
            utility = x_curr[0] * weight_with_offset[0] + x_curr[1] * weight_with_offset[1] + weight_with_offset[2]
            logit = numpyro.deterministic(
                "logits",
                utility
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
            # ====Mechanistic part====
            true_utility = x_curr[0] * true_weight_mean[stage_curr][0] + x_curr[1] * true_weight_mean[stage_curr][1] + true_weight_mean[stage_curr][2]
            delta = numpyro.deterministic("delta", (utility - true_utility) * x_curr * obs) # * obs because this update can only happen if approach happened
            print(delta)
#             with numpyro.handlers.reparam(config={"AR(1) with learning": TransformReparam()}):
#                 new_weights = numpyro.sample(
#                     "AR(1) with learning",
#                     dist.TransformedDistribution(
#                         dist.Normal(jnp.zeros(3), 1).to_event(1),
#                         dist.transforms.AffineTransform(
#                             (1 - inverse_mean_reversion[stage_curr]) * weight_prev
#                             - learning_rate[stage_curr] * delta,
#                             volatility * (1 - switch) + switch_scale * switch,
#                         ),
#                     ),
#                 )
            new_weights = numpyro.deterministic(
                "AR(1) with learning",
                weight_prev - learning_rate[stage_curr] * delta)
            # TODO: Make it possible to change the true utility function
#             with numpyro.handlers.reparam(config={"true_weight_mean": TransformReparam()}):
#                 true_weight_mean = numpyro.sample(
#                     "true_weight_mean",
#                     dist.TransformedDistribution(
#                         dist.Normal(jnp.zeros(3), 1).to_event(1),
#                         dist.transforms.AffineTransform(
#                             true_weight_mean,
#                             volatility * (1 - switch) + switch_scale * switch,
#                         ),
#                     ),
#                 )
#             new_weights = numpyro.deterministic(
#                 "stimulated_weights", weight_prev + ar_1 - learning_rate * delta
#             )
#             new_weights = weight_prev + ar_1 - learning_rate[stage_curr] * delta
#             new_weights = weight_prev + ar_1 - delta
            # ====Mechanistic part====end
#             return (new_weights, ema_pos_curr, ema_neg_curr, obs, ar_1), (obs)
            return (new_weights, ema_pos_curr, ema_neg_curr, obs), (obs)

        _, (obs) = scan(
            transition,
            #             (init_weight, np.array([ema_pos_init]), np.array([ema_neg_init]), np.array([y_prev_init])),
#             (np.zeros(3), 0.5, 0.5, -1, jnp.zeros(3)),
            (np.zeros(3), 0.5, 0.5, -1),
            (stage, switch_indicator, X, y),
            length=len(X),
        )

    return model


def generate_hierarchical_mechanistic_model(config):
    assert not (config['saturating_ema'] and config['poisson_cdf_diminishing_perseverance'])
#     switch_scale = config['switch_scale']
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
            ind_shape=(3,3),
            target_shape=(3,3),
            loc=0.0,
            scale=1.0,
        )
        true_weight_hyper_scale = numpyro_sample(
            "true_weight_hyper_scale",
            dist.HalfNormal,
            ind_shape=(3,3),
            target_shape=(3,3),
            scale=1.0,
        )
#         true_weight_mean_0 = numpyro_sample('true_weight_mean_nostim', dist.Normal, (3,), (1, 3))
#         with numpyro.handlers.reparam(config={"true_weight_mean_stim": TransformReparam()}):
#             true_weight_mean_1 = numpyro.sample(
#                 "true_weight_mean_stim",
#                 dist.TransformedDistribution(
#                     dist.Normal(0, 1).expand((1,3)).to_event(1),
#                     dist.transforms.AffineTransform(
#                         true_weight_mean_0,
#                         1
#                     ),
#                 ),
#             )
#         if True:
#             true_weight_mean_2 = numpyro.deterministic("true_weight_mean_resid", true_weight_mean_0)
#         else:
#             with numpyro.handlers.reparam(config={"true_weight_mean_resid": TransformReparam()}):
#                 true_weight_mean_2 = numpyro.sample(
#                     "true_weight_mean_resid",
#                     dist.TransformedDistribution(
#                         dist.Normal(0, 1).expand((1,3)).to_event(1),
#                         dist.transforms.AffineTransform(
#                             true_weight_mean_1,
#                             1
#                         ),
#                     ),
#                 )
#         # true_weight_mean += [jnp.expand_dims(true_weight_mean_i, 0)]
#         true_weight_mean = jnp.concatenate([true_weight_mean_0, true_weight_mean_1, true_weight_mean_2], axis=0)

#         volatility = numpyro_config_sample("volatility", target_shape=3)
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
        forget_rate_hyper = numpyro_config_sample(
            "forget_rate_hyper",
            target_shape=2,
        )
        init_weight_deviation_hyper = numpyro_sample(
            "init_weight_deviation_hyper",
            dist.HalfNormal,
            ind_shape=(3,),
            target_shape=3,
            scale=1.0,
        )
        learning_rate_hyper = numpyro_config_sample("learning_rate_hyper", target_shape=(2, 3, 3))
        lapse_prob = numpyro_config_sample(
            "lapse_prob",
            target_shape=(3,),
        )
        approach_given_lapse = numpyro_config_sample(
            "approach_given_lapse",
            target_shape=(3,),
        )
#         learning_rate = numpyro_config_sample("learning_rate", target_shape=(3, 3))
        for X, stage, y, date in zip(X_sep, stage_sep, y_sep, str_dates):
            X = np.concatenate((X, np.ones((X.shape[0],1))),axis=1)
            with numpyro.handlers.reparam(config={f"{date}_true_weight": TransformReparam()}):
                true_weight_mean = numpyro.sample(
                    f"{date}_true_weight",
                    dist.TransformedDistribution(
                        dist.Normal(jnp.zeros((3,3)), 1).to_event(1),
                        dist.transforms.AffineTransform(
                            true_weight_hyper_mean,
                            true_weight_hyper_scale,
                        ),
                    ),
                )
            with numpyro.handlers.reparam(config={f"{date}_initial_weight": TransformReparam()}):
                init_weight = numpyro.sample(
                    f"{date}_initial_weight",
                    dist.TransformedDistribution(
                        dist.Normal(jnp.zeros(3), 1).to_event(1),
                        dist.transforms.AffineTransform(
                            true_weight_mean[0],
                            init_weight_deviation_hyper,
                        ),
                    ),
                )
            repetition_kernel = numpyro_config_sample(
                "repetition_kernel", target_shape=(3, 2), date=date, loc=repetition_kernel_hyper_mean, scale=repetition_kernel_hyper_scale
            )
            perseverance_growth_rate = numpyro_config_sample(
                "perseverance_growth_rate",
                target_shape=(3, 2), date=date, scale=1
            )
            perseverance_growth_rate = numpyro.deterministic(f"{date}_perseverance_growth_rate_scaled", perseverance_growth_rate * perseverance_growth_rate_hyper_scale)
#         drift = numpyro_config_sample(
#             "drift",
#             target_shape=(3, 3),
#         )
#         stimulation_immediate = numpyro_config_sample(
#             "stimulation_immediate",
#             target_shape=(3, 3),
#         )
            switch_indicator = generate_switch_indicator(stage)
            forget_rate = numpyro_config_sample(
                "forget_rate", 
                target_shape=2, date=date, concentration0=forget_rate_hyper[0], concentration1=forget_rate_hyper[1]
            )
            learning_rate = numpyro_config_sample(
                "learning_rate", 
                target_shape=(3,3), date=date, concentration0=learning_rate_hyper[0], concentration1=learning_rate_hyper[1]
            )
#             if y is not None:
#                 for j in np.flatnonzero(y==-1):
#                     y_j = numpyro.sample(f"y_null_{j}", dist.Bernoulli(0.5).mask(False))
#                     y = jax.ops.index_update(y, j, y_j)

            def transition(carry, xs):
    #             weight_prev, ema_pos_prev, ema_neg_prev, y_prev, ar_1_prev = carry
                weight_prev, ema_pos_prev, ema_neg_prev, y_prev = carry
                stage_curr, switch, x_curr, y_curr = xs
                weight_with_offset = numpyro.deterministic(f"{date}_stimulated_weights", init_weight + weight_prev)
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
                utility = x_curr[0] * weight_with_offset[0] + x_curr[1] * weight_with_offset[1] + weight_with_offset[2]
                logit = numpyro.deterministic(
                    f"{date}_logits",
                    utility
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
                # ====Mechanistic part====
                true_utility = x_curr[0] * true_weight_mean[stage_curr][0] + x_curr[1] * true_weight_mean[stage_curr][1] + true_weight_mean[stage_curr][2]
                delta = numpyro.deterministic(f"{date}_delta", (utility - true_utility) * x_curr * obs) # * obs because this update can only happen if approach happened
#                 print(delta)
    #             with numpyro.handlers.reparam(config={"AR(1) with learning": TransformReparam()}):
    #                 new_weights = numpyro.sample(
    #                     "AR(1) with learning",
    #                     dist.TransformedDistribution(
    #                         dist.Normal(jnp.zeros(3), 1).to_event(1),
    #                         dist.transforms.AffineTransform(
    #                             (1 - inverse_mean_reversion[stage_curr]) * weight_prev
    #                             - learning_rate[stage_curr] * delta,
    #                             volatility * (1 - switch) + switch_scale * switch,
    #                         ),
    #                     ),
    #                 )
                new_weights = numpyro.deterministic(
                    f"{date}_AR(1) with learning",
                    weight_prev - learning_rate[stage_curr] * delta)
                return (new_weights, ema_pos_curr, ema_neg_curr, obs), (obs)

            _, (obs) = scan(
                transition,
                #             (init_weight, np.array([ema_pos_init]), np.array([ema_neg_init]), np.array([y_prev_init])),
    #             (np.zeros(3), 0.5, 0.5, -1, jnp.zeros(3)),
                (np.zeros(3), 0.5, 0.5, -1),
                (stage, switch_indicator, X, y),
                length=len(X),
            )

    return model
