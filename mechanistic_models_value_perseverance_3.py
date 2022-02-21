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

HN_SIGMA = 1.2533 # To make mean=1 for HalfNormal


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

hierarchical_mechanistic_base_config = {
    "repetition_kernel": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": HN_SIGMA},
    },
    # "repetition_kernel_hyper_mean": {
    #     "shape": (3,),
    #     "dist_type": dist.Normal,
    #     "params": {"loc": 0.0, "scale": 10},
    # },
    "repetition_kernel_hyper_scale": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": HN_SIGMA},
    },
    "perseverance_growth_rate": {
        "shape": (2,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "perseverance_growth_rate_hyper_scale": {
        "shape": (2,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "learning_rate_hyper": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 0.1},
    },
    "learning_rate": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {'scale': 0.1},
    },
    "bias_learning_rate_hyper": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 0.1},
    },
    "bias_learning_rate": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {'scale': 0.1},
    },
    "init_weight_deviation_hyper": {
        "shape": (2,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 1},
    },
    "forget_rate_pers": {
        "shape": (1,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "forget_rate_offer": {
        "shape": (1,),
        "dist_type": dist.Beta,
        "params": {"concentration0": 1.0, "concentration1": 1.0},
    },
    "bias_weight": {
        "shape": (3,),
        "dist_type": dist.Uniform,
        "params": {"low": 0.0, "high": 1.0},
    },
    "forget_rate_pers_hyper": {
        "shape": (2,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "forget_rate_offer_hyper": {
        "shape": (2,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 10},
    },
    "rtm_scale_hyper": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 1},
    },
    "rtm": {
        "shape": (3,),
        "dist_type": dist.HalfNormal,
        "params": {"scale": 1},
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
    "switch_scale": 1.0,
    "saturating_ema": False,
    "poisson_cdf_diminishing_perseverance": True,
    "baselined_pers": True
}



def generate_hierarchical_mechanistic_model(config):
    assert not (config['saturating_ema'] and config['poisson_cdf_diminishing_perseverance'])
#     switch_scale = config['switch_scale']
    saturating_ema = int(config['saturating_ema'])

    bp = int(config['baselined_pers'])

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

#         true_weight_hyper_mean = numpyro_sample(
#             "true_weight_hyper_mean",
#             dist.Normal,
#             ind_shape=(3,3),
#             target_shape=(3,3),
#             loc=0.0,
#             scale=1.0,
#         )
        # global_intercept = numpyro.sample("global_intercept_mean", dist.Normal(0,1))
        global_intercept = 0
        # true_weight_hyper_mean = numpyro_sample(
        #     "true_weight_hyper_mean",
        #     dist.Normal,
        #     ind_shape=(2,),
        #     target_shape=(2,),
        #     loc=0.0,
        #     scale=5.0,
        # )
        # true_weight_hyper_mean = jax.ops.index_update(true_weight_hyper_mean, 2, 0)
        true_weight_hyper_scale = numpyro_sample(
            "true_weight_hyper_scale",
            dist.HalfNormal,
            # ind_shape=(1,3),
            ind_shape=(2,),
            target_shape=(2,),
            scale=5.0,
        )
        # init_weight_deviation_hyper = numpyro_sample(
        #     "init_weight_deviation_hyper",
        #     dist.HalfNormal,
        #     ind_shape=(2,),
        #     target_shape=2,
        #     scale=5.0,
        # )
#         repetition_kernel_hyper_mean = numpyro_config_sample(
#             "repetition_kernel_hyper_mean", target_shape=3
#         )
# #         repetition_kernel_hyper_mean = jnp.cumsum(repetition_kernel_hyper_mean, axis=0)
        # repetition_kernel_hyper_scale = numpyro_config_sample(
        #     "repetition_kernel_hyper_scale", target_shape=3
        # )
#         perseverance_growth_rate_hyper_scale = numpyro_config_sample(
#             "perseverance_growth_rate_hyper_scale",
#             target_shape=(3, 2),
#         )
        # forget_rate_pers_hyper = numpyro_config_sample(
        #     "forget_rate_pers_hyper",
        #     target_shape=2,
        # )
        # forget_rate_offer_hyper = numpyro_config_sample(
        #     "forget_rate_offer_hyper",
        #     target_shape=2,
        # )
#         forget_rate_pers = numpyro_config_sample(
#             "forget_rate_pers",
#             target_shape=3,
#         )

        learning_rate_hyper_1 = numpyro_config_sample("learning_rate_hyper", target_shape=(3,), scale=1)
        learning_rate_hyper = learning_rate_hyper_1 * config['learning_rate_hyper']['params']['scale']

        bias_learning_rate_hyper_1 = numpyro_config_sample("bias_learning_rate_hyper", target_shape=(3,), scale=1)
        bias_learning_rate_hyper = bias_learning_rate_hyper_1 * config['bias_learning_rate_hyper']['params']['scale']

#         rtm_scale_hyper = numpyro_config_sample("rtm_scale_hyper", target_shape=3)
#         lapse_prob = numpyro_config_sample(
#             "lapse_prob",
#             target_shape=(3,),
#         )
#         approach_given_lapse = numpyro_config_sample(
#             "approach_given_lapse",
#             target_shape=(3,),
#         )
#         approach_given_lapse = jnp.array([0.5, 0.5, 0.5])
        for X, stage, y, date in zip(X_sep, stage_sep, y_sep, str_dates):
            # X = np.concatenate((X, np.ones((X.shape[0],1))),axis=1)
#             with numpyro.handlers.reparam(config={f"{date}_true_weight": TransformReparam()}):
#                 true_weight_mean = numpyro.sample(
#                     f"{date}_true_weight",
#                     dist.TransformedDistribution(
#                         dist.Normal(jnp.zeros((3,3)), 1).to_event(1),
#                         dist.transforms.AffineTransform(
#                             true_weight_hyper_mean,
#                             true_weight_hyper_scale,
#                         ),
#                     ),
#                 ) # Old setting
            with numpyro.handlers.reparam(config={f"{date}_true_weight_nostim": TransformReparam()}):
                true_weight_nostim = numpyro.sample(
                    f"{date}_true_weight_nostim",
                    dist.TransformedDistribution(
                        dist.HalfNormal(HN_SIGMA).expand((1,2)).to_event(1),
                        dist.transforms.AffineTransform(
                            0,
                            true_weight_hyper_scale,
                        ),
                    ),
                )
            with numpyro.handlers.reparam(config={f"{date}_true_weight_stim": TransformReparam()}):
                true_weight_stim = numpyro.sample(
                    f"{date}_true_weight_stim",
                    dist.TransformedDistribution(
                        dist.HalfNormal(HN_SIGMA).expand((1,2)).to_event(1),
                        dist.transforms.AffineTransform(
                            0,
                            true_weight_nostim,
                        ),
                    ),
                )
            with numpyro.handlers.reparam(config={f"{date}_true_weight_resid": TransformReparam()}):
                true_weight_resid = numpyro.sample(
                    f"{date}_true_weight_resid",
                    dist.TransformedDistribution(
                        dist.HalfNormal(HN_SIGMA).expand((1,2)).to_event(1),
                        dist.transforms.AffineTransform(
                            0,
                            true_weight_stim,
                        ),
                    ),
                )
            true_weight_mean = jnp.concatenate([true_weight_nostim, true_weight_stim, true_weight_resid], axis=0)
#             print(true_weight_mean.shape)

            with numpyro.handlers.reparam(config={f"{date}_initial_weight": TransformReparam()}):
                init_weight = numpyro.sample(
                    f"{date}_initial_weight",
                    dist.TransformedDistribution(
                        dist.HalfNormal(jnp.ones(2) * HN_SIGMA).to_event(1),
                        dist.transforms.AffineTransform(
                            0,
                            true_weight_nostim[0],
                        ),
                    ),
                )
            repetition_kernel = numpyro_config_sample(
                "repetition_kernel", target_shape=3, date=date,
            )
            # repetition_kernel = numpyro.deterministic("repetition_kernel_scaled", repetition_kernel_base * repetition_kernel_hyper_scale)
            # perseverance_growth_rate = numpyro_config_sample(
            #     "perseverance_growth_rate",
            #     target_shape=(3, 2), date=date, scale=1
            # )
            # perseverance_growth_rate = numpyro.deterministic(f"{date}_perseverance_growth_rate_scaled", perseverance_growth_rate * perseverance_growth_rate_hyper_scale)
#             perseverance_growth_rate = numpyro.deterministic(f"{date}_perseverance_growth_rate_scaled", perseverance_growth_rate_hyper_scale)
            switch_indicator = generate_switch_indicator(stage)
            # forget_rate_pers = numpyro_config_sample(
            #     "forget_rate_pers",
            #     target_shape=3, date=date, concentration0=forget_rate_pers_hyper[0], concentration1=forget_rate_pers_hyper[1]
            # )
            # forget_rate_offer = numpyro_config_sample(
            #     "forget_rate_offer",
            #     target_shape=3, date=date, concentration0=forget_rate_offer_hyper[0], concentration1=forget_rate_offer_hyper[1]
            # )
            learning_rate = numpyro_config_sample(
                "learning_rate",
                target_shape=3, date=date, scale=HN_SIGMA
            )
            learning_rate = numpyro.deterministic(f"{date}_learning_rate_scaled", learning_rate * learning_rate_hyper)
            bias_learning_rate = numpyro_config_sample(
                "bias_learning_rate",
                target_shape=3, date=date, scale=HN_SIGMA
            )
            bias_learning_rate = numpyro.deterministic(f"{date}_bias_learning_rate_scaled", bias_learning_rate * bias_learning_rate_hyper)

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
            bias_init = 0.0
            # bias_init = numpyro.sample(f"{date}_bias_init", dist.Normal(0,1))

            local_intercept = numpyro.sample(f"{date}_local_intercept", dist.Normal(0,1))
            # local_intercept = 0

            intercept = numpyro.deterministic(f"{date}_net_intercept", global_intercept + local_intercept)
#             rtm = numpyro_config_sample(
#                 "rtm",
#                 target_shape=3,
#                 date=date,
#                 scale=rtm_scale_hyper
#             )
            def transition(carry, xs):
                weight_prev, x_prev, bias_curr, true_utility_prev, y_prev = carry
                stage_curr, switch, x_curr, y_curr = xs
                weight_with_offset = numpyro.deterministic(f"{date}_stimulated_weights", init_weight + weight_prev)
                utility = numpyro.deterministic(f"{date}_predicted_utility", x_curr[0] * weight_with_offset[0] - x_curr[1] * weight_with_offset[1] + intercept)
                bias_weighted = numpyro.deterministic(f"{date}_bias_weighted", repetition_kernel[stage_curr] * bias_curr)
                probs = numpyro.deterministic(f"{date}_probs_given_stimulus", jax.nn.sigmoid(utility + bias_weighted))
#                 perseverance_curr = (
#                     (1-forget_rate_pers) * perseverance_prev + forget_rate_pers * repetition_kernel[stage_curr,0] * x_prev[0] + repetition_kernel[stage_curr,1] * x_prev[1]
#                 )
#                 print(perseverance_curr.shape)
#                 prob_given_perseverance = numpyro.deterministic(f"{date}_probs_given_perseverance", jax.nn.sigmoid(perseverance_curr))
#                 prob_mixture = (1-perseverance_weight[stage_curr]) * prob_given_stimulus + perseverance_weight[stage_curr] * prob_given_perseverance
#                 prob_with_lapse = numpyro.deterministic(
#                     f"{date}_probs_with_lapse",
#                     (1 - lapse_prob[stage_curr]) * prob_mixture
#                     + lapse_prob[stage_curr] * approach_given_lapse[stage_curr],
#                 )
                obs = numpyro.sample(
                    f"{date}_y", dist.Bernoulli(probs=clamp_probs(probs)), obs=y_curr
                )
                # ====Mechanistic part====
                true_utility = x_curr[0] * true_weight_mean[stage_curr][0] - x_curr[1] * true_weight_mean[stage_curr][1] + intercept
                delta = numpyro.deterministic(f"{date}_delta", (true_utility - utility) * x_curr * jnp.array([1.0, -1.0]) * obs) # * obs because this update can only happen if approach happened
                new_weights = numpyro.deterministic(
                    f"{date}_AR(1) with learning",
                    weight_prev + learning_rate[stage_curr] * delta) #TODO: Maybe weigh this update with perseverance_weight?
                # perseverance_next = (
                #     (1-forget_rate_pers[stage_curr]) * bias_curr + forget_rate_pers[stage_curr] * obs * (true_utility - bp * utility) * perseverance_learning_rate[stage_curr]
                # )
                bias_next = numpyro.deterministic(
                    f"{date}_bias_base",
                    bias_curr + obs * (true_utility - bp * utility) * bias_learning_rate[stage_curr]
                )
                return (new_weights, x_curr, bias_next, true_utility, obs), (obs)

            _, (obs) = scan(
                transition,
                (np.zeros(2), np.zeros(2), bias_init, 0.0, -1),
                (stage, switch_indicator, X, y),
                length=len(X),
            )

    return model
