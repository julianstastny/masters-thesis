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
    with numpyro.validation_enabled():
        mcmc.run(rng_key, **kwargs)
    return mcmc, az.from_numpyro(mcmc)



def poisson_cdf(value, rate):
#     k = jnp.floor(value) + 1
    k = value + 1
    return gammaincc(k, rate)


def generate_onpolicy_model(
    drift_scale,
    repetition_kernel_scale,
    stimulation_immediate_scale,
    random_walk_scale,
    switch_scale=None,
    saturating_ema=False,
    stage_dependence=True,
    log_diminishing_perseverance=True,
    lapse=True,
):
    assert not (saturating_ema and log_diminishing_perseverance)
    stage_dependence = int(stage_dependence)
    switch_scale = 1 if switch_scale is None else switch_scale
    if drift_scale:

        def drift_fn():
            return numpyro.sample(
                "drift",
                dist.Normal(jnp.zeros((3, 3)), drift_scale).to_event(
                    1 + stage_dependence
                ),
            )

    else:

        def drift_fn():
            return jnp.zeros((3, 3))

    if repetition_kernel_scale:

        def repetition_kernel_fn():
            return numpyro.sample(
                "repetition_kernel",
                dist.Normal(jnp.zeros((3, 2)), repetition_kernel_scale).to_event(
                    1 + stage_dependence
                ),
            )

    else:

        def repetition_kernel_fn():
            return jnp.zeros((3, 2))

    if stimulation_immediate_scale:

        def stimulation_immediate_fn():
            return numpyro.sample(
                "stimulation_immediate",
                dist.Normal(jnp.zeros((3, 3)), 10).to_event(1 + stage_dependence),
            )

    else:

        def stimulation_immediate_fn():
            return jnp.zeros((3, 3))

    def generate_switch_indicator(stage):
        array = np.zeros_like(stage)
        array[np.argmax(stage == 1)] = 1
        array[np.argmax(stage == 2)] = 1
        return array

    if lapse:

        def lapse_fn():
            lapse_prob = numpyro.sample("lapse_prob", dist.Beta(np.ones(3), np.ones(3)).to_event(1))
            approach_given_lapse = numpyro.sample(
                "approach_given_lapse", dist.Beta(np.ones(3), np.ones(3)).to_event(1)
            )
            return lapse_prob, approach_given_lapse
    else:

        def lapse_fn():
            return (0, 0)

    if saturating_ema:

        def ema(ema_prev, value, alpha):
            return (1 - alpha) * ema_prev + alpha * value

    else:

        def ema(ema_prev, value, alpha):
            return (1 - alpha) * ema_prev + value

    if log_diminishing_perseverance:

#         def process_ema(x):
#             return jnp.log(x + 1)
        def process_ema(x, rate):
            return poisson_cdf(x, rate)
    else:

        def process_ema(x, *args):
            return x

    def model(X, stage, y=None, ema_pos_init=0.5, ema_neg_init=0.5, y_prev_init=-1, obs_mask=None):

        true_weight_mean = numpyro.sample(
            "true_weight_mean", dist.Normal(jnp.zeros(3), 1).to_event(1)
        ).flatten()
        true_weight_scale = numpyro.sample(
            "true_weight_scale", dist.HalfNormal(jnp.ones(3)).to_event(1)
        ).flatten()
        volatility = numpyro.sample(
            "volatility", dist.HalfNormal(jnp.ones(3)*0.1).to_event(1)
        ).flatten()
        with numpyro.handlers.reparam(config={"initial_weight": TransformReparam()}):
            init_weight = numpyro.sample(
                "initial_weight",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(3), 1).to_event(1),
                    dist.transforms.AffineTransform(
                        true_weight_mean,
                        true_weight_scale,
                    ),
                )
            )
        repetition_kernel = repetition_kernel_fn()
        repetition_kernel_positive = repetition_kernel[:, 0]
        repetition_kernel_negative = repetition_kernel[:, 1]
        perseverance_growth_rate = numpyro.sample('perseverance_growth_rate', dist.HalfNormal(np.ones(2)*10))
        drift = drift_fn()
        stimulation_immediate = stimulation_immediate_fn()
        switch_indicator = generate_switch_indicator(stage)
        alpha = numpyro.sample(
            "alpha", dist.Beta(np.ones(2), np.ones(2)).to_event(1)
        ).flatten()
        lapse_prob, approach_given_lapse = lapse_fn()
        mean_reversion = numpyro.sample(
            "mean_reversion", dist.Uniform(np.zeros(3), np.ones(3)).to_event(1)
        )
        if y is not None:
            for i in np.flatnonzero(y==-1):
                y[i] = numpyro.sample(f"y_null_{i}", dist.Bernoulli(0.5).mask(False))     
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
                "ema_positive", ema(ema_pos_prev, y_prev == 1, alpha[0])
            )
            ema_neg_curr = numpyro.deterministic(
                "ema_negative", ema(ema_pos_prev, y_prev == 0, alpha[1])
            )
            lema_pos_curr = process_ema(ema_pos_curr, perseverance_growth_rate[0])
            lema_neg_curr = process_ema(ema_neg_curr, perseverance_growth_rate[1])
            autocorr = numpyro.deterministic(
                "autocorr",
                lema_pos_curr * repetition_kernel_positive[stage_curr]
                + lema_neg_curr * repetition_kernel_negative[stage_curr],
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
            (init_weight, ema_pos_init, ema_neg_init, y_prev_init),
            (stage, switch_indicator, X, y),
            length=len(X),
        )

    return model


