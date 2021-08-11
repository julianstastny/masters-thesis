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

def student_t_cdf(value, df, loc):
    scale = jnp.sqrt((df-2)/df)
#     scale = 1.0
    # Ref: https://en.wikipedia.org/wiki/Student's_t-distribution#Related_distributions
    # X^2 ~ F(1, df) -> df / (df + X^2) ~ Beta(df/2, 0.5)
    scaled = (value - loc) / scale
    scaled_squared = scaled * scaled
    beta_value = df / (df + scaled_squared)
    # when scaled < 0, returns 0.5 * Beta(df/2, 0.5).cdf(beta_value)
    # when scaled > 0, returns 1 - 0.5 * Beta(df/2, 0.5).cdf(beta_value)
    scaled_sign_half = 0.5 * jnp.sign(scaled)
    return (
        0.5
        + scaled_sign_half
        - 0.5 * jnp.sign(scaled) * betainc(0.5 * df, 0.5, beta_value)
    )

def poisson_cdf(value, rate):
#     k = jnp.floor(value) + 1
    k = value + 1
    return gammaincc(k, rate)


# def generate_pooled_model_log(
#     drift_scale,
#     repetition_kernel_scale,
#     stimulation_immediate_scale,
#     random_walk_scale,
#     stage_dependence=True,
# ):
#     stage_dependence = int(stage_dependence)
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

#     if random_walk_scale:

#         def random_walk_fn(num_steps):
#             return numpyro.sample(
#                 "random_walk",
#                 dist.GaussianRandomWalk(
#                     scale=np.ones(3) * random_walk_scale, num_steps=num_steps
#                 ).to_event(1),
#             ).T

#     else:

#         def random_walk_fn(num_steps):
#             return 0

#     def preprocess(y_prev):
#         for i in reversed(list(range(1, len(y_prev)))):
#             if y_prev[i] >= 1:
#                 num_cont = 1
#                 j = i
#                 while y_prev[j - 1] >= 1:
#                     num_cont += 1
#                     j -= 1
#             if y_prev[i] >= 1:
#                 y_prev[i] = num_cont
#         return np.log(y_prev + 1)

#     #     drift_fn = lambda: numpyro.sample('drift', dist.Normal(jnp.zeros((3, 3)), drift_scale)) if drift_scale else lambda: jnp.zeros((3, 3))
#     #     repetition_kernel_fn = lambda: numpyro.sample('repetition_kernel', dist.Normal(jnp.zeros(3), repetition_kernel_scale)).flatten() if repetition_kernel_scale else lambda: jnp.zeros(3)
#     def model(X, stage, y_prev_indicator, y=None):
#         init_weight = numpyro.sample(
#             "initial_weight", dist.Normal(jnp.zeros(3), 10).to_event(1)
#         ).flatten()
#         repetition_kernel = repetition_kernel_fn()
#         repetition_kernel_positive = repetition_kernel[:, 0]
#         repetition_kernel_negative = repetition_kernel[:, 1]
#         drift = drift_fn()
#         stimulation_immediate = stimulation_immediate_fn()
#         y_prev_positive = preprocess(
#             np.where(y_prev_indicator == 1, y_prev_indicator, 0)
#         )
#         y_prev_negative = preprocess(
#             np.where(y_prev_indicator == -1, np.abs(y_prev_indicator), 0)
#         )
#         switch_scale = numpyro.sample("switch_scale", dist.HalfNormal(1))
#         #         def transition(weight_prev, xs):
#         #             x_curr, stage_curr, y_prev = xs
#         #             weight_curr = numpyro.deterministic('stimulated_weights', weight_prev + drift[stage_curr])
#         #             return weight_curr, (weight_curr)
#         random_walk = random_walk_fn(len(X))
#         weights_summed = init_weight + jnp.cumsum(drift[stage], axis=0)
#         offset = stimulation_immediate[stage]
#         weights = numpyro.deterministic(
#             "stimulated_weights", weights_summed + offset + random_walk
#         )
#         #         print(weights.shape)
#         #         _, (weights) = scan(transition, init_weight, (X, stage, y_prev_indicator), length=len(X))
#         autocorrelation = numpyro.deterministic(
#             "autocorr",
#             y_prev_positive * repetition_kernel_positive[stage]
#             + y_prev_negative * repetition_kernel_negative[stage],
#         )
#         logits = numpyro.deterministic(
#             "logits",
#             X[:, 0] * weights[:, 0]
#             + X[:, 1] * weights[:, 1]
#             + weights[:, 2]
#             + autocorrelation,
#         )
#         with numpyro.plate("num_trials", len(X)):
#             obs = numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

#     return model


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

    #     if repetition_kernel_scale:
    #         def repetition_kernel_fn():
    #             initial_rk = numpyro.sample('repetition_kernel', dist.Normal(jnp.zeros(3), 1).to_event(1))
    #             rk_drift = numpyro.sample('rk_drift', dist.Normal(jnp.zeros((3, 2)), repetition_kernel_scale).to_event(1 + stage_dependence))
    #             return initial_rk, rk_drift
    #     else:
    #         def repetition_kernel_fn():
    #             return jnp.zeros((3, 2)), jnp.zeros((3, 2))
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

#         def lapse_fn():
#             lapse_prob = numpyro.sample("lapse_prob", dist.Beta(1, 1))
#             approach_given_lapse = numpyro.sample(
#                 "approach_given_lapse", dist.Beta(1, 1)
#             )
#             return lapse_prob, approach_given_lapse
        def lapse_fn():
            lapse_prob = numpyro.sample("lapse_prob", dist.Beta(np.ones(3), np.ones(3)).to_event(1))
#             approach_given_lapse = numpyro.sample(
#                 "approach_given_lapse", dist.Beta(1, 1)
#             )
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

    #     if random_walk_scale:
    #         def stimulated_weights_fn(weight_prev, drift, stage_curr, random_walk_scale, switch_scale, switch):
    #             with numpyro.handlers.reparam(config={"stimulated_weights": TransformReparam()}):
    #                 weight_curr = numpyro.sample(
    #                     "stimulated_weights",
    #                     dist.TransformedDistribution(
    #                         dist.Normal(jnp.zeros(3), 1),
    #                         dist.transforms.AffineTransform(weight_prev + drift[stage_curr], random_walk_scale * (1-switch) + switch_scale * switch),
    #                     ),
    #                 )
    #             return weight_curr
    #     else:
    #         def stimulated_weights_fn(weight_prev, drift, stage_curr, random_walk_scale, switch_scale, switch)
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
#         init_weight = numpyro.sample(
#             "initial_weight", dist.Normal(jnp.zeros(3), 1).to_event(1)
#         ).flatten()
#         init_weight = numpyro.sample(
#             "initial_weight", dist.Normal(true_weight_mean, 1).to_event(1)
#         ).flatten()
#         init_weight = numpyro.sample(
#             "initial_weight", dist.Normal(true_weight_mean, true_weight_scale).to_event(1)
#         ).flatten()
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
        perseverance_growth_rate = numpyro.sample('perseverance_growth_rate', dist.HalfNormal(10))
        drift = drift_fn()
        stimulation_immediate = stimulation_immediate_fn()
        switch_indicator = generate_switch_indicator(stage)
        alpha = numpyro.sample(
            "alpha", dist.Beta(np.ones(2), np.ones(2)).to_event(1)
        ).flatten()
        lapse_prob, approach_given_lapse = lapse_fn()
        #         lapse_prob = numpyro.sample('lapse_prob', dist.Beta(np.ones(3), np.ones(3)))
        #         approach_given_lapse = numpyro.sample('approach_given_lapse', dist.Beta(np.ones(3), np.ones(3)))
        #         mean_reversion_prior = numpyro.sample('mean_reversion_prior', dist.HalfCauchy(np.ones(2)).to_event(1)).flatten()
        #         mean_reversion_hyperprior_mean = numpyro.sample('mean_reversion_hyperprior_mean', dist.Uniform())
        #         mean_reversion_hyperprior_concentration = numpyro.sample('mean_reversion_hyperprior_concentration', dist.HalfNormal(10))
        #         with numpyro.handlers.reparam(config={"mean_reversion_prior": TransformReparam()}):
        #             mean_reversion_prior = numpyro.sample(
        #                 "mean_reversion_prior",
        #                 dist.TransformedDistribution(
        #                     dist.HalfNormal(np.ones((3,2))),
        #                     dist.transforms.AffineTransform(np.zeros((3,2)), mean_reversion_hyperprior),
        # #                         dist.transforms.AffineTransform(weight_prev + drift[stage_curr], random_walk_scale * (1-switch) + switch_scale * switch),
        #                 ).to_event(2),
        #             )
        #         mean_reversion_prior = numpyro.sample('mean_reversion_prior', dist.HalfNormal(np.ones((3,2))*10).to_event(2))
        #         mean_reversion_prior = numpyro.sample('mean_reversion_prior', dist.HalfNormal(np.ones((3,2))*10).to_event(2))
        #         mean_reversion_prior = numpyro.sample('mean_reversion_prior', dist.HalfNormal(np.ones(2)*10).to_event(1)).flatten()
        mean_reversion = numpyro.sample(
            "mean_reversion", dist.Uniform(np.zeros(3), np.ones(3)).to_event(1)
        )
        
#         nu = numpyro.sample('nu', dist.Gamma(2, 0.1))

#         if y is not None:
#             for i in np.flatnonzero(y==-1):
#                 y[i] = numpyro.sample(f"y_null_{i}", dist.Bernoulli(0.5).mask(False))
# # #                 print(i)
        if obs_mask is None:
            obs_mask = np.array([[True]] * len(X))
        
        def transition(carry, xs):
            weight_prev, ema_pos_prev, ema_neg_prev, y_prev = carry
            stage_curr, switch, x_curr, y_curr, obs_mask_curr = xs
            #             mean_reversion = numpyro.sample('mean_reversion', dist.Beta(*mean_reversion_prior[stage_curr]))
            #             mean_reversion = numpyro.sample('mean_reversion', dist.BetaProportion(mean_reversion_hyperprior_mean, mean_reversion_hyperprior_concentration))

            with numpyro.handlers.reparam(config={"AR(1)": TransformReparam()}):
                weight_curr = numpyro.sample(
                    "AR(1)",
                    dist.TransformedDistribution(
                        dist.Normal(jnp.zeros(3), 1).to_event(1),
                        #                         dist.transforms.AffineTransform(mean_reversion * init_weight + (1 - mean_reversion) * weight_prev + drift[stage_curr], random_walk_scale * (1-switch) + switch_scale * switch),
                        dist.transforms.AffineTransform(
                            mean_reversion[stage_curr] * weight_prev
                            + drift[stage_curr],
                            volatility * (1 - switch) + switch_scale * switch,
                        ),
                        #                         dist.transforms.AffineTransform(weight_prev + drift[stage_curr], random_walk_scale * (1-switch) + switch_scale * switch),
                    ),
                )
            #             with numpyro.handlers.reparam(config={"stimulated_weights": TransformReparam()}):
            #                 weight_curr = numpyro.sample(
            #                     "stimulated_weights",
            #                     dist.TransformedDistribution(
            #                         dist.Normal(jnp.zeros(3), 1),
            # #                         dist.transforms.AffineTransform(mean_reversion * init_weight + (1 - mean_reversion) * weight_prev + drift[stage_curr], random_walk_scale * (1-switch) + switch_scale * switch),
            #                         dist.transforms.AffineTransform(mean_reversion * weight_prev + drift[stage_curr], random_walk_scale * (1-switch) + switch_scale * switch),
            # #                         dist.transforms.AffineTransform(weight_prev + drift[stage_curr], random_walk_scale * (1-switch) + switch_scale * switch),
            #                     ),
            #                 )
            weights_with_offset = numpyro.deterministic(
                "stimulated_weights", init_weight + weight_curr
            )
            ema_pos_curr = numpyro.deterministic(
                "ema_positive", ema(ema_pos_prev, y_prev == 1, alpha[0])
            )  # (1 - alpha[0]) * ema_pos_prev + alpha[0] * (y_prev==1))
            ema_neg_curr = numpyro.deterministic(
                "ema_negative", ema(ema_pos_prev, y_prev == 0, alpha[1])
            )  # (1 - alpha[1]) * ema_neg_prev + alpha[1] * (y_prev==-1))
            lema_pos_curr = process_ema(ema_pos_curr, perseverance_growth_rate)
            lema_neg_curr = process_ema(ema_neg_curr, perseverance_growth_rate)
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
#             logit = numpyro.deterministic('logits', x_curr[0] * weight_curr[0] + x_curr[1] * weight_curr[1] + weight_curr[2] + autocorr)
#             prob_with_lapse = numpyro.deterministic(
#                 "probs_with_lapse",
#                 (1 - lapse_prob) * jax.nn.sigmoid(logit)
#                 + lapse_prob * approach_given_lapse,
#             )
            prob_with_lapse = numpyro.deterministic(
                "probs_with_lapse",
                (1 - lapse_prob[stage_curr]) * jax.nn.sigmoid(logit)
                + lapse_prob[stage_curr] * approach_given_lapse[stage_curr],
            )
            obs = numpyro.sample(
                "y", dist.Bernoulli(probs=clamp_probs(prob_with_lapse)), obs=y_curr, obs_mask=obs_mask_curr
            )

            return (weight_curr, ema_pos_curr, ema_neg_curr, obs), (obs)
        _, (obs) = scan(
            transition,
            (init_weight, ema_pos_init, ema_neg_init, y_prev_init),
            (stage, switch_indicator, X, y),
            length=len(X),
        )

    return model

#     def transition(carry, y_curr):
#         probs = some_function_of(carry)
# # y_curr = cond(
# #     y_curr != -1,
# #     lambda _: numpyro.deterministic('y_curr', y_curr),
# #     lambda _: numpyro.deterministic('y_curr', None),
# #     None,
# # )
#         obs = numpyro.sample(
#             "y", dist.Bernoulli(probs=probs, obs=y_curr
#         )
#     def transition(carry, y_curr):
#         probs = some_function_of(carry)
#         obs = numpyro.sample(
#             "y", dist.Bernoulli(probs=probs, obs=y_curr
#         )
#     _, (obs) = scan(transition, (carry_init), (y), length=len(y))
# def transition(carry, y_curr):
#     probs = some_function_of(carry)
#     y_curr = cond(
#         y_curr != -1,
#         lambda _: numpyro.deterministic('y_curr', y_curr),
#         lambda _: numpyro.deterministic('y_curr', None),
#         None,
#     )
#     obs = numpyro.sample(
#         "y", dist.Bernoulli(probs=probs, obs=y_curr
#     )

def generate_pooled_model_ema(
    drift_scale,
    repetition_kernel_scale,
    stimulation_immediate_scale,
    random_walk_scale,
    switch_scale=None,
    saturating_ema=False,
    stage_dependence=True,
    log_diminishing_perseverance=True,
):
    stage_dependence = int(stage_dependence)
    switch_scale = 10 * random_walk_scale if switch_scale is None else switch_scale
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

    if random_walk_scale:

        def random_walk_fn(num_steps):
            return numpyro.sample(
                "random_walk",
                dist.GaussianRandomWalk(
                    scale=np.ones(3) * random_walk_scale, num_steps=num_steps
                ).to_event(1),
            ).T

    else:

        def random_walk_fn(num_steps):
            return 0

    def generate_switch_indicator(stage):
        array = np.zeros_like(stage)
        array[np.argmax(stage == 1)] = 1
        array[np.argmax(stage == 2)] = 1
        return array

    if lapse:

        def lapse_fn():
            lapse_prob = numpyro.sample("lapse_prob", dist.Beta(1, 5))
            approach_given_lapse = numpyro.sample(
                "approach_given_lapse", dist.Beta(1, 1)
            )
            return lapse_prob, approach_given_lapse

    else:

        def lapse_fn():
            0, 0

    if saturating_ema:

        def ema(ema_prev, value, alpha):
            return (1 - alpha) * ema_prev + alpha * value

    else:

        def ema(ema_prev, value, alpha):
            return (1 - alpha) * ema_prev + value

    if log_diminishing_perseverance:

        def process_ema(x):
            return jnp.log(x + 1)

    else:

        def process_ema(x):
            return x

    def model(X, stage, y_prev_indicator, y=None):
        init_weight = numpyro.sample(
            "initial_weight", dist.Normal(jnp.zeros(3), 10).to_event(1)
        ).flatten()
        repetition_kernel = repetition_kernel_fn()
        repetition_kernel_positive = repetition_kernel[:, 0]
        repetition_kernel_negative = repetition_kernel[:, 1]
        drift = drift_fn()
        stimulation_immediate = stimulation_immediate_fn()
        switch_indicator = generate_switch_indicator(stage)
        alpha = numpyro.sample(
            "alpha", dist.Beta(np.ones(2), np.ones(2) * 2).to_event(1)
        ).flatten()
        lapse_prob, approach_given_lapse = lapse_fn()

        def transition(carry, xs):
            weight_prev, ema_pos_prev, ema_neg_prev = carry
            stage_curr, switch, y_prev = xs
            with numpyro.handlers.reparam(
                config={"stimulated_weights": TransformReparam()}
            ):
                weight_curr = numpyro.sample(
                    "stimulated_weights",
                    dist.TransformedDistribution(
                        dist.Normal(jnp.zeros(3), 1),
                        dist.transforms.AffineTransform(
                            weight_prev + drift[stage_curr],
                            random_walk_scale * (1 - switch) + switch_scale * switch,
                        ),
                    ),
                )
            ema_pos_curr = numpyro.deterministic(
                "ema_positive", ema(ema_pos_prev, y_prev == 1, alpha[0])
            )  # (1 - alpha[0]) * ema_pos_prev + alpha[0] * (y_prev==1))
            ema_neg_curr = numpyro.deterministic(
                "ema_negative", ema(ema_pos_prev, y_prev == -1, alpha[1])
            )  # (1 - alpha[1]) * ema_neg_prev + alpha[1] * (y_prev==-1))
            lema_pos_curr = process_ema(ema_pos_curr)
            lema_neg_curr = process_ema(ema_neg_curr)
            return (weight_curr, ema_pos_curr, ema_neg_curr), (
                weight_curr,
                lema_pos_curr,
                lema_neg_curr,
            )

        _, (weights, ema_positive, ema_negative) = scan(
            transition,
            (init_weight, 0.5, 0.5),
            (stage, switch_indicator, y_prev_indicator),
            length=len(y_prev_indicator),
        )
        autocorrelation = numpyro.deterministic(
            "autocorr",
            ema_positive * repetition_kernel_positive[stage]
            + ema_negative * repetition_kernel_negative[stage],
        )
        logits = numpyro.deterministic(
            "logits",
            X[:, 0] * weights[:, 0]
            + X[:, 1] * weights[:, 1]
            + weights[:, 2]
            + autocorrelation,
        )
        probs_with_lapse = numpyro.deterministic(
            "probs_with_lapse",
            (1 - lapse_prob) * jax.nn.sigmoid(logits)
            + lapse_prob * approach_given_lapse,
        )
        with numpyro.plate("num_trials", len(X)):
            obs = numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

    return model


def generate_pooled_model_ema_nonsaturating(
    drift_scale,
    repetition_kernel_scale,
    stimulation_immediate_scale,
    random_walk_scale,
    switch_scale=None,
    stage_dependence=True,
    log=False,
):
    stage_dependence = int(stage_dependence)
    switch_scale = 10 * random_walk_scale if switch_scale is None else switch_scale
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

    if log:

        def process_ema(x):
            return jnp.log(x + 1)

    else:

        def process_ema(x):
            return x

    #     if random_walk_scale:
    #         def random_walk_fn(num_steps):
    #             return numpyro.sample("random_walk", dist.GaussianRandomWalk(scale=np.ones(3)*random_walk_scale, num_steps=num_steps).to_event(1)).T
    #     else:
    #         def random_walk_fn(num_steps):
    #             return 0
    #     def preprocess(y_prev):
    #         for i in reversed(list(range(1, len(y_prev)))):
    #             if y_prev[i] >= 1:
    #                 num_cont = 1
    #                 j = i
    #                 while y_prev[j-1] >= 1:
    #                     num_cont += 1
    #                     j -= 1
    #             if y_prev[i] >= 1:
    #                 y_prev[i] = num_cont
    #         return np.log(y_prev + 1)
    def generate_switch_indicator(stage):
        array = np.zeros_like(stage)
        array[np.argmax(stage == 1)] = 1
        array[np.argmax(stage == 2)] = 1
        return array

    #     drift_fn = lambda: numpyro.sample('drift', dist.Normal(jnp.zeros((3, 3)), drift_scale)) if drift_scale else lambda: jnp.zeros((3, 3))
    #     repetition_kernel_fn = lambda: numpyro.sample('repetition_kernel', dist.Normal(jnp.zeros(3), repetition_kernel_scale)).flatten() if repetition_kernel_scale else lambda: jnp.zeros(3)
    def model(X, stage, y_prev_indicator, y=None):
        init_weight = numpyro.sample(
            "initial_weight", dist.Normal(jnp.zeros(3), 10).to_event(1)
        ).flatten()
        repetition_kernel = repetition_kernel_fn()
        repetition_kernel_positive = repetition_kernel[:, 0]
        repetition_kernel_negative = repetition_kernel[:, 1]
        drift = drift_fn()
        stimulation_immediate = stimulation_immediate_fn()
        #         y_prev_positive = preprocess(np.where(y_prev_indicator == 1, y_prev_indicator, 0))
        #         y_prev_negative = preprocess(np.where(y_prev_indicator == -1, np.abs(y_prev_indicator), 0))
        switch_indicator = generate_switch_indicator(stage)
        #         switch_scale = numpyro.sample('switch_scale', dist.HalfNormal(1))
        alpha = numpyro.sample(
            "alpha", dist.Beta(np.ones(2), np.ones(2) * 2).to_event(1)
        ).flatten()
        #         def transition(weight_prev, xs):
        #             x_curr, stage_curr, y_prev = xs
        #             weight_curr = numpyro.deterministic('stimulated_weights', weight_prev + drift[stage_curr])
        #             return weight_curr, (weight_curr)
        #         random_walk = random_walk_fn(len(X))
        lapse_prob = numpyro.sample("lapse_prob", dist.Beta(1, 5))
        approach_given_lapse = numpyro.sample("approach_given_lapse", dist.Beta(1, 1))

        def transition(carry, xs):
            weight_prev, ema_pos_prev, ema_neg_prev = carry
            stage_curr, switch, y_prev = xs
            with numpyro.handlers.reparam(
                config={"stimulated_weights": TransformReparam()}
            ):
                weight_curr = numpyro.sample(
                    "stimulated_weights",
                    dist.TransformedDistribution(
                        dist.Normal(jnp.zeros(3), 1),
                        dist.transforms.AffineTransform(
                            weight_prev + drift[stage_curr],
                            random_walk_scale * (1 - switch) + switch_scale * switch,
                        ),
                    ),
                )
            #             ema_pos_curr = numpyro.deterministic('ema_positive', (1 - alpha) * ema_pos_prev + alpha * (1-(y + y_prev)%2) * y)
            #             ema_neg_curr = numpyro.deterministic('ema_negative', (1 - alpha) * ema_neg_prev + alpha * (1-(y + y_prev)%2) * (1-y))
            ema_pos_curr = numpyro.deterministic(
                "ema_positive", (1 - alpha[0]) * ema_pos_prev + (y_prev == 1)
            )
            ema_neg_curr = numpyro.deterministic(
                "ema_negative", (1 - alpha[1]) * ema_neg_prev + (y_prev == -1)
            )
            lema_pos_curr = process_ema(ema_pos_curr)
            lema_neg_curr = process_ema(ema_neg_curr)
            return (weight_curr, ema_pos_curr, ema_neg_curr), (
                weight_curr,
                lema_pos_curr,
                lema_neg_curr,
            )

        _, (weights, ema_positive, ema_negative) = scan(
            transition,
            (init_weight, 0.5, 0.5),
            (stage, switch_indicator, y_prev_indicator),
            length=len(y_prev_indicator),
        )
        #         weights_summed = init_weight + jnp.cumsum(drift[stage], axis=0)
        #         offset = stimulation_immediate[stage]
        #         weights = numpyro.deterministic('stimulated_weights', weights_summed + offset + random_walk)
        #         print(weights.shape)
        #         _, (weights) = scan(transition, init_weight, (X, stage, y_prev_indicator), length=len(X))
        #         autocorrelation = numpyro.deterministic('autocorr', y_prev_positive * repetition_kernel_positive[stage] + y_prev_negative * repetition_kernel_negative[stage])
        autocorrelation = numpyro.deterministic(
            "autocorr",
            ema_positive * repetition_kernel_positive[stage]
            + ema_negative * repetition_kernel_negative[stage],
        )
        logits = numpyro.deterministic(
            "logits",
            X[:, 0] * weights[:, 0]
            + X[:, 1] * weights[:, 1]
            + weights[:, 2]
            + autocorrelation,
        )
        probs_with_lapse = numpyro.deterministic(
            "probs_with_lapse",
            (1 - lapse_prob) * jax.nn.sigmoid(logits)
            + lapse_prob * approach_given_lapse,
        )
        with numpyro.plate("num_trials", len(X)):
            obs = numpyro.sample("y", dist.Bernoulli(probs=probs_with_lapse), obs=y)

    return model


def generate_pooled_model_ema_nonsaturating_all(
    drift_scale,
    repetition_kernel_scale,
    stimulation_immediate_scale,
    random_walk_scale,
    switch_scale=None,
    stage_dependence=True,
    log=False,
):
    stage_dependence = int(stage_dependence)
    switch_scale = 10 * random_walk_scale if switch_scale is None else switch_scale
    if drift_scale:

        def drift_fn(i):
            return numpyro.sample(
                f"drift_{i}",
                dist.Normal(jnp.zeros((3, 3)), drift_scale).to_event(
                    1 + stage_dependence
                ),
            )

    else:

        def drift_fn(i):
            return jnp.zeros((3, 3))

    if repetition_kernel_scale:

        def repetition_kernel_fn(i):
            return numpyro.sample(
                f"repetition_kernel_{i}",
                dist.Normal(jnp.zeros((3, 2)), repetition_kernel_scale).to_event(
                    1 + stage_dependence
                ),
            )

    else:

        def repetition_kernel_fn(i):
            return jnp.zeros((3, 2))

    if stimulation_immediate_scale:

        def stimulation_immediate_fn(i):
            return numpyro.sample(
                f"stimulation_immediate_{i}",
                dist.Normal(jnp.zeros((3, 3)), 10).to_event(1 + stage_dependence),
            )

    else:

        def stimulation_immediate_fn(i):
            return jnp.zeros((3, 3))

    if log:

        def process_ema(x):
            return jnp.log(x + 1)

    else:

        def process_ema(x):
            return x

    #     if random_walk_scale:
    #         def random_walk_fn(num_steps):
    #             return numpyro.sample("random_walk", dist.GaussianRandomWalk(scale=np.ones(3)*random_walk_scale, num_steps=num_steps).to_event(1)).T
    #     else:
    #         def random_walk_fn(num_steps):
    #             return 0
    #     def preprocess(y_prev):
    #         for i in reversed(list(range(1, len(y_prev)))):
    #             if y_prev[i] >= 1:
    #                 num_cont = 1
    #                 j = i
    #                 while y_prev[j-1] >= 1:
    #                     num_cont += 1
    #                     j -= 1
    #             if y_prev[i] >= 1:
    #                 y_prev[i] = num_cont
    #         return np.log(y_prev + 1)
    def generate_switch_indicator(stage):
        array = np.zeros_like(stage)
        array[np.argmax(stage == 1)] = 1
        array[np.argmax(stage == 2)] = 1
        return array

    #     drift_fn = lambda: numpyro.sample('drift', dist.Normal(jnp.zeros((3, 3)), drift_scale)) if drift_scale else lambda: jnp.zeros((3, 3))
    #     repetition_kernel_fn = lambda: numpyro.sample('repetition_kernel', dist.Normal(jnp.zeros(3), repetition_kernel_scale)).flatten() if repetition_kernel_scale else lambda: jnp.zeros(3)
    def model(X_sep, stage_sep, y_prev_indicator_sep, y_sep=None):
        for i, (X, stage, y_prev_indicator, y) in enumerate(
            zip(X_sep, stage_sep, y_prev_indicator_sep, y_sep)
        ):
            init_weight = numpyro.sample(
                f"initial_weight_{i}", dist.Normal(jnp.zeros(3), 10).to_event(1)
            ).flatten()
            repetition_kernel = repetition_kernel_fn(i)
            repetition_kernel_positive = repetition_kernel[:, 0]
            repetition_kernel_negative = repetition_kernel[:, 1]
            drift = drift_fn(i)
            stimulation_immediate = stimulation_immediate_fn(i)
            #         y_prev_positive = preprocess(np.where(y_prev_indicator == 1, y_prev_indicator, 0))
            #         y_prev_negative = preprocess(np.where(y_prev_indicator == -1, np.abs(y_prev_indicator), 0))
            switch_indicator = generate_switch_indicator(stage)
            #         switch_scale = numpyro.sample('switch_scale', dist.HalfNormal(1))
            alpha = numpyro.sample(
                f"alpha_{i}", dist.Beta(np.ones(2), np.ones(2) * 2).to_event(1)
            ).flatten()
            #         def transition(weight_prev, xs):
            #             x_curr, stage_curr, y_prev = xs
            #             weight_curr = numpyro.deterministic('stimulated_weights', weight_prev + drift[stage_curr])
            #             return weight_curr, (weight_curr)
            #         random_walk = random_walk_fn(len(X))
            def transition(carry, xs):
                weight_prev, ema_pos_prev, ema_neg_prev = carry
                stage_curr, switch, y_prev = xs
                with numpyro.handlers.reparam(
                    config={f"stimulated_weights_{i}": TransformReparam()}
                ):
                    weight_curr = numpyro.sample(
                        f"stimulated_weights_{i}",
                        dist.TransformedDistribution(
                            dist.Normal(jnp.zeros(3), 1),
                            dist.transforms.AffineTransform(
                                weight_prev + drift[stage_curr],
                                random_walk_scale * (1 - switch)
                                + switch_scale * switch,
                            ),
                        ),
                    )
                #             ema_pos_curr = numpyro.deterministic('ema_positive', (1 - alpha) * ema_pos_prev + alpha * (1-(y + y_prev)%2) * y)
                #             ema_neg_curr = numpyro.deterministic('ema_negative', (1 - alpha) * ema_neg_prev + alpha * (1-(y + y_prev)%2) * (1-y))
                ema_pos_curr = numpyro.deterministic(
                    f"ema_positive_{i}", (1 - alpha[0]) * ema_pos_prev + (y_prev == 1)
                )
                ema_neg_curr = numpyro.deterministic(
                    f"ema_negative_{i}", (1 - alpha[1]) * ema_neg_prev + (y_prev == -1)
                )
                lema_pos_curr = process_ema(ema_pos_curr)
                lema_neg_curr = process_ema(ema_neg_curr)
                return (weight_curr, ema_pos_curr, ema_neg_curr), (
                    weight_curr,
                    lema_pos_curr,
                    lema_neg_curr,
                )

            _, (weights, ema_positive, ema_negative) = scan(
                transition,
                (init_weight, 0.5, 0.5),
                (stage, switch_indicator, y_prev_indicator),
                length=len(y_prev_indicator),
            )
            #         weights_summed = init_weight + jnp.cumsum(drift[stage], axis=0)
            #         offset = stimulation_immediate[stage]
            #         weights = numpyro.deterministic('stimulated_weights', weights_summed + offset + random_walk)
            #         print(weights.shape)
            #         _, (weights) = scan(transition, init_weight, (X, stage, y_prev_indicator), length=len(X))
            #         autocorrelation = numpyro.deterministic('autocorr', y_prev_positive * repetition_kernel_positive[stage] + y_prev_negative * repetition_kernel_negative[stage])
            autocorrelation = numpyro.deterministic(
                f"autocorr_{i}",
                ema_positive * repetition_kernel_positive[stage]
                + ema_negative * repetition_kernel_negative[stage],
            )
            logits = numpyro.deterministic(
                f"logits_{i}",
                X[:, 0] * weights[:, 0]
                + X[:, 1] * weights[:, 1]
                + weights[:, 2]
                + autocorrelation,
            )
            with numpyro.plate(f"num_trials_{i}", len(X)):
                obs = numpyro.sample(f"y_{i}", dist.Bernoulli(logits=logits), obs=y)

    return model


def generate_pooled_model_onpolicy(
    drift_scale, repetition_kernel_scale, stage_dependence=True
):
    stage_dependence = int(stage_dependence)
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

    #     drift_fn = lambda: numpyro.sample('drift', dist.Normal(jnp.zeros((3, 3)), drift_scale)) if drift_scale else lambda: jnp.zeros((3, 3))
    #     repetition_kernel_fn = lambda: numpyro.sample('repetition_kernel', dist.Normal(jnp.zeros(3), repetition_kernel_scale)).flatten() if repetition_kernel_scale else lambda: jnp.zeros(3)
    def model(X, stage, y=None):
        init_weight = numpyro.sample(
            "initial_weight", dist.Normal(jnp.zeros(3), 1).to_event(1)
        ).flatten()
        repetition_kernel = repetition_kernel_fn()
        repetition_kernel_positive = repetition_kernel[:, 0]
        repetition_kernel_negative = repetition_kernel[:, 1]
        drift = drift_fn()
        #         stimulation_immediate = numpyro.sample('stimulation_immediate', dist.Normal(jnp.zeros((3, 3)), 10).to_event(2))
        weights_summed = numpyro.deterministic(
            "stimulated_weights", init_weight + jnp.cumsum(drift[stage], axis=0)
        )

        def transition(y_prev, xs):
            weights_curr, stage_curr, x_curr, y_curr = xs
            autocorrelation = (y_prev == 1) * repetition_kernel_positive[stage_curr] + (
                y_prev == 0
            ) * repetition_kernel_negative[stage_curr]
            logits = numpyro.deterministic(
                "logits",
                x_curr[0] * weights_curr[0]
                + x_curr[1] * weights_curr[1]
                + weights_curr[2]
                + autocorrelation,
            )
            y = numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y_curr)
            return y, (y)

        #         weights_summed = jnp.cumsum(init_weight + drift[stage], axis=0)
        #         offset = stimulation_immediate[stage]
        #         weights = numpyro.deterministic('stimulated_weights', weights_summed + offset)
        #         print(weights.shape)
        _, (obs) = scan(transition, -1, (weights_summed, stage, X, y), length=len(X))

    #         autocorrelation = y_prev_positive * repetition_kernel_positive[stage] + y_prev_negative * repetition_kernel_negative[stage]
    #         logits = numpyro.deterministic('logits', X[:,0] * weights[:,0] + X[:,1] * weights[:,1] + weights[:,2] + autocorrelation)
    #         with numpyro.plate('num_trials', len(X)):
    #             obs = numpyro.sample('y', dist.Bernoulli(logits=logits), obs=y)

    return model


def generate_hierarchical_model(drift_scale, repetition_kernel_scale):
    if drift_scale:

        def drift_fn():
            return numpyro.sample("drift", dist.Normal(jnp.zeros((3, 3)), drift_scale))

    else:

        def drift_fn():
            return jnp.zeros((3, 3))

    if repetition_kernel_scale:

        def repetition_kernel_fn(name):
            return numpyro.sample(
                f"repetition_kernel_{name}",
                dist.Normal(jnp.zeros(3), repetition_kernel_scale),
            ).flatten()

    else:

        def repetition_kernel_fn(_):
            return jnp.zeros(3)


def partial_pooling_model(Xs, stages, y_prev_indicators, ys=None):
    drift_prior_mu = numpyro.sample(
        "drift_prior_mu", dist.Normal(jnp.zeros((3, 3)), 10)
    )
    drift_prior_sigma = numpyro.sample(
        "drift_prior_sigma", dist.HalfNormal(jnp.ones((3, 3)))
    )

    init_prior_mu = numpyro.sample("init_prior_mu", dist.Normal(jnp.zeros(3), 10))
    init_prior_sigma = numpyro.sample("init_prior_sigma", dist.HalfNormal(jnp.ones(3)))
    #     repetition_kernel_prior_mu = numpyro.sample('repetition_kernel_positive_prior_mu', dist.Normal(jnp.zeros(3), 1))
    repetition_kernel_prior_mu = numpyro.sample(
        "repetition_kernel_prior_mu", dist.Normal(jnp.zeros((3, 2)), 10)
    )
    #     repetition_kernel_prior_sigma = numpyro.sample('repetition_kernel_positive_prior_sigma', dist.HalfNormal(1))
    repetition_kernel_prior_sigma = numpyro.sample(
        "repetition_kernel_prior_sigma", dist.HalfNormal(jnp.ones((3, 2)))
    )
    #     repetition_kernel_negative_prior_mu = numpyro.sample('repetition_kernel_negative_prior_mu', dist.Normal(jnp.zeros(3), 1))
    #     repetition_kernel_negative_prior_sigma = numpyro.sample('repetition_kernel_negative_prior_sigma', dist.HalfNormal(1))
    #     for X, stage, y_prev_indicator, y in zip(Xs, stages, y_prev_indicators, ys):
    #         init_weight = numpyro.sample('initial_weight', dist.Normal(jnp.zeros(3), 1)).flatten()
    #         repetition_kernel_positive = numpyro.sample('repetition_kernel_positive', dist.Normal(repetition_kernel_positive_prior_mu, repetition_kernel_positive_prior_sigma)).flatten()
    #         repetition_kernel_negative = numpyro.sample('repetition_kernel_negative', dist.Normal(repetition_kernel_negative_prior_mu, repetition_kernel_negative_prior_sigma)).flatten()
    #         drift = numpyro.sample('drift', dist.Normal(drift_prior_mu, drift_prior_sigma))
    #         y_prev_positive = np.where(y_prev_indicator == 1, y_prev_indicator, 0)
    #         y_prev_negative = np.where(y_prev_indicator == -1, y_prev_indicator, 0)
    #         def transition(weight_prev, xs):
    #             x_curr, stage_curr, y_prev = xs
    #             weight_curr = numpyro.deterministic('stimulated_weights', weight_prev + drift[stage_curr])
    #             return weight_curr, (weight_curr)

    #         _, (weights) = scan(transition, init_weight, (X, stage, y_prev_indicator), length=len(X))
    #         autocorrelation = y_prev_positive * repetition_kernel_positive[stage] + y_prev_negative * repetition_kernel_negative[stage]
    #         logits = numpyro.deterministic('logits', X[:,0] * weights[:,0] + X[:,1] * weights[:,1] + weights[:,2] + autocorrelation)
    #         with numpyro.plate('num_trials', len(X)):
    #             obs = numpyro.sample('y', dist.Bernoulli(logits=logits), obs=y)
    for i, (X, stage, y_prev_indicator, y) in enumerate(
        zip(Xs, stages, y_prev_indicators, ys)
    ):
        init_weight = numpyro.sample(
            f"initial_weight_{i}",
            dist.Normal(init_prior_mu, init_prior_sigma).to_event(1),
        ).flatten()
        #         repetition_kernel_positive = numpyro.sample(f'repetition_kernel_positive_{i}', dist.Normal(repetition_kernel_positive_prior_mu, repetition_kernel_positive_prior_sigma)).flatten()
        #         repetition_kernel_negative = numpyro.sample(f'repetition_kernel_negative_{i}', dist.Normal(repetition_kernel_negative_prior_mu, repetition_kernel_negative_prior_sigma)).flatten()
        repetition_kernel = numpyro.sample(
            f"repetition_kernel_{i}",
            dist.Normal(
                repetition_kernel_prior_mu, repetition_kernel_prior_sigma
            ).to_event(2),
        )
        repetition_kernel_negative = repetition_kernel[:, 0]
        repetition_kernel_positive = repetition_kernel[:, 1]
        drift = numpyro.sample(
            f"drift_{i}", dist.Normal(drift_prior_mu, drift_prior_sigma).to_event(2)
        )
        y_prev_positive = np.where(y_prev_indicator == 1, y_prev_indicator, 0)
        y_prev_negative = np.where(y_prev_indicator == -1, np.abs(y_prev_indicator), 0)
        #         def transition(weight_prev, xs):
        #             x_curr, stage_curr, y_prev = xs
        #             weight_curr = numpyro.deterministic(f'stimulated_weights_{i}', weight_prev + drift[stage_curr])
        #             return weight_curr, (weight_curr)

        #         _, (weights) = scan(transition, init_weight, (X, stage, y_prev_indicator), length=len(X))
        weights = numpyro.deterministic(
            "stimulated_weights", init_weight + jnp.cumsum(drift[stage], axis=0)
        )

        autocorrelation = (
            y_prev_positive * repetition_kernel_positive[stage]
            + y_prev_negative * repetition_kernel_negative[stage]
        )
        logits = numpyro.deterministic(
            f"logits_{i}",
            X[:, 0] * weights[:, 0]
            + X[:, 1] * weights[:, 1]
            + weights[:, 2]
            + autocorrelation,
        )
        with numpyro.plate(f"num_trials_{i}", len(X)):
            obs = numpyro.sample(f"y_{i}", dist.Bernoulli(logits=logits), obs=y)


def latent_utility_model_nostim(X, y=None):
    X = np.concatenate((X, np.ones((len(X), 1))), 1)
    w_true = numpyro.sample(
        "true utility-function", dist.Normal(jnp.zeros(3), 1)
    ).flatten()
    #     init_weight = numpyro.sample('initial weight', dist.Normal(jnp.zeros(3), 1)).flatten()
    #     init_weight = numpyro.sample('initial weight', dist.Normal(w_true, 1)).flatten()
    with numpyro.handlers.reparam(config={"initial weight": TransformReparam()}):
        init_weight = numpyro.sample(
            "initial weight",
            dist.TransformedDistribution(
                dist.Normal(jnp.zeros(3), 1),
                dist.transforms.AffineTransform(w_true, 1),
            ),
        ).flatten()
    #     alpha = numpyro.sample('learning rate', dist.HalfNormal(jnp.ones(3))).flatten()
    alpha = numpyro.sample("learning rate", dist.HalfNormal())

    def transition(weight_prev, x):
        prob_approach = jax.nn.sigmoid(
            x[0] * weight_prev[0] + x[1] * weight_prev[1] + weight_prev[2]
        )
        delta = (w_true - weight_prev) * (
            x ** 2
        )  # The multiplication with x**2 is done under the assumption of a MSE between utilities
        with numpyro.handlers.reparam(
            config={"stimulated_weights": TransformReparam()}
        ):
            weight_prev = numpyro.sample(
                "stimulated_weights",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(3), 1),
                    dist.transforms.AffineTransform(
                        weight_prev + prob_approach * alpha * delta, 0.01
                    ),
                ),
            )
        return weight_prev, (weight_prev, prob_approach)

    _, (weights, probs) = scan(transition, init_weight, X, length=len(X))
    #     logits = numpyro.deterministic('logits', X[:,0] * weights[:,0] + X[:,1] * weights[:,1] + weights[:,2])
    with numpyro.plate("num_trials", len(X)):
        obs = numpyro.sample("y", dist.Bernoulli(probs), obs=y)


def latent_utility_model_stim(X, stim_indicator, y=None):
    X = np.concatenate((X, np.ones((len(X), 1))), 1)
    w_true = numpyro.sample(
        "true utility-function", dist.Normal(jnp.zeros(3) * 10, 1).to_event(1)
    ).flatten()
    w_stim = numpyro.sample(
        "stimulated utility-function delta",
        dist.Normal(jnp.zeros(3) * 10, 1).to_event(1),
    ).flatten()
    init_weight = numpyro.sample(
        "initial weight", dist.Normal(jnp.zeros(3), 1).to_event(1)
    ).flatten()
    #     init_weight = numpyro.sample('initial weight', dist.Normal(w_true, 1)).flatten()
    #     with numpyro.handlers.reparam(config={'initial weight': TransformReparam()}):
    #         init_weight = numpyro.sample(
    #             'initial weight',
    #             dist.TransformedDistribution(
    #                 dist.Normal(jnp.zeros(3), 1).to_event(1),
    #                 dist.transforms.AffineTransform(w_true, 1),
    #             ),
    #         ).flatten()

    #     alpha = numpyro.sample('learning rate', dist.HalfNormal(jnp.ones(3))).flatten()
    alpha = numpyro.sample(
        "learning rate", dist.HalfNormal(jnp.ones(3) * 10).to_event(1)
    )
    #     alpha_stim = numpyro.sample('learning rate multiplier', dist .HalfNormal(jnp.zeros(3), 1))
    with numpyro.handlers.reparam(
        config={"learning rate multiplier": TransformReparam()}
    ):
        alpha_stim = numpyro.sample(
            "learning rate multiplier",
            dist.TransformedDistribution(
                dist.HalfNormal(jnp.ones(3)).to_event(1),
                dist.transforms.AffineTransform(1, 1),
            ),
        )

    def transition(weight_prev, xs):
        x, stim = xs
        logits = x[0] * weight_prev[0] + x[1] * weight_prev[1] + weight_prev[2]
        prob_approach = jax.nn.sigmoid(logits)
        delta = (
            w_true + stim * w_stim - weight_prev
        )  # * (x**2) #The multiplication with x**2 is done under the assumption of a MSE between utilities
        learning_rate = alpha * (alpha_stim ** stim)
        weight_prev = numpyro.deterministic(
            "stimulated_weights", weight_prev + prob_approach * learning_rate * delta
        )
        #         with numpyro.handlers.reparam(config={"stimulated_weights": TransformReparam()}):
        #             weight_prev = numpyro.sample(
        #                 "stimulated_weights",
        #                 dist.TransformedDistribution(
        #                     dist.Normal(jnp.zeros(3), 1),
        #                     dist.transforms.AffineTransform(weight_prev + prob_approach * learning_rate * delta, prob_approach * 0.1),
        #                 ),
        #             )
        return weight_prev, (weight_prev, logits)

    _, (weights, logits) = scan(
        transition, init_weight, (X, stim_indicator), length=len(X)
    )
    logits = numpyro.deterministic("logits", logits)
    #     logits = numpyro.deterministic('logits', X[:,0] * weights[:,0] + X[:,1] * weights[:,1] + weights[:,2])
    with numpyro.plate("num_trials", len(X)):
        obs = numpyro.sample("y", dist.BernoulliLogits(logits), obs=y)


def drift_perseverance_model(X, stage, y_prev, y=None):
    init_weight = numpyro.sample(
        "initial_weight", dist.Normal(jnp.zeros(3), 1)
    ).flatten()
    #     repetition_prob = numpyro.sample('repetition_prob', dist.Beta(jnp.ones(3), jnp.ones(3))).flatten()
    repetition_kernel = numpyro.sample(
        "repetition_kernel", dist.Normal(jnp.zeros(3), 1)
    ).flatten()
    #     print(repetition_prob)
    drift = numpyro.sample("drift", dist.Normal(jnp.zeros((3, 3)), 0.1))
    #     with numpyro.handlers.reparam(config={"drift": TransformReparam()}):
    #         drift = numpyro.sample(
    #             "drift",
    #             dist.TransformedDistribution(
    #                 dist.Normal(jnp.zeros((3, 3)), 1),
    #                 dist.transforms.AffineTransform(0, 0.01),
    #             ),
    #         )
    def transition(weight_prev, xs):
        x_curr, stage_curr, y_prev = xs
        #         weight_prev = numpyro.sample('stimulated_weights', dist.Normal(weight_prev + drift[stim_curr], 0.1)).flatten()
        with numpyro.handlers.reparam(
            config={"stimulated_weights": TransformReparam()}
        ):
            weight_curr = numpyro.sample(
                "stimulated_weights",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(3), 1),
                    dist.transforms.AffineTransform(
                        weight_prev + drift[stage_curr], 0.01
                    ),
                ),
            )

        #         prob_x = jax.nn.sigmoid(x_curr[0] * weight_curr[0] + x_curr[1] * weight_curr[1] + weight_curr[2])
        #         prob_x = jax.nn.sigmoid(x_curr[0] * weight_curr[0] + x_curr[1] * weight_curr[1] + weight_curr[2] + y_prev.astype(float) * repetition_kernel[stage_curr])
        #         print(prob_x)
        #         if y_prev == -1:
        #             prob = numpyro.deterministic('probs', prob_x)
        #         else:
        #         prob = numpyro.deterministic('probs', prob_x * (1 - repetition_prob[stage_curr]) + repetition_prob[stage_curr] * y_prev)
        #         prob = numpyro.deterministic('probs', prob_x)
        #         print(prob)
        return weight_curr, (weight_curr)

    _, (weights) = scan(transition, init_weight, (X, stage, y_prev), length=len(X))
    #     print(probs.shape)
    autocorrelation = numpyro.deterministic(
        "autocorrelation", y_prev * repetition_kernel[stage]
    )
    logits = numpyro.deterministic(
        "logits",
        X[:, 0] * weights[:, 0]
        + X[:, 1] * weights[:, 1]
        + weights[:, 2]
        + autocorrelation,
    )
    with numpyro.plate("num_trials", len(X)):
        obs = numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)
