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

import arviz as az

import numpy as np

def fit(model, num_chains, num_warmup=1000, num_samples=1000, rng_seed=0, **kwargs):
    assert set(kwargs.keys()) <= set(model.__code__.co_varnames), model.__code__.co_varnames
    assert ('X' in kwargs.keys()) and ('y' in kwargs.keys())
    nuts_kernel = NUTS(model, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_chains=num_chains, num_warmup=num_warmup, num_samples=num_samples)
    rng_key = jax.random.PRNGKey(rng_seed)
    mcmc.run(rng_key, **kwargs)
    return mcmc, az.from_numpyro(mcmc)



def constant_model(X, y=None):
    w = numpyro.sample('initial_weight', dist.Normal(jnp.zeros((3, 1)), 10))
    logits = X[:,0] * w[0] + X[:,1] * w[1] + w[2]
    with numpyro.plate('num_trials', len(X)):
        obs = numpyro.sample('y', dist.BernoulliLogits(logits=logits), obs=y)

def additive_effect_model(X, stim_durations, y=None):
    weights = numpyro.sample('weights', dist.Normal(jnp.zeros(3), 10))
    stim_effect = numpyro.sample('stim_effect', dist.Normal(jnp.zeros((2, 3)), jnp.ones((2, 3))*10))
    stimulated_weights = numpyro.deterministic(
        'stimulated_weights', weights + stim_effect[0] * jnp.log(jnp.tile(stim_durations[:,0], (3, 1)).T + 1) + stim_effect[1] * jnp.log(jnp.tile(stim_durations[:,1], (3, 1)).T + 1))
    logits = X[:,0] * stimulated_weights[:,0] + X[:,1] * stimulated_weights[:,1] + stimulated_weights[:,2]
    with numpyro.plate('num_trials', len(y)):
        obs = numpyro.sample('y', dist.BernoulliLogits(logits=logits), obs=y)

def regularized_random_walk_model(X, y=None):
    X = numpyro.deterministic('X', X)
    init_w = numpyro.sample('initial_weight', dist.Normal(jnp.zeros((3, 1)), 10))
    with numpyro.handlers.reparam(config={"weight_walk": TransformReparam()}):
        w = numpyro.sample(
            "weight_walk",
            dist.TransformedDistribution(
                dist.GaussianRandomWalk(scale=np.ones(3), num_steps=len(y)),
                dist.transforms.AffineTransform(0, 0.1),
            ),
        )
    stimulated_weights = numpyro.deterministic('stimulated_weights', (w + init_w).T)
    logits = X[:,0] * stimulated_weights[:,0] + X[:,1] * stimulated_weights[:,1] + stimulated_weights[:,2]
    with numpyro.plate('num_trials', len(y)):
        obs = numpyro.sample('y', dist.BernoulliLogits(logits=logits), obs=y)


def simple_mechanistic_model(X, stage, y=None):
    init_weight = numpyro.sample('initial_weight', dist.Normal(jnp.zeros(3), 1)).flatten()
    drift = numpyro.sample('drift', dist.Normal(jnp.zeros((3, 3)), 0.01))
#     with numpyro.handlers.reparam(config={"drift": TransformReparam()}):
#         drift = numpyro.sample(
#             "drift",
#             dist.TransformedDistribution(
#                 dist.Normal(jnp.zeros((3, 3)), 1),
#                 dist.transforms.AffineTransform(0, 0.01),
#             ),
#         )
    def transition(weight_prev, stage_curr):
#         weight_prev = numpyro.sample('stimulated_weights', dist.Normal(weight_prev + drift[stim_curr], 0.1)).flatten()
        with numpyro.handlers.reparam(config={"stimulated_weights": TransformReparam()}):
            weight_prev = numpyro.sample(
                "stimulated_weights",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(3), 1),
                    dist.transforms.AffineTransform(weight_prev + drift[stage_curr], 0.01),
                ),
            )
        return weight_prev, (weight_prev)

    _, (weights) = scan(transition, init_weight, stage, length=len(X))
    logits = numpyro.deterministic('logits', X[:,0] * weights[:,0] + X[:,1] * weights[:,1] + weights[:,2])
    with numpyro.plate('num_trials', len(X)):
        obs = numpyro.sample('y', dist.BernoulliLogits(logits=logits), obs=y)

def latent_utility_model_nostim(X, y=None):
    X = np.concatenate((X, np.ones((len(X), 1))), 1)
    w_true = numpyro.sample('true utility-function', dist.Normal(jnp.zeros(3), 1)).flatten()
#     init_weight = numpyro.sample('initial weight', dist.Normal(jnp.zeros(3), 1)).flatten()
#     init_weight = numpyro.sample('initial weight', dist.Normal(w_true, 1)).flatten()
    with numpyro.handlers.reparam(config={'initial weight': TransformReparam()}):
        init_weight = numpyro.sample(
            'initial weight',
            dist.TransformedDistribution(
                dist.Normal(jnp.zeros(3), 1),
                dist.transforms.AffineTransform(w_true, 1),
            ),
        ).flatten()
#     alpha = numpyro.sample('learning rate', dist.HalfNormal(jnp.ones(3))).flatten()
    alpha = numpyro.sample('learning rate', dist.HalfNormal())
    def transition(weight_prev, x):
        prob_approach = jax.nn.sigmoid(x[0] * weight_prev[0] + x[1] * weight_prev[1] + weight_prev[2])
        delta = (w_true - weight_prev) * (x**2) #The multiplication with x**2 is done under the assumption of a MSE between utilities
        with numpyro.handlers.reparam(config={"stimulated_weights": TransformReparam()}):
            weight_prev = numpyro.sample(
                "stimulated_weights",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(3), 1),
                    dist.transforms.AffineTransform(weight_prev + prob_approach * alpha * delta, 0.01),
                ),
            )
        return weight_prev, (weight_prev, prob_approach)
    _, (weights, probs) = scan(transition, init_weight, X, length=len(X))
#     logits = numpyro.deterministic('logits', X[:,0] * weights[:,0] + X[:,1] * weights[:,1] + weights[:,2])
    with numpyro.plate('num_trials', len(X)):
        obs = numpyro.sample('y', dist.Bernoulli(probs), obs=y)


def latent_utility_model_stim(X, stim_indicator, y=None):
    X = np.concatenate((X, np.ones((len(X), 1))), 1)
    w_true = numpyro.sample('true utility-function', dist.Normal(jnp.zeros(3), 1)).flatten()
    w_stim = numpyro.sample('stimulated utility-function delta', dist.Normal(jnp.zeros(3), 1)).flatten()
#     init_weight = numpyro.sample('initial weight', dist.Normal(jnp.zeros(3), 1)).flatten()
#     init_weight = numpyro.sample('initial weight', dist.Normal(w_true, 1)).flatten()
    with numpyro.handlers.reparam(config={'initial weight': TransformReparam()}):
        init_weight = numpyro.sample(
            'initial weight',
            dist.TransformedDistribution(
                dist.Normal(jnp.zeros(3), 1),
                dist.transforms.AffineTransform(w_true, 1),
            ),
        ).flatten()
#     alpha = numpyro.sample('learning rate', dist.HalfNormal(jnp.ones(3))).flatten()
    alpha = numpyro.sample('learning rate', dist.HalfNormal())
#     alpha_stim = numpyro.sample('learning rate multiplier', dist .HalfNormal(jnp.zeros(3), 1))
    with numpyro.handlers.reparam(config={'learning rate multiplier': TransformReparam()}):
        alpha_stim = numpyro.sample(
            'learning rate multiplier',
            dist.TransformedDistribution(
                dist.HalfNormal(jnp.ones(3)),
                dist.transforms.AffineTransform(1, 1),
            ),
        )
    def transition(weight_prev, xs):
        x, stim = xs
        logits = x[0] * weight_prev[0] + x[1] * weight_prev[1] + weight_prev[2]
        prob_approach = jax.nn.sigmoid(logits)
        delta = (w_true + stim * w_stim - weight_prev) * (x**2) #The multiplication with x**2 is done under the assumption of a MSE between utilities
        learning_rate = alpha * (alpha_stim**stim)
        with numpyro.handlers.reparam(config={"stimulated_weights": TransformReparam()}):
            weight_prev = numpyro.sample(
                "stimulated_weights",
                dist.TransformedDistribution(
                    dist.Normal(jnp.zeros(3), 1),
                    dist.transforms.AffineTransform(weight_prev + prob_approach * learning_rate * delta, 0.001),
                ),
            )
        return weight_prev, (weight_prev, logits)
    _, (weights, logits) = scan(transition, init_weight, (X, stim_indicator), length=len(X))
    logits = numpyro.deterministic('logits', logits)
#     logits = numpyro.deterministic('logits', X[:,0] * weights[:,0] + X[:,1] * weights[:,1] + weights[:,2])
    with numpyro.plate('num_trials', len(X)):
        obs = numpyro.sample('y', dist.BernoulliLogits(logits), obs=y)