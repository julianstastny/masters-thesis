import arviz as az
import numpyro
import numpyro.distributions as dist
import jax.random as random
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import jax.numpy as jnp

from arviz.stats import stats_utils as azsu


from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.distributions.util import is_identically_one, sum_rightmost
from numpyro.handlers import condition, replay, seed, substitute, trace
from numpyro.infer.initialization import init_to_uniform, init_to_value
from numpyro.util import not_jax_tracer, soft_vmap, while_loop

def log_likelihood(
    model, posterior_samples, *args, parallel=False, batch_ndims=1, **kwargs
):
    """
    (EXPERIMENTAL INTERFACE) Returns log likelihood at observation nodes of model,
    given samples of all latent variables.

    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param args: model arguments.
    :param batch_ndims: the number of batch dimensions in posterior samples. Some usages:

        + set `batch_ndims=0` to get log likelihoods for 1 single sample

        + set `batch_ndims=1` to get log likelihoods for `posterior_samples`
          with shapes `(num_samples x ...)`

        + set `batch_ndims=2` to get log likelihoods for `posterior_samples`
          with shapes `(num_chains x num_samples x ...)`

    :param kwargs: model kwargs.
    :return: dict of log likelihoods at observation sites.
    """

    def single_loglik(samples):
        substituted_model = (
            substitute(model, samples) if isinstance(samples, dict) else model
        )
        model_trace = trace(substituted_model).get_trace(*args, **kwargs)
        return {
            name: site["fn"].log_prob(site["value"])
            for name, site in model_trace.items()
            if site["type"] == "sample" and site["is_observed"]
        }

    prototype_site = batch_shape = None
    for name, sample in posterior_samples.items():
        if batch_shape is not None and jnp.shape(sample)[:batch_ndims] != batch_shape:
            raise ValueError(
                f"Batch shapes at site {name} and {prototype_site} "
                f"should be the same, but got "
                f"{sample.shape[:batch_ndims]} and {batch_shape}"
            )
        else:
            prototype_site = name
            batch_shape = jnp.shape(sample)[:batch_ndims]

    if batch_shape is None:  # posterior_samples is an empty dict
        batch_shape = (1,) * batch_ndims
        posterior_samples = np.zeros(batch_shape)

    batch_size = int(np.prod(batch_shape))
    chunk_size = batch_size if parallel else 1
    return soft_vmap(single_loglik, posterior_samples, len(batch_shape), chunk_size)


class NumPyroSamplingWrapper(az.SamplingWrapper):
    def __init__(self, model, **kwargs):        
        self.model_fun = model.sampler.model
        self.rng_key = kwargs.pop("rng_key", random.PRNGKey(0))
        
        super(NumPyroSamplingWrapper, self).__init__(model, **kwargs)
        
    def log_likelihood__i(self, excluded_obs, idata__i):
        """
        idata__i: The idata object resulting from fitting the model with data__i
        excluded_obs: The observation that was excluded from fitting
        """
#         samples = {
#             key: values.values.reshape((-1, *values.values.shape[2:]))
#             for key, values
#             in idata__i.posterior.items()
#         }
#         print(samples['AR(1)'].shape)
#         log_likelihood_dict = numpyro.infer.log_likelihood(
#             self.model_fun, samples
#         )        
#         idata.observed_data['y'][excluded_obs['idx']] = excluded_obs['y']
        log_likelihood_dict = numpyro.infer.log_likelihood(
            self.model_fun, idata__i, **excluded_obs['data']
        )
        log_likelihood_dict['y'] = log_likelihood_dict['y'][:,excluded_obs['idx']]
#         print(log_likelihood_dict)
#         print(log_likelihood_dict['y'].shape)
        if len(log_likelihood_dict) > 1:
            raise ValueError("multiple likelihoods found")
        data = {}
#         nchains = idata__i.posterior.dims["chain"]
#         ndraws = idata__i.posterior.dims["draw"]]
        ndraws = log_likelihood_dict['y'].shape[0]
        for obs_name, log_like in log_likelihood_dict.items():
#             shape = (nchains, ndraws) + log_like.shape[1:]
            shape = (1, ndraws, 1)
            data[obs_name] = np.reshape(log_like.copy(), shape)
        return az.dict_to_dataset(data)[obs_name]
    
    def sample(self, modified_observed_data):
        self.rng_key, subkey = random.split(self.rng_key)
        mcmc = MCMC(**self.sample_kwargs)
        mcmc.run(subkey, **modified_observed_data)
        mcmc.print_summary(exclude_deterministic=True)
        return mcmc

    def get_inference_data(self, mcmc):
#         idata = az.from_numpyro(mcmc, **self.idata_kwargs)
        idata = mcmc.get_samples()
        return idata
    
class OnPolicyModelWrapper(NumPyroSamplingWrapper):
    def sel_observations(self, idx):
        if not isinstance(idx, int):
            assert len(idx) == 1
            idx = idx[0]
        X_data = self.idata_orig.constant_data["X"].values
        stage_data = self.idata_orig.constant_data["stage"].values
        ydata = self.idata_orig.observed_data["y"].values
#         mask = np.expand_dims(np.isin(np.arange(len(X_data)), idx, invert=True), 1)
#         mask = np.isin(np.arange(len(X_data)), idx, invert=True)
#         print(idx)
#         print(mask)
#         print(mask.shape)
#         print(ydata.shape)
#         data__i = {"X": X_data[~mask], "y": ydata[~mask], "stage": stage_data[~mask]}
#         data_ex = {"X": X_data[mask], "y": ydata[mask], "stage": stage_data[mask]}
#         data_ex = {"X": X_data, "y": ydata, "stage": stage_data}
#         print(data__i['y'].shape)
#         print(data_ex['y'].shape)
        ydata_i = ydata.copy()
# # #         print(y_data_i)
# # #         print(idx)
        ydata_i[idx] = -1
            
        data__i = {"X": X_data, "y": ydata_i, "stage": stage_data}
        data_ex = {"idx": idx, "data": {"y": ydata, "X": X_data, "stage": stage_data}}
        return data__i, data_ex
    
def compute_reloo(model, mcmc, **data_kwargs):
    idata_kwargs = {"constant_data": data_kwargs}
    idata = az.from_numpyro(mcmc, **idata_kwargs)
    sample_kwargs = dict(
        sampler=DiscreteHMCGibbs(NUTS(model), modified=True), 
        num_warmup=1000, 
        num_samples=1000, 
        num_chains=2, 
    )    
#     sample_kwargs = dict(
#         sampler=NUTS(model), 
#         num_warmup=1000, 
#         num_samples=1000, 
#         num_chains=1, 
#     )
    numpyro_wrapper = OnPolicyModelWrapper(
        mcmc, 
        rng_key=random.PRNGKey(5),
        idata_orig=idata, 
        sample_kwargs=sample_kwargs, 
        idata_kwargs=idata_kwargs
    )
    loo_orig = az.loo(idata, pointwise=True)
    num_warning = np.sum(loo_orig.pareto_k > 0.7)
    if num_warning > 0:
        print(f"Pareto-k over 0.7 for {num_warning} points.")
        if num_warning <= 3: #Should be 3!!!!!
            loo = az.reloo(numpyro_wrapper, loo_orig=loo_orig)
        else:
            print("Too many points over 0.7, refitting is expensive.")
            loo = loo_orig
    else:
        loo = loo_orig
    return loo

class LFOWrapper(NumPyroSamplingWrapper):
    def sel_observations(self, start_idx, end_idx):
        X_data = self.idata_orig.constant_data["X"].values
        stage_data = self.idata_orig.constant_data["stage"].values
        ydata = self.idata_orig.observed_data["y"].values
#         mask = np.isin(np.arange(len(X_data)), idx)
        mask = (np.arange(len(X_data)) >= start_idx) * (np.arange(len(X_data)) <= end_idx)
        data_before = {"X": X_data[mask], "y": ydata[mask], "stage": stage_data[mask]}
        data_after = {"X": X_data[~mask], "y": ydata[~mask], "stage": stage_data[~mask]}
        print(data_before['y'].shape)
        print(data_after['y'].shape)
        return data_before, data_after
    
def _exact_elpd(model, idata):
    log_likelihood = azsu.get_log_likelihood(idata)
    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
    shape = log_likelihood.shape
    n_samples = shape[-1]
    
    ufunc_kwargs = {"n_dims": 1, "ravel": False} # No idea what this is doing
    kwargs = {"input_core_dims": [["__sample__"]]} # No idea what this is doing
    
    lppd = np.sum(
        azsu.wrap_xarray_ufunc(
            azsu.logsumexp,
            log_likelihood,
            func_kwargs={"b_inv": n_samples},
            ufunc_kwargs=ufunc_kwargs,
            **kwargs,
        ).values
    )


def psis_lfo_cv(mcmc, min_num_observations, num_steps_ahead):
    sampling_wrapper = LFOWrapper(
        mcmc, 
        rng_key=random.PRNGKey(5),
        idata_orig=idata, 
        sample_kwargs=sample_kwargs, 
        idata_kwargs=idata_kwargs
    )
    data_before, data_after = sampling_wrapper.sel_observations(min_num_observations)
    model_star = sampling_wrapper.sample(data_before)
    elpd = _exact_elpd(model_star, )
