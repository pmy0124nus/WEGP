import torch
import numpy as np
import math
from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_sample,
    init_to_value
)

from .numpryo_dists import MollifiedUniform

from wegp_bayes.models import WEGP

from copy import deepcopy

def cov_map(cov_func, xs, xs2=None):
    """Compute a covariance matrix from a covariance function and data points.
    Args:
      cov_func: callable function, maps pairs of data points to scalars.
      xs: array of data points, stacked along the leading dimension.
    Returns:
      A 2d array `a` such that `a[i, j] = cov_func(xs[i], xs[j])`.
    """
    if xs2 is None:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T
    
def rbfkernel(x1, x2):
    return jnp.exp(-0.5*jnp.sum((x1 - x2)**2))

def matern52kernel(x1,x2):
    r = jnp.sqrt(jnp.sum((x1 - x2)**2) + 1e-12)
    exp_component = jnp.exp(-math.sqrt(5)*r)
    constant_component =1 + math.sqrt(5)*r + 5/3*(r**2)
    return constant_component*exp_component

# define global dictionary of kernels
kernel_names = {
    'rbfkernel':rbfkernel,
    'matern52kernel':matern52kernel
}

class ExpHalfCauchy(dist.TransformedDistribution):
    def __init__(self,scale):
        
        base_dist = dist.HalfCauchy(scale)
        super().__init__(
            base_dist,dist.transforms.ExpTransform().inv
        )

def get_samples(samples,num_samples=None, group_by_chain=False):
    """
    Get samples from the MCMC run
    :param int num_samples: Number of samples to return. If `None`, all the samples
        from an MCMC chain are returned in their original ordering.
    :param bool group_by_chain: Whether to preserve the chain dimension. If True,
        all samples will have num_chains as the size of their leading dimension.
    :return: dictionary of samples keyed by site name.
    """
    if num_samples is not None:
        batch_dim = 0
        sample_tensor = list(samples.values())[0]
        batch_size, device = sample_tensor.shape[batch_dim], sample_tensor.device
        idxs = torch.linspace(0,batch_size-1,num_samples,dtype=torch.long,device=device).flip(0)
        samples = {k: v.index_select(batch_dim, idxs) for k, v in samples.items()}
    return samples

def run_hmc_numpyro_wegp(
    model,
    latents_list,#这个传进来，是因为model里面没有latents_list这个参数，其他model里有的，比如qual和quant_index，比如num_levels_per_var，就直接读就行了
    num_samples:int=500,
    warmup_steps:int=500,
    num_model_samples:int=100,
    disable_progbar:bool=True,
    num_chains:int=1,
    num_jobs:int=1,
    max_tree_depth:int=5,
    initialize_from_state:bool=False,
    seed:int=0,
):
    kwargs = {
        'x':jnp.array(model.train_inputs[0].numpy()),
        'y':jnp.array(model.train_targets.numpy()),
        
    }
    dense_mass=False

    if isinstance(model, WEGP):
        numpyro_model = numpyro_wegp
        kwargs.update({
            'latents_list':latents_list,
            'qual_index':model.qual_index.tolist(),
            'quant_index':model.quant_index.tolist(),
            'num_levels_per_var':model.num_levels_per_var.tolist(),
            'num_permutations':model.num_permutations.tolist(),
            'jitter':model.likelihood.noise_covar.raw_noise_constraint.lower_bound.item()
        })
    
    # else:
        #raise error
    #初始化参数,选init_to_value或init_to_sample,默认选第二种，那这里就不需要改
    if initialize_from_state:
        init_values = {}
        for name,module,_,closure,_ in model.named_priors():
            init_values[name[:-6]] = jnp.array(closure(module).detach().clone().numpy())
        init_strategy = init_to_value(values=init_values)
    else:
        init_strategy = init_to_sample
    
    kernel = NUTS(
        numpyro_model,
        step_size=0.1,
        adapt_step_size=True,
        init_strategy=init_strategy,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass
    )

    mcmc_runs = MCMC(
        kernel,
        num_warmup=warmup_steps,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar= (not disable_progbar),
        chain_method='sequential',#Alternatives are 'parallel'
        #jit_model_args=True
    )

    mcmc_runs.run(random.PRNGKey(seed),**kwargs)
    samples = {
        k:torch.from_numpy(np.array(v)).to(model.train_targets) for k,v in mcmc_runs.get_samples().items()
    }
    samples = {k:v for k,v in get_samples(samples,num_model_samples).items()}

    model.train_inputs = tuple(tri.unsqueeze(0).expand(num_model_samples, *tri.shape) for tri in model.train_inputs)
    model.train_targets = model.train_targets.unsqueeze(0).expand(num_model_samples, *model.train_targets.shape)
    # print(model.state_dict().keys())
    state_dict = deepcopy(model.state_dict())
    state_dict.update(samples)

    model.load_strict_shapes(False)
    model.load_state_dict(state_dict)

    return mcmc_runs

########
# Numpyro models
########


##WEGP model
def numpyro_wegp(
    x,y,latents_list,qual_index,quant_index,num_levels_per_var,num_permutations,jitter=1e-6,
):  #定义pyro模型 1.定义先验分布 2.定义模型 3.定义观测数据
    mean = numpyro.sample('mean_module.raw_constant',dist.Normal(0,1))
    outputscale = numpyro.sample("covar_module.raw_outputscale", dist.Normal(0.0, 1)) #outputscale就是kernel variance
    noise = numpyro.sample(
        "likelihood.noise_covar.raw_noise",
        ExpHalfCauchy(1e-2).expand([1])
    )#TODO：mean output noise不太确定对不对
    num_qual = len(qual_index)
    #alpha是参数，每一层的alpha要不一样！因为每一层的categorical variable的categories个数不一样
    alpha = [
        numpyro.sample(
            'lv_weighting_layers.%d.raw_alpha'%i, #d会被替换成i
            dist.HalfCauchy(
                1e-2
            ).expand([1])
        ) for i in range(num_qual)
    ]
    #weights是参数
    weights =[
        numpyro.sample(
            'lv_weighting_layers.%d.raw_weights'%i,
            dist.HalfCauchy(
                alpha[i]
            ).expand([1,num_permutations[i]])
        ) for i in range(num_qual)
    ]
    latents_list = [jnp.array(latent.cpu().numpy()) if isinstance(latent, torch.Tensor) else latent for latent in latents_list]

    #latents要在这个模型外面生成，然后传进来
    #对于第i个qual变量，latent在latent_list[i]中
    x2 = jnp.column_stack([
        jnp.take(
            latents_list[i],x[:,qual_index[i]].astype(jnp.int32),axis=0 #提取第latents[i]行的第x[:,qual_index[i]]列
        ) for i in range(num_qual) 
    ])
    weights_combined = jnp.concatenate([w for w in weights], axis=1)

    x2_weighted = x2 * weights_combined 
    #处理数量特征
    if len(quant_index) > 0:
        lengthscale = numpyro.sample(
            'covar_module.base_kernel.kernels.1.raw_lengthscale',
            MollifiedUniform(math.log(0.1),math.log(10)).expand([1,len(quant_index)])
        )

        x2_quant = x[:,quant_index]/jnp.exp(lengthscale)
        x2 = jnp.column_stack([x2_weighted,x2_quant])
    
    # compute kernel
    k = jnp.exp(outputscale)*cov_map(rbfkernel,x2)
    # add noise and jitter
    k += (jnp.exp(noise)+jitter)*jnp.eye(x.shape[0])
    
    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "y",
        dist.MultivariateNormal(loc=mean*jnp.ones(x.shape[0]), covariance_matrix=k),
        obs=y,
    )

