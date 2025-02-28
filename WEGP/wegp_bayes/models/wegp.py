import torch
import math
import gpytorch
from gpytorch.constraints import Positive,GreaterThan
from gpytorch.priors import NormalPrior,HalfCauchyPrior
from gpytorch.distributions import MultivariateNormal
from .gpregression import GPR
from .. import kernels
from ..priors.exp_gamma import ExpGammaPrior
from ..priors.CauchyLoc import CauchyLocPrior
from ..priors.HalfCauchyLoc import HalfCauchyLocPrior

from ..priors.mollified_uniform import MollifiedUniformPrior
from ..utils.matrix_ops import translate_and_rotate
from typing import List,Optional
import numpy as np
import itertools
from gpytorch.priors import Prior
from torch.distributions import HalfCauchy

class LVWeighting(gpytorch.Module):
    def __init__(self,num_levels, num_permutations,latents):
        super().__init__()
        self.num_permutations = num_permutations
        if num_levels == 1:
            raise ValueError('Categorical variable has only one level!')
        elif num_levels == 2:
            raise ValueError('Binary categorical variables should not be supplied')
        self.register_buffer('num_levels', torch.tensor(num_levels))      

        self.register_parameter(
            name='raw_alpha',
            parameter=torch.nn.Parameter(torch.tensor(0.1))
        )
        self.register_prior(
            name='raw_alpha_prior',
            prior=HalfCauchyLocPrior(scale=0.1),
            param_or_closure='raw_alpha'
        )
        self.register_constraint(
                param_name=f'raw_alpha',
                constraint=Positive(transform=torch.exp, inv_transform=torch.log)
            )
        self.register_parameter(
            name='raw_weights',
            parameter=torch.nn.Parameter(torch.distributions.HalfCauchy(0.1).sample(sample_shape=[1,num_permutations]))#这里的sample出的不是一个数，是[1,num_permutations]要怎么改
        )
        # self.register_parameter(
        #     name='raw_weights',
        #     parameter=torch.nn.Parameter(torch.distributions.HalfCauchy(self.raw_alpha).sample(sample_shape=[1,num_permutations]))#这里的sample出的不是一个数，是[1,num_permutations]要怎么改
        # )
        self.register_prior(
            name='raw_weights_prior',
            prior=HalfCauchyLocPrior(scale=self.raw_alpha),
            param_or_closure='raw_weights'
        )
        self.register_constraint(
                param_name=f'raw_weights',
                constraint=Positive(transform=torch.exp, inv_transform=torch.log)
            )
        
        if latents is not None:
            self.latents = latents
        else:
            print("latent generation is wrong, please check")
            exit()
            self.latents = self._generate_latents()
    def _generate_latents(self):
        permutations = self.generate_full_rank_permutations()
        # print("permutations", permutations)
        perm_tensor = torch.tensor(permutations).T.float()
        return perm_tensor
    def generate_full_rank_permutations(self):
#         """Generate a set of full-rank permutations."""
        all_permutations = list(itertools.permutations(range(self.num_levels)))
        permutations = self.sample_permutations(all_permutations)
        while True:
            permutations = self.sample_permutations(all_permutations)
            if self.is_full_rank(permutations):
                return permutations
    def sample_permutations(self, all_permutations):

        """Randomly sample a set of permutations."""

        selected_indices = np.random.choice(len(all_permutations), self.num_permutations, replace=False)
        selected_permutations = [all_permutations[i] for i in selected_indices]

        return selected_permutations


    def is_full_rank(self, permutations):
        """Check if the permutations form a full-rank distance matrix."""
        num_distances = self.num_levels * (self.num_levels - 1) // 2
        distance_matrix = torch.zeros((num_distances, self.num_permutations))
        
        for i, perm in enumerate(permutations):
            distances = [abs(perm[j] - perm[k]) for j in range(self.num_levels) for k in range(j + 1, self.num_levels)]
            distance_matrix[:, i] = torch.tensor(distances)
        return torch.linalg.matrix_rank(distance_matrix) >= min(distance_matrix.size())
    @property
    def weighted_latents(self):
        weights = self.raw_weights
        weighted_latents = self.latents * weights

        return weighted_latents
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """Map the levels of the qualitative factor onto the latent variable space.

        :param x: 1D tensor of levels (which need to be encoded as integers) of size N
        :type x: torch.LongTensor

        :returns: a N tensor with weights
        :rtype: torch.Tensor
        """
        weighted_latents = self.weighted_latents
        if weighted_latents.ndim == 3:
            return torch.stack([
                torch.nn.functional.embedding(x[i,:],weighted_latents[i,...]) \
                    for i in range(weighted_latents.shape[0])
            ])
        return torch.nn.functional.embedding(x, weighted_latents)


class WEGP(GPR):
    """The weighted Bayesian ordinal regression model.

    This is based on the work of `Zhang et al. (2019)`_. WEBO extends GPs to handle ordinal 
    categorical inputs. WEBO first projects each ordinal input onto a numerical latent variable 
    space, which can then be used with standard GP kernels for numerical inputs. These latent 
    variables are jointly estimated along with the other GP hyperparameters.

    :param train_x: The training inputs (size N x d). Qualitative inputs needed to be encoded as 
        integers 0,...,L-1 where L is the number of levels. For best performance, sum the weighted
        numerical variables.
    :type train_x: torch.Tensor
    :param train_y: The training targets (size N)
    :type train_y: torch.Tensor
    :param num_levels_per_var: List specifying the number of levels for each ordinal variable.
        The order should correspond to the one specified in `qual_index`. This list cannot be empty.
    :type num_levels_per_var: List[int]
    :param num_permutations: The dimension of the latent variable space for each ordinal input. Defaults to n(n-1)/2.
    :param quant_correlation_class: A string specifying the kernel for the quantitative inputs. Needs
        to be one of the following strings - 'RBFKernel' (radial basis kernel), 'Matern52Kernel' (twice 
        differentiable Matern kernel), 'Matern32Kernel' (first order differentiable Matern
        kernel). The generate kernel uses a separate lengthscale for each input variable. Defaults to
        'RBFKernel'.
    :type quant_correlation_class: str, optional
    :param noise: The (initial) noise variance.
    :type noise: float, optional
    :param fix_noise: Fixes the noise variance at the current level if `True` is specifed.
        Defaults to `False`
    :type fix_noise: bool, optional
    :param lb_noise: Lower bound on the noise variance. Setting a higher value results in
        more stable computations, when optimizing noise variance, but might reduce 
        prediction quality. Defaults to 1e-6
    :type lb_noise: float, optional

    .. _Zhang et al. (2019):
        https://doi.org/10.
    """

    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        qual_index:List[int],
        quant_index:List[int],
        num_levels_per_var:List[int],
        num_permutations:List[int],
        latents_list:List[torch.Tensor],
        quant_correlation_class:str='Matern32Kernel',
        noise:float=1e-4,
        fix_noise:bool=True,
        lb_noise:float=1e-6,
    ) -> None:
        num_active = 0
        # print("num_permutations", num_permutations)# [3, 10]
        # print("num_levels_per_var", num_levels_per_var)# [3, 5]
        # print("qual_index", qual_index)# [2, 3]
        for num_per in num_permutations:
            num_active = num_active + num_per
            qual_kernel = kernels.Matern32Kernel(
            active_dims=torch.arange(num_active)
        )#TODO:这里要改，不同的qual_index，num_permutations不一样
        # print("num_active", num_active) #12
        qual_kernel.initialize(**{'lengthscale':1.0})
        qual_kernel.raw_lengthscale.requires_grad_(False)

        if len(quant_index) == 0:
            correlation_kernel = qual_kernel
        else:
            try:
                quant_correlation_class = getattr(kernels,quant_correlation_class)
            except:
                raise RuntimeError(
                    "%s not an allowed kernel" % quant_correlation_class
                )
            quant_kernel = quant_correlation_class(
                ard_num_dims=len(quant_index),
                active_dims=num_active+torch.arange(len(quant_index)),
                lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
            )
            quant_kernel.register_prior(
                'raw_lengthscale_prior',MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale',
            )
            correlation_kernel = qual_kernel*quant_kernel

        super(WEGP,self).__init__(
            train_x=train_x,train_y=train_y,
            correlation_kernel=correlation_kernel,
            noise=noise,fix_noise=fix_noise,lb_noise=lb_noise
        )

        # register index and transforms
        self.register_buffer('num_levels_per_var',torch.tensor(num_levels_per_var))
        self.register_buffer('num_permutations',torch.tensor(num_permutations))
        self.register_buffer('quant_index',torch.tensor(quant_index))
        self.register_buffer('qual_index',torch.tensor(qual_index))
        # latent variable weighting
        self.lv_weighting_layers = torch.nn.ModuleList([
            LVWeighting(num_levels,num_permutations=num_permutations[k],latents=latents_list[k]) \
                for k,num_levels in enumerate(num_levels_per_var)
        ])
        # print("lv_weighting_layers", self.lv_weighting_layers)
    
    def forward(self,x:torch.Tensor) -> MultivariateNormal:
        # x.shape = (n_train, n_cat + n_quant) (80, 8)
        # n_cat = 1, n_quant = 6
        embeddings = []
        for i,e in enumerate(self.lv_weighting_layers):
            embeddings.append(e(x[...,self.qual_index[i]].long()))#在这里做embedding，输出一下embedding是否正确
        embeddings = torch.cat(embeddings,-1)  # (n_train, latent_dim) (80, 2)
        if len(self.quant_index) > 0:  # 6
            x = torch.cat([embeddings,x[...,self.quant_index]],-1)  # (n_train, latent_dim + n_quant) (80, 8)
        else:
            x = embeddings
        mean_x = self.mean_module(x) 
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x,covar_x)
    
    def named_hyperparameters(self):
        """Return all hyperparameters other than the latent variables

        This method is useful when different learning rates to the latent variables. To 
        include the latent variables along with others use `.named_parameters` method
        """
        #TODO: 这里要改 但是没用到 可以先不管
        for name, param in self.named_parameters():
            print("name: ", name)
            if "lv_weighting" not in name:
                yield name, param
    def to_pyro_random_module(self):
        new_module = super().to_pyro_random_module()
        # some modules are not registered as Pyro modules
        if isinstance(self.covar_module.base_kernel,gpytorch.kernels.ProductKernel):
            new_module.covar_module.base_kernel.kernels[1] = \
                new_module.covar_module.base_kernel.kernels[1].to_pyro_random_module()
        for i,layer in enumerate(new_module.lv_weighting_layers):
            new_module.lv_weighting_layers[i]= layer.to_pyro_random_module()

        return new_module
    