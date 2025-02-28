import math
from gpytorch.priors import Prior
import torch
from torch import inf
from torch.distributions import constraints
from torch.distributions.transforms import AbsTransform
from torch.distributions.cauchy import Cauchy
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.nn import Module as TModule
__all__ = ['HalfLocCauchy']

class HalfLocCauchy(TransformedDistribution):
    r"""
    Creates a half-Cauchy distribution parameterized by `scale` where::

        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = HalfCauchy(torch.tensor([1.0]))
        >>> m.sample()  # half-cauchy distributed with scale=1
        tensor([ 2.3214])

    Args:
        scale (float or Tensor): scale of the full Cauchy distribution
    """


    has_rsample = True

    def __init__(self, scale, validate_args=None):
        base_dist = Cauchy(0, scale, validate_args=False)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(HalfLocCauchy, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        return torch.full(self._extended_shape(), math.inf, dtype=self.scale.dtype, device=self.scale.device)

    @property
    def mode(self):
        return torch.zeros_like(self.scale)

    @property
    def variance(self):
        return self.base_dist.variance

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = torch.as_tensor(value, dtype=self.base_dist.scale.dtype,
                                device=self.base_dist.scale.device)
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob = torch.where(value >= 0, log_prob, -inf)
        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob):
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self):
        return self.base_dist.entropy() - math.log(2)


class HalfCauchyLocPrior(Prior, HalfLocCauchy):
    """
    Half-Cauchy prior.
    """

    def __init__(self, scale, validate_args=None, transform=None):
        TModule.__init__(self)
        HalfLocCauchy.__init__(self, scale=scale, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        return HalfCauchyLocPrior(self.scale.expand(batch_shape))
    
    def log_prob(self, value):
        epsilon = 1e-6
        value = torch.clamp(value, epsilon, float('inf'))
        return -torch.log(self.scale * (1 + (value / self.scale) ** 2))