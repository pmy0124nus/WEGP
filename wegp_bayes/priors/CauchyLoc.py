import torch
import math
from torch.distributions import constraints, TransformedDistribution, AbsTransform
from torch.nn import Module as TModule
class CauchyLoc(TransformedDistribution):
    """
    Creates a Cauchy distribution parameterized by `loc` and `scale` where:
        X ~ Cauchy(loc, scale)
    Example:
        >>> m = CauchyLoc(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # cauchy distributed with loc=1, scale=1
        tensor([ 2.3214])
    Args:
        loc (float or Tensor): location parameter of the Cauchy distribution
        scale (float or Tensor): scale parameter of the Cauchy distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = torch.distributions.Cauchy(loc, scale, validate_args=False)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(CauchyLoc, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        return torch.full(self._extended_shape(), math.nan, dtype=self.scale.dtype, device=self.scale.device)

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return torch.full(self._extended_shape(), math.inf, dtype=self.scale.dtype, device=self.scale.device)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = torch.as_tensor(value, dtype=self.base_dist.scale.dtype,
                                device=self.base_dist.scale.device)
        log_prob = self.base_dist.log_prob(value)
        log_prob = torch.where(value >= self.loc, log_prob, -math.inf)
        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.base_dist.cdf(value)

    def icdf(self, prob):
        return self.base_dist.icdf(prob)

    def entropy(self):
        return self.base_dist.entropy()

class CauchyLocPrior(TModule, CauchyLoc):
    """
    Cauchy prior with specified local parameter.
    """

    def __init__(self, loc, scale, validate_args=None, transform=None):
        TModule.__init__(self)
        CauchyLoc.__init__(self, loc=loc, scale=scale, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        return CauchyLocPrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape))
    

    def log_prob(self, value):
        epsilon = 1e-6
        value = torch.clamp(value, self.loc + epsilon, self.loc + self.scale - epsilon)
        return -torch.log(self.scale * (1 + ((value - self.loc) / self.scale) ** 2))


