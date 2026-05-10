import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

class LatentOutput:
    z: Tensor
    kl: Tensor
    kl_raw: Tensor


def _gumbel_rao_jvp(
    D: Tensor,
    pi_D: Tensor,
    grad: Tensor,
    tau: Tensor,
    K: int,
    eps: float = 1e-6,
) -> Tensor:
    """
        theta_D_j + G_j | (D = I_i) =
            -log(E_i),                          if j == i
            -log(E_j / pi_D_j + E_i),           otherwise

    """
    pi_D_safe = pi_D.clamp_min(eps)
    D_bool = D.bool()
    E = torch.empty(
        pi_D.shape + (K,),
        device=pi_D.device,
        dtype=pi_D.dtype,
    )
    E.exponential_()
    # E_i: the exponential at the sampled index, broadcast over channels.
    E_i = (E * D.unsqueeze(-1)).sum(dim=-2)                       # [..., K]

    ratio = E / pi_D_safe.unsqueeze(-1)                            # [..., C, K]
    ratio = ratio.masked_fill(D_bool.unsqueeze(-1), 0.0)
    cond_logits = -(ratio + E_i.unsqueeze(-2) + eps).log()         # [..., C, K]
    pi_GR = (cond_logits / tau).softmax(dim=-2)                    # [..., C, K]

    inner = (pi_GR * grad.unsqueeze(-1)).sum(dim=-2, keepdim=True)
    return (pi_GR * (grad.unsqueeze(-1) - inner)).mean(dim=-1) / tau


class _CategoricalReinMaxCVST(torch.autograd.Function):
    """Hard categorical sample with the ReinMax-CV gradient estimator.

    Wang & Bui (2026), "Beyond ReinMax: Low-Variance Gradient Estimators
    for Discrete Latent Variables" (arXiv:2603.08257).

    Combines ReinMax (Heun midpoint at tau=1) with a Gumbel-Softmax control
    variate centered at ``theta_D = log((pi+D)/2)`` and a Gumbel-Rao
    estimate of its expectation::

        grad = 2*J(pi_D)^T g  -  0.5*J(pi)^T g                       [ReinMax]
             + eta * ( J_GR^T g  -  J_STGS^T g )  at theta_D, tau    [CV]

    Note: the paper's equation 18 drops the leading factor of 2 on the
    first ReinMax term -- this is a typo. The reference implementation and
    the structural derivation in section 3.2 use the full ReinMax expression
    with the CV correction added, which is what we implement here.
    """

    @staticmethod
    def forward(
        ctx,
        logits: Tensor,
        cv_tau: Tensor,
        cv_eta: float,
        mc_samples: int,
    ) -> Tensor:
        probs = logits.float().softmax(dim=-1)
        u = torch.rand(
            (*probs.shape[:-1], 1),
            device=probs.device,
            dtype=probs.dtype,
        )
        sample = (u > probs.cumsum(dim=-1)).sum(dim=-1)
        sample = sample.clamp_max(probs.shape[-1] - 1)
        z = F.one_hot(sample, probs.shape[-1]).to(logits.dtype)

        ctx.save_for_backward(z.float(), probs, cv_tau.float())
        ctx.cv_eta = float(cv_eta)
        ctx.K = int(mc_samples)
        ctx.logits_dtype = logits.dtype
        return z

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        D, pi, tau = ctx.saved_tensors
        eta = ctx.cv_eta
        K = ctx.K
        eps = 1e-6
        grad = grad_output.float()

        pi_D = 0.5 * (pi + D)

        # Standard ReinMax (Heun midpoint, tau = 1).
        grad_rm = 2.0 * pi_D * (grad - (pi_D * grad).sum(dim=-1, keepdim=True))
        grad_rm = grad_rm - 0.5 * pi * (
            grad - (pi * grad).sum(dim=-1, keepdim=True)
        )

        # Single-sample STGS at theta_D, temperature cv_tau.
        log_pi_D = pi_D.clamp_min(eps).log()
        gumbel = -torch.empty_like(pi_D).exponential_().log()
        pi_GS = ((log_pi_D + gumbel) / tau).softmax(dim=-1)
        grad_GS = pi_GS * (
            grad - (pi_GS * grad).sum(dim=-1, keepdim=True)
        ) / tau

        # Gumbel-Rao at theta_D with K conditional samples.
        grad_GR = _gumbel_rao_jvp(D, pi_D, grad, tau, K, eps)

        grad_logits = grad_rm + eta * (grad_GR - grad_GS)
        grad_logits = grad_logits - grad_logits.mean(dim=-1, keepdim=True)
        return grad_logits.to(ctx.logits_dtype), None, None, None


class _CategoricalReinMaxRaoST(torch.autograd.Function):
    """Hard categorical sample with the ReinMax-Rao gradient estimator.


    """

    @staticmethod
    def forward(
        ctx,
        logits: Tensor,
        rao_tau: Tensor,
        mc_samples: int,
    ) -> Tensor:
        probs = logits.float().softmax(dim=-1)
        u = torch.rand(
            (*probs.shape[:-1], 1),
            device=probs.device,
            dtype=probs.dtype,
        )
        sample = (u > probs.cumsum(dim=-1)).sum(dim=-1)
        sample = sample.clamp_max(probs.shape[-1] - 1)
        z = F.one_hot(sample, probs.shape[-1]).to(logits.dtype)

        ctx.save_for_backward(z.float(), probs, rao_tau.float())
        ctx.K = int(mc_samples)
        ctx.logits_dtype = logits.dtype
        return z

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        D, pi, tau = ctx.saved_tensors
        K = ctx.K
        eps = 1e-20
        grad = grad_output.float()

        pi_D = 0.5 * (pi + D)
        grad_GR = _gumbel_rao_jvp(D, pi_D, grad, tau, K, eps)
        grad_st = pi * (grad - (pi * grad).sum(dim=-1, keepdim=True))

        grad_logits = 2.0 * grad_GR - 0.5 * grad_st
        grad_logits = grad_logits - grad_logits.mean(dim=-1, keepdim=True)
        return grad_logits.to(ctx.logits_dtype), None, None


def _categorical_sample(logits: Tensor, temperature: float) -> Tensor:
    """Hard categorical sample for non-straight-through prior/posterior draws."""
    probs = (logits.float() / float(temperature)).softmax(dim=-1)
    u = torch.rand(
        (*probs.shape[:-1], 1),
        device=probs.device,
        dtype=probs.dtype,
    )
    sample = (u > probs.cumsum(dim=-1)).sum(dim=-1)
    sample = sample.clamp_max(probs.shape[-1] - 1)
    return F.one_hot(sample, probs.shape[-1]).to(logits.dtype)


class CategoricalReinMaxMapper(nn.Module):
    """Categorical mapper using the ReinMax-CV gradient estimator.

    Wang & Bui (2026). Replaces the original ReinMax with the
    control-variate variant which is empirically lower-variance on
    high-dimensional discrete latent VAEs.
    """

    def __init__(
        self,
        bits: int,
        kl_threshold: float = 0.0,
        *,
        cv_tau: float = 0.7,
        cv_eta: float = 1.5,
        cv_mc_samples: int = 32,
    ):
        super().__init__()
        self.bits = bits
        self.num_codes = 2 ** bits
        self.kl_threshold = kl_threshold
        self.cv_tau = float(cv_tau)
        self.cv_eta = float(cv_eta)
        self.cv_mc_samples = int(cv_mc_samples)

    @property
    def projection_dim(self) -> int:
        return self.num_codes

    @property
    def latent_width(self) -> int:
        return self.num_codes

    def empty_cache(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.empty(batch, 0, self.num_codes, device=device, dtype=dtype)

    def _zero_kl(self, params: Tensor) -> Tensor:
        return torch.zeros_like(params[..., 0])

    def _draw(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        straight_through: bool = True,
    ) -> Tensor:
        if straight_through:
            return _CategoricalReinMaxCVST.apply(
                logits,
                logits.new_tensor(self.cv_tau),
                self.cv_eta,
                self.cv_mc_samples,
            )
        return _categorical_sample(logits, temperature)

    def sample_prior(
        self,
        prior_params: Tensor,
        temperature: float = 1.0,
        z_cache: Optional[Tensor] = None,
    ) -> LatentOutput:
        z = self._draw(prior_params, temperature, straight_through=False)
        if z_cache is not None and z_cache.shape[1] > 0:
            cache_len = min(z_cache.shape[1], z.shape[1])
            z = z.clone()
            z[:, :cache_len] = z_cache[:, :cache_len].to(z.dtype)

        kl = self._zero_kl(prior_params)
        return LatentOutput(z=z, kl=kl, kl_raw=kl)

    def sample_posterior(
        self,
        posterior_params: Tensor,
        temperature: float = 1.0,
    ) -> LatentOutput:
        z = self._draw(posterior_params, temperature, straight_through=True)
        kl = self._zero_kl(posterior_params)
        return LatentOutput(z=z, kl=kl, kl_raw=kl)

    def posterior_with_kl(
        self,
        posterior_logits: Tensor,
        prior_logits: Tensor,
        temperature: float = 1.0,
    ) -> LatentOutput:
        z = self._draw(posterior_logits, temperature, straight_through=True)
        q_log = F.log_softmax(posterior_logits.float(), dim=-1)
        p_log = F.log_softmax(prior_logits.float(), dim=-1)
        q = q_log.exp()
        kl_raw = (q * (q_log - p_log)).sum(dim=-1)
        kl = F.relu(kl_raw - self.kl_threshold)
        return LatentOutput(z=z, kl=kl, kl_raw=kl_raw)


class CategoricalReinMaxRaoMapper(nn.Module):
    """Categorical mapper using the ReinMax-Rao gradient estimator.

    Wang & Bui (2026). Lowest-variance of the three ReinMax variants;
    drops the STGS term and uses only the Gumbel-Rao estimate. 
    """

    def __init__(
        self,
        bits: int,
        kl_threshold: float = 0.0,
        *,
        rao_tau: float = 0.7,
        rao_mc_samples: int = 32,
    ):
        super().__init__()
        self.bits = bits
        self.num_codes = 2 ** bits
        self.kl_threshold = kl_threshold
        self.rao_tau = float(rao_tau)
        self.rao_mc_samples = int(rao_mc_samples)

    @property
    def projection_dim(self) -> int:
        return self.num_codes

    @property
    def latent_width(self) -> int:
        return self.num_codes

    def empty_cache(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.empty(batch, 0, self.num_codes, device=device, dtype=dtype)

    def _zero_kl(self, params: Tensor) -> Tensor:
        return torch.zeros_like(params[..., 0])

    def _draw(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        straight_through: bool = True,
    ) -> Tensor:
        if straight_through:
            return _CategoricalReinMaxRaoST.apply(
                logits,
                logits.new_tensor(self.rao_tau),
                self.rao_mc_samples,
            )
        return _categorical_sample(logits, temperature)

    def sample_prior(
        self,
        prior_params: Tensor,
        temperature: float = 1.0,
        z_cache: Optional[Tensor] = None,
    ) -> LatentOutput:
        z = self._draw(prior_params, temperature, straight_through=False)
        if z_cache is not None and z_cache.shape[1] > 0:
            cache_len = min(z_cache.shape[1], z.shape[1])
            z = z.clone()
            z[:, :cache_len] = z_cache[:, :cache_len].to(z.dtype)

        kl = self._zero_kl(prior_params)
        return LatentOutput(z=z, kl=kl, kl_raw=kl)

    def sample_posterior(
        self,
        posterior_params: Tensor,
        temperature: float = 1.0,
    ) -> LatentOutput:
        z = self._draw(posterior_params, temperature, straight_through=True)
        kl = self._zero_kl(posterior_params)
        return LatentOutput(z=z, kl=kl, kl_raw=kl)

    def posterior_with_kl(
        self,
        posterior_logits: Tensor,
        prior_logits: Tensor,
        temperature: float = 1.0,
    ) -> LatentOutput:
        z = self._draw(posterior_logits, temperature, straight_through=True)
        q_log = F.log_softmax(posterior_logits.float(), dim=-1)
        p_log = F.log_softmax(prior_logits.float(), dim=-1)
        q = q_log.exp()
        kl_raw = (q * (q_log - p_log)).sum(dim=-1)
        kl = F.relu(kl_raw - self.kl_threshold)
        return LatentOutput(z=z, kl=kl, kl_raw=kl_raw)

