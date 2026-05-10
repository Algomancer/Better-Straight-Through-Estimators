import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

class LatentOutput:
    z: Tensor
    kl: Tensor
    kl_raw: Tensor

_LOG_EPS = 1e-12


def _safe_exponential_like(x: Tensor) -> Tensor:
    """Draw Exp(1) samples with the same shape/dtype/device as ``x``,
    clamped above ``_LOG_EPS`` so subsequent ``log`` is finite."""
    out = torch.empty_like(x)
    out.exponential_()
    return out.clamp_min(_LOG_EPS)


def _gumbel_rao_jvp(
    D: Tensor,
    pi_D: Tensor,
    grad: Tensor,
    tau: Tensor,
    K: int,
    eps: float = _LOG_EPS,
    sample_exp: Tensor | None = None,
) -> Tensor:
    D = D.float()
    pi_D = pi_D.float()
    grad = grad.float()
    tau = tau.float()

    pi_D_safe = pi_D.clamp_min(eps)
    D_bool = D.bool()
    if sample_exp is None:
        E_shape = pi_D.shape + (K,)
        E = torch.empty(E_shape, device=pi_D.device, dtype=pi_D.dtype)
        E.exponential_()
    else:
        E = sample_exp.to(device=pi_D.device, dtype=pi_D.dtype)

    finfo = torch.finfo(E.dtype)
    E = torch.nan_to_num(E, nan=1.0, posinf=finfo.max, neginf=eps)
    E = E.clamp(min=eps, max=finfo.max)

    sample_idx = D.argmax(dim=-1, keepdim=True)
    gather_idx = sample_idx.unsqueeze(-1).expand(*sample_idx.shape, K)
    E_i = E.gather(dim=-2, index=gather_idx).squeeze(-2)           # [..., K]

    log_E = E.log()
    log_E_i = E_i.log()
    log_pi_D = pi_D_safe.log()
    nonselected = -torch.logaddexp(
        log_E - log_pi_D.unsqueeze(-1),
        log_E_i.unsqueeze(-2),
    )
    selected = -log_E_i.unsqueeze(-2)
    cond_logits = torch.where(D_bool.unsqueeze(-1), selected, nonselected)
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
        eps = _LOG_EPS
        grad = grad_output.float()

        pi_D = 0.5 * (pi + D)

        # Standard ReinMax (Heun midpoint, tau = 1).
        grad_rm = 2.0 * pi_D * (grad - (pi_D * grad).sum(dim=-1, keepdim=True))
        grad_rm = grad_rm - 0.5 * pi * (
            grad - (pi * grad).sum(dim=-1, keepdim=True)
        )

        # Single-sample STGS at theta_D, temperature cv_tau.
        log_pi_D = pi_D.clamp_min(eps).log()
        gumbel = -_safe_exponential_like(pi_D).log()
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

    Wang & Bui (2026), eq 17. Lowest-variance of the three ReinMax variants
    in the paper's Tables 2-3, and topped both highest-cat-dim configurations
    (16x12 and 64x8). Drops the STGS term entirely::

        grad = 2 * J_GR^T g  at theta_D, tau    -    0.5 * J(pi)^T g

    Note: the paper's equation 17 places the second term at ``theta_D``;
    this is a typo. The second term comes from the original ReinMax
    decomposition and is at ``theta``, which is what we implement here.
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
        grad = grad_output.float()

        pi_D = 0.5 * (pi + D)
        grad_GR = _gumbel_rao_jvp(D, pi_D, grad, tau, K, _LOG_EPS)
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

    Has no learnable parameters; ``cv_tau``/``cv_eta``/``cv_mc_samples``
    are plain Python attributes so the state dict matches a vanilla
    ReinMax mapper -- existing checkpoints can resume directly.
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
    drops the STGS term and uses only the Gumbel-Rao estimate. Best on
    the largest-cat-dim configurations in the paper's Tables 2-3.
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


