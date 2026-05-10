import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

class LatentOutput:
    z: Tensor
    kl: Tensor
    kl_raw: Tensor



class _CategoricalReinMaxCVST(torch.autograd.Function):
    """Hard categorical sample with the ReinMax-CV gradient estimator.

    ReinMax-CV (Wang & Bui, "Beyond ReinMax", arXiv:2603.08257, 2026) reduces
    the variance of ReinMax by applying a Gumbel-Rao control variate to the
    high-variance term ST_{tau=1}(D, theta_D), where theta_D is the Heun
    midpoint reparameterization log((pi + D) / 2).

    Backward gradient:

        grad = 2 * J(pi_D)^T g  -  0.5 * J(pi)^T g                        [ReinMax]
             + eta * ( grad_GR(theta_D, tau)  -  grad_STGS(theta_D, tau) ) [CV]

    where J(p) = diag(p) - p p^T, GR is the Rao-Blackwellised STGS using K
    conditional Gumbels, and STGS is a single-sample Gumbel-Softmax JVP.

    Defaults follow the paper: tau in [0.7, 1.3] (tunable), eta = 1.5, K = 100.
    """

    @staticmethod
    def forward(
        ctx,
        logits: Tensor,
        tau: Tensor,
        eta: float,
        mc_samples: int,
    ) -> Tensor:
        probs = logits.float().softmax(dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        idx = torch.multinomial(flat, num_samples=1, replacement=True)
        idx = idx.reshape(*probs.shape[:-1])
        z = F.one_hot(idx, num_classes=probs.shape[-1]).to(logits.dtype)

        ctx.save_for_backward(z.float(), probs, tau)
        ctx.eta = float(eta)
        ctx.K = int(mc_samples)
        ctx.logits_dtype = logits.dtype
        return z

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        D, pi, tau = ctx.saved_tensors
        eta = ctx.eta
        K = ctx.K
        eps = 1e-20
        grad = grad_output.float()

        # ---- Standard ReinMax (Heun, tau = 1) ----
        pi_D = 0.5 * (pi + D)
        # 2 * J(pi_D)^T grad
        grad_rm = 2.0 * pi_D * (grad - (pi_D * grad).sum(-1, keepdim=True))
        # -0.5 * J(pi)^T grad
        grad_rm = grad_rm - 0.5 * pi * (grad - (pi * grad).sum(-1, keepdim=True))

        # ---- CV at theta_D = log(pi_D); note Z(theta_D) = 1 ----
        pi_D_safe = pi_D.clamp_min(eps)
        log_pi_D = pi_D_safe.log()

        # Single-sample STGS at theta_D (control variate)
        gumbel = -torch.empty_like(pi_D).exponential_().log()
        pi_GS = ((log_pi_D + gumbel) / tau).softmax(dim=-1)
        grad_GS = pi_GS * (grad - (pi_GS * grad).sum(-1, keepdim=True)) / tau

        # Gumbel-Rao at theta_D via conditional Gumbel reparam (Maddison 2014):
        #   theta_D_j + G_j | (D = I_i) = -log( E_j / pi_D_j + E_i )
        # with the convention E_j / pi_D_j := 0 at j = i (then -> -log E_i).
        D_bool = D.bool()
        E = torch.empty(pi_D.shape + (K,), device=pi_D.device, dtype=pi_D.dtype)
        E.exponential_()
        E_i = E[D_bool].reshape(*pi_D.shape[:-1], K)              # [..., K]

        ratio = E / pi_D_safe.unsqueeze(-1)                       # [..., C, K]
        ratio = ratio.masked_fill(D_bool.unsqueeze(-1), 0.0)
        cond_logits = -(ratio + E_i.unsqueeze(-2) + eps).log()    # [..., C, K]
        pi_GR_all = (cond_logits / tau).softmax(dim=-2)           # [..., C, K]

        inner = (pi_GR_all * grad.unsqueeze(-1)).sum(dim=-2, keepdim=True)
        grad_GR = (pi_GR_all * (grad.unsqueeze(-1) - inner)).mean(dim=-1) / tau

        # ---- Combine ----
        grad_logits = grad_rm + eta * (grad_GR - grad_GS)
        # Sum-zero projection: J^T g is sum-zero analytically; numerical safety.
        grad_logits = grad_logits - grad_logits.mean(dim=-1, keepdim=True)
        return grad_logits.to(ctx.logits_dtype), None, None, None


class CategoricalReinMaxCVMapper(nn.Module):
    """Vae mapper"""

    def __init__(
        self,
        bits: int,
        kl_threshold: float = 0.0,
        cv_eta: float = 1.5,
        mc_samples: int = 100,
    ):
        super().__init__()
        self.bits = bits
        self.num_codes = 2 ** bits
        self.kl_threshold = kl_threshold
        self.cv_eta = cv_eta
        self.mc_samples = mc_samples

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
                logits.new_tensor(float(temperature)),
                self.cv_eta,
                self.mc_samples,
            )
        probs = (logits.float() / float(temperature)).softmax(dim=-1)
        u = torch.rand((*probs.shape[:-1], 1), device=probs.device, dtype=probs.dtype)
        sample = (u > probs.cumsum(dim=-1)).sum(dim=-1).clamp_max(probs.shape[-1] - 1)
        return F.one_hot(sample, probs.shape[-1]).to(logits.dtype)

    def sample_prior(
        self,
        prior_params: Tensor,
        temperature: float = 1.0,
        z_cache: Optional[Tensor] = None,
    ) -> "LatentOutput":
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
    ) -> "LatentOutput":
        z = self._draw(posterior_params, temperature, straight_through=True)
        kl = self._zero_kl(posterior_params)
        return LatentOutput(z=z, kl=kl, kl_raw=kl)

    def posterior_with_kl(
        self,
        posterior_logits: Tensor,
        prior_logits: Tensor,
        temperature: float = 1.0,
    ) -> "LatentOutput":
        z = self._draw(posterior_logits, temperature, straight_through=True)
        q_log = F.log_softmax(posterior_logits.float(), dim=-1)
        p_log = F.log_softmax(prior_logits.float(), dim=-1)
        q = q_log.exp()
        kl_raw = (q * (q_log - p_log)).sum(dim=-1)
        kl = F.relu(kl_raw - self.kl_threshold)
        return LatentOutput(z=z, kl=kl, kl_raw=kl_raw)
