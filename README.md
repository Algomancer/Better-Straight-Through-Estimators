# Better Straight Through Estimator

Couldn't find an implimentation - quick hack. Setup for catagorical / discrete vaes.

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
