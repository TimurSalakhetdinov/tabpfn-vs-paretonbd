"""Standard Pareto/NBD (Schmittlein, Morrison and Colombo 1987), maximum-likelihood fit.

Independent implementation from the published formulas:
- likelihood: Fader, Hardie and Lee (2005), "A Note on Deriving the Pareto/NBD Model and Related Expressions";
- P(alive) and conditional expectation: same note, eqs. (11) and (13).

This is the classic model with independent gamma heterogeneity on lambda and mu. It is NOT
Abe's (2009) hierarchical model with correlated log-normal heterogeneity; see `abe_hb.py` for that.

Inputs use the sufficient statistics (x = repeat occasions, t_x = recency, T = customer age) in any
time unit; predictions for a horizon `t` use the same unit.

P(no occasion in (T, T+t]) is NOT P(alive). It is computed as
    (1 - P(alive)) + P(alive) * E[ mu/(lam+mu) + lam/(lam+mu) * exp(-(lam+mu) t) | alive ]
with lam ~ Gamma(r+x, alpha+T) and mu ~ Gamma(s, beta+T) (the posterior given alive at T),
evaluated by seeded Monte Carlo; the Monte Carlo standard error is returned alongside.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1


def _log_a0(r, alpha, s, beta, x, t_x, T):
    """log of A0 (Fader, Hardie and Lee 2005, eq. 22-23), switching branch on alpha vs beta."""
    rsx = r + s + x
    if alpha >= beta:
        a, b, z1, z2 = alpha + t_x, alpha + T, (alpha - beta) / (alpha + t_x), (alpha - beta) / (alpha + T)
        f1, f2 = hyp2f1(rsx, s + 1, rsx + 1, z1), hyp2f1(rsx, s + 1, rsx + 1, z2)
    else:
        a, b, z1, z2 = beta + t_x, beta + T, (beta - alpha) / (beta + t_x), (beta - alpha) / (beta + T)
        f1, f2 = hyp2f1(rsx, r + x, rsx + 1, z1), hyp2f1(rsx, r + x, rsx + 1, z2)
    t1 = np.log(f1) - rsx * np.log(a)
    t2 = np.log(f2) - rsx * np.log(b)
    # log(exp(t1) - exp(t2)), t1 >= t2 because t_x <= T
    with np.errstate(divide="ignore"):
        return t1 + np.log1p(-np.exp(t2 - t1))


def log_likelihood(params, x, t_x, T):
    r, alpha, s, beta = params
    base = gammaln(r + x) - gammaln(r) + r * np.log(alpha) + s * np.log(beta)
    l1 = -(r + x) * np.log(alpha + T) - s * np.log(beta + T)
    la0 = _log_a0(r, alpha, s, beta, x, t_x, T)
    l2 = np.log(s / (r + s + x)) + la0
    # t_x == T gives A0 = 0 (log = -inf); logaddexp handles it
    return base + np.logaddexp(l1, l2)


@dataclass
class ParetoNBD:
    r: float = np.nan
    alpha: float = np.nan
    s: float = np.nan
    beta: float = np.nan
    converged: bool = False
    loglik: float = np.nan

    at_bound: bool = False
    LOG_BOUNDS = (-9.2, 9.2)  # 1e-4 .. 1e4 in rescaled time; guards the s, beta -> infinity (no-heterogeneity) ridge

    def fit(self, x, t_x, T, init=(0.5, 10.0, 0.5, 10.0)):
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        scale = float(np.mean(T)) / 30.0  # rescale time for a well-conditioned optimisation
        xs, txs, Ts = x, t_x / scale, T / scale
        bounds = [self.LOG_BOUNDS] * 4

        def nll(lp):
            with np.errstate(all="ignore"):
                ll = log_likelihood(np.exp(lp), xs, txs, Ts)
            v = -np.sum(ll)
            return v if np.isfinite(v) else 1e300

        best = None
        for start in [np.log(init), np.log([1.0, 1.0, 1.0, 1.0]), np.log([0.3, 30.0, 0.3, 30.0])]:
            res = minimize(nll, start, method="Nelder-Mead", bounds=bounds,
                           options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-9})
            res = minimize(nll, res.x, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        if not np.isfinite(best.fun) or best.fun >= 1e300:
            raise RuntimeError("Pareto/NBD likelihood is not finite at any start")
        r, a, s, b = np.exp(best.x)
        self.r, self.alpha, self.s, self.beta = r, a * scale, s, b * scale
        self.converged = bool(best.success)
        self.at_bound = bool(np.any(np.abs(best.x) > self.LOG_BOUNDS[1] - 1e-3))
        self.loglik = float(np.sum(log_likelihood(self.params, x, t_x, T)))  # in the caller's time unit
        return self

    @property
    def params(self):
        return np.array([self.r, self.alpha, self.s, self.beta])

    def p_alive(self, x, t_x, T):
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        r, alpha, s, beta = self.params
        la0 = _log_a0(r, alpha, s, beta, x, t_x, T)
        lr = np.log(s / (r + s + x)) + (r + x) * np.log(alpha + T) + s * np.log(beta + T) + la0
        return 1.0 / (1.0 + np.exp(lr))

    def expected_count(self, t, x, t_x, T):
        """E[X(T, T+t) | x, t_x, T] (eq. 13 of the note); requires s != 1."""
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        r, alpha, s, beta = self.params
        pa = self.p_alive(x, t_x, T)
        return pa * (r + x) * (beta + T) / ((alpha + T) * (s - 1)) * (1 - ((beta + T) / (beta + T + t)) ** (s - 1))

    def prob_no_purchase(self, t, x, t_x, T, n_draws: int = 4000, seed: int = 0, chunk: int = 2000):
        """P(no occasion in (T, T+t] | data), with Monte Carlo standard error. Returns (p, se)."""
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        r, alpha, s, beta = self.params
        pa = self.p_alive(x, t_x, T)
        rng = np.random.default_rng(seed)
        mean, se = np.empty(len(x)), np.empty(len(x))
        for i in range(0, len(x), chunk):
            sl = slice(i, i + chunk)
            lam = rng.gamma(r + x[sl, None], 1.0 / (alpha + T[sl, None]), size=(len(x[sl]), n_draws))
            mu = rng.gamma(s, 1.0 / (beta + T[sl, None]), size=(len(x[sl]), n_draws))
            q = mu / (lam + mu) + lam / (lam + mu) * np.exp(-(lam + mu) * t)
            mean[sl], se[sl] = q.mean(1), q.std(1) / np.sqrt(n_draws)
        p = (1 - pa) + pa * mean
        return p, pa * se
