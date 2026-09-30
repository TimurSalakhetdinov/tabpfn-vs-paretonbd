"""Hierarchical-Bayes Pareto/NBD, model M1 of Abe (2009), "Counting Your Customers One by One".

Independent implementation of the published model (no covariates):
    purchases while alive ~ Poisson(lambda_i), lifetime tau_i ~ Exponential(mu_i),
    (log lambda_i, log mu_i) ~ N(beta0, Gamma0)          (correlated log-normal heterogeneity)
    Gamma0 ~ Inv-Wishart(nu00, Gamma00),  beta0 | Gamma0 ~ N(beta00, Gamma0 / kappa0)
Sampling is Metropolis-within-Gibbs with data augmentation (Abe 2009, appendix):
    1. z_i (alive at T) ~ Bernoulli(P(alive | lambda_i, mu_i)),
       P(alive) = 1 / (1 + mu/(lambda+mu) * (exp((lambda+mu)(T - t_x)) - 1));
    2. tau_i | dead ~ truncated Exponential(lambda+mu) on (t_x, T);
    3. (log lambda_i, log mu_i): vectorised random-walk Metropolis with the likelihood
       lambda^x * mu^(1-z) * exp(-(lambda+mu) * min(T, tau)); proposal scale adapted per customer during burn-in only;
    4. (beta0, Gamma0): conjugate normal-inverse-Wishart update.
Predictions average over retained draws. For each draw, P(no occasion in (T, T+t]) =
    (1 - P(alive)) + P(alive) * (mu/(lambda+mu) + lambda/(lambda+mu) * exp(-(lambda+mu) t)),
which is not P(alive). Expected count: P(alive) * lambda/mu * (1 - exp(-mu t)).

Time is rescaled internally by `time_unit` (default 7: days in, weeks inside the sampler) so that priors are
on a familiar scale; all public inputs and outputs use the caller's unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import invwishart

HYPER_NAMES = ["beta_log_lambda", "beta_log_mu", "var_log_lambda", "var_log_mu", "cov_log_lambda_mu"]


def _p_alive(lam, mu, t_x, T):
    s = lam + mu
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + mu / s * np.expm1(np.minimum(s * (T - t_x), 700.0)))


def _loglik(eta, x, t_x, T, z, tau):
    lam, mu = np.exp(eta[:, 0]), np.exp(eta[:, 1])
    span = np.where(z, T, tau)
    return x * eta[:, 0] + (1 - z) * eta[:, 1] - (lam + mu) * span


def _logprior(eta, beta, prec):
    d = eta - beta
    return -0.5 * np.einsum("ni,ij,nj->n", d, prec, d)


def split_rhat(chains: np.ndarray) -> float:
    """Split-R-hat (Gelman et al., BDA3) for draws shaped (n_chains, n_draws)."""
    n = chains.shape[1] // 2
    c = np.concatenate([chains[:, :n], chains[:, n:2 * n]])
    w = c.var(axis=1, ddof=1).mean()
    b = n * c.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def ess(chains: np.ndarray) -> float:
    """Effective sample size with Geyer's initial positive sequence, pooled over chains."""
    m, n = chains.shape
    c = chains - chains.mean(axis=1, keepdims=True)
    f = np.fft.rfft(c, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    w = acov[:, 0].mean() * n / (n - 1)
    var = (n - 1) / n * w + chains.mean(axis=1).var(ddof=1) if m > 1 else w
    rho = 1 - (w - acov.mean(axis=0)) / var
    rho[0] = 1.0
    s, k = 0.0, 0
    while k + 1 < n:
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        s += pair
        k += 2
    return float(m * n / max(-1 + 2 * s, 1e-12))


@dataclass
class AbeHB:
    n_chains: int = 4
    burnin: int = 2000
    draws: int = 2000
    thin: int = 2
    mh_steps: int = 3  # Metropolis updates of each customer's (log lambda, log mu) per Gibbs sweep
    seed: int = 0
    time_unit: float = 7.0
    # weakly informative hyperpriors (on log-rates per `time_unit`)
    beta00: tuple = (-3.0, -3.0)
    kappa0: float = 0.01
    nu00: float = 4.0
    gamma00_scale: float = 1.0
    hyper_draws: np.ndarray = field(default=None, repr=False)   # (chains, kept, 5)
    eta_draws: np.ndarray = field(default=None, repr=False)     # (chains*kept, N, 2), on the internal unit
    accept_rate: float = np.nan

    def fit(self, x, t_x, T):
        x, t_x, T = (np.asarray(v, float) for v in (x, t_x, T))
        t_x, T = t_x / self.time_unit, T / self.time_unit
        self._T_fit = T
        n = len(x)
        kept = self.draws // self.thin
        hyper = np.empty((self.n_chains, kept, 5))
        etas = np.empty((self.n_chains, kept, n, 2), dtype=np.float32)
        acc = []
        for c in range(self.n_chains):
            rng = np.random.default_rng([self.seed, c])
            hyper[c], etas[c], a = self._chain(x, t_x, T, rng, kept)
            acc.append(a)
        self.hyper_draws, self.eta_draws, self.accept_rate = hyper, etas.reshape(-1, n, 2), float(np.mean(acc))
        return self

    def _chain(self, x, t_x, T, rng, kept):
        n = len(x)
        # overdispersed start: jittered empirical rates
        lam0 = (x + 0.5) / np.maximum(T, 1e-3)
        eta = np.column_stack([np.log(lam0), np.full(n, np.log(1.0 / np.maximum(T.mean(), 1.0)))])
        eta += rng.normal(0, 0.5, size=eta.shape)
        beta = np.array(self.beta00, float)
        gamma = np.eye(2) * self.gamma00_scale
        step = np.full(n, 0.5)
        n_acc = np.zeros(n)
        hyper = np.empty((kept, 5))
        store = np.empty((kept, n, 2), dtype=np.float32)
        b00, k0, nu0, G00 = np.array(self.beta00, float), self.kappa0, self.nu00, np.eye(2) * self.gamma00_scale
        total = self.burnin + self.draws
        for it in range(total):
            lam, mu = np.exp(eta[:, 0]), np.exp(eta[:, 1])
            # 1-2. latent alive indicator and death time
            z = rng.random(n) < _p_alive(lam, mu, t_x, T)
            s = lam + mu
            u = rng.random(n)
            # inverse CDF of Exp(s) truncated to (t_x, T), computed stably relative to t_x
            tau = t_x - np.log1p(-u * -np.expm1(-s * (T - t_x))) / s
            tau = np.where(z, T, np.minimum(tau, T))
            # 3. individual log-rates, random-walk Metropolis with correlated proposal from Gamma0
            prec = np.linalg.inv(gamma)
            L = np.linalg.cholesky(gamma)
            cur = _loglik(eta, x, t_x, T, z, tau) + _logprior(eta, beta, prec)
            for _ in range(self.mh_steps):
                prop = eta + step[:, None] * (rng.standard_normal((n, 2)) @ L.T)
                new = _loglik(prop, x, t_x, T, z, tau) + _logprior(prop, beta, prec)
                ok = np.log(rng.random(n)) < new - cur
                eta[ok], cur[ok] = prop[ok], new[ok]
                if it < self.burnin:  # per-customer adaptation towards ~35% acceptance, frozen after burn-in
                    step *= np.exp((ok - 0.35) / np.sqrt(it + 1))
                else:
                    n_acc += ok / self.mh_steps
            # 4. hyperparameters: conjugate normal-inverse-Wishart
            ebar = eta.mean(axis=0)
            S = (eta - ebar).T @ (eta - ebar)
            kn, nun = k0 + n, nu0 + n
            d = (ebar - b00)[:, None]
            Gn = G00 + S + (k0 * n / kn) * (d @ d.T)
            gamma = invwishart.rvs(df=nun, scale=Gn, random_state=rng)
            beta = rng.multivariate_normal((k0 * b00 + n * ebar) / kn, gamma / kn)
            j = it - self.burnin
            if j >= 0 and j % self.thin == 0:
                k = j // self.thin
                hyper[k] = [beta[0], beta[1], gamma[0, 0], gamma[1, 1], gamma[0, 1]]
                store[k] = eta
        return hyper, store, float(n_acc.mean() / self.draws)

    # --- diagnostics ---------------------------------------------------------------
    def diagnostics(self) -> dict:
        out = {}
        for i, name in enumerate(HYPER_NAMES):
            ch = self.hyper_draws[:, :, i]
            q = np.quantile(ch, [0.025, 0.5, 0.975])
            out[name] = {"median": float(q[1]), "q025": float(q[0]), "q975": float(q[2]),
                         "rhat": split_rhat(ch), "ess": ess(ch)}
        corr = self.hyper_draws[:, :, 4] / np.sqrt(self.hyper_draws[:, :, 2] * self.hyper_draws[:, :, 3])
        out["corr_log_lambda_mu"] = {"median": float(np.median(corr)), "rhat": split_rhat(corr), "ess": ess(corr)}
        out["accept_rate"] = self.accept_rate
        out["time_unit_days"] = self.time_unit
        return out

    # --- predictions for the fitted customers --------------------------------------------
    def _rates(self):
        return np.exp(self.eta_draws[..., 0].astype(float)), np.exp(self.eta_draws[..., 1].astype(float))

    def predict(self, horizon, t_x, T):
        """Posterior means for the customers the model was fitted on.

        Returns dict with p_alive, p_no_purchase (+ Monte Carlo s.e. across draws), expected_count.
        `horizon`, `t_x` and `T` are in the caller's unit.
        """
        t_x, T = np.asarray(t_x, float) / self.time_unit, np.asarray(T, float) / self.time_unit
        h = horizon / self.time_unit
        lam, mu = self._rates()
        pa = _p_alive(lam, mu, t_x[None], T[None])
        s = lam + mu
        q = mu / s + lam / s * np.exp(-s * h)
        p0 = (1 - pa) + pa * q
        ec = pa * lam / mu * -np.expm1(-mu * h)
        d = p0.shape[0]
        return {"p_alive": pa.mean(0), "p_no_purchase": p0.mean(0), "p_no_purchase_se": p0.std(0) / np.sqrt(d),
                "expected_count": ec.mean(0), "n_draws": d}
