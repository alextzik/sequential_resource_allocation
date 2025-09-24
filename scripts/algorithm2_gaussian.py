#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.stats import norm, multivariate_normal


@dataclass
class GaussianHorizon:
    T: int
    prices: np.ndarray  # shape (T,)
    L: float  # total resource budget
    mu: np.ndarray  # shape (T,)
    Sigma: np.ndarray  # shape (T,T)


def survival_inverse_normal(mu: float, sigma: float, s: float) -> float:
    # S(u) = P(D >= u) for D ~ N(mu, sigma^2) = 1 - Phi((u-mu)/sigma)
    # For s in (0,1), S^{-1}(s) = mu + sigma * Phi^{-1}(1 - s)
    s = float(np.clip(s, 1e-12, 1 - 1e-12))
    if sigma <= 0:
        # Degenerate variance: D = mu a.s.
        # Then S(u) = 1 if u <= mu else 0; inverse picks u = mu for any s in (0,1)
        return max(0.0, mu)
    z = norm.ppf(1.0 - s)
    return max(0.0, mu + sigma * z)


def algorithm1_bisection(prices: np.ndarray, L: float, mu_vec: np.ndarray, sigma_vec: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, float]:
    # Implements Algorithm 1 using bisection on nu* with Normal survival inverses
    T = len(prices)
    assert mu_vec.shape == (T,) and sigma_vec.shape == (T,)
    pmin = float(np.min(prices))
    nu_low, nu_up = 0.0, pmin
    # Number of iterations from paper: ceil(log(min{p}/eps))
    N = int(np.ceil(np.log(max(pmin/eps, 1 + 1e-9))))

    def alloc_for(nu: float) -> np.ndarray:
        s = np.clip(nu / prices, 0.0, 1.0)
        a = np.array([survival_inverse_normal(mu_vec[t], sigma_vec[t], s[t]) for t in range(T)])
        return a

    a_mid = None
    for _ in range(max(N, 30)):  # ensure enough iterations even if pmin/eps small
        nu_mid = 0.5 * (nu_low + nu_up)
        a_mid = alloc_for(nu_mid)
        if np.sum(a_mid) >= L:
            nu_low = nu_mid
        else:
            nu_up = nu_mid
        if nu_up - nu_low <= eps:
            break

    # Final allocation at nu_low 
    a_final = alloc_for(nu_low)
    # Small adjustment to meet exactly the budget within tolerance by scaling if necessary
    total = np.sum(a_final)
    if total > 0 and abs(total - L) / max(1.0, L) > 1e-3:
        scale = min(1.0, L / total)
        a_final = a_final * scale
    return a_final, nu_low


def conditional_gaussian(mu: np.ndarray, Sigma: np.ndarray, obs_idx: np.ndarray, obs_vals: np.ndarray, target_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (mu_cond, Sigma_cond) for target given the observations
    if obs_idx.size == 0:
        return mu[target_idx], Sigma[np.ix_(target_idx, target_idx)]
    A = obs_idx
    B = target_idx
    mu_A = mu[A]
    mu_B = mu[B]
    Sigma_AA = Sigma[np.ix_(A, A)]
    Sigma_BA = Sigma[np.ix_(B, A)]
    Sigma_AB = Sigma[np.ix_(A, B)]
    Sigma_BB = Sigma[np.ix_(B, B)]

    try:
        K = np.linalg.solve(Sigma_AA, (obs_vals - mu_A))
        Sigma_AA_inv = np.linalg.inv(Sigma_AA)
    except LinAlgError:
        # Regularize if badly conditioned
        reg = 1e-8 * np.eye(Sigma_AA.shape[0])
        Sigma_AA_inv = np.linalg.inv(Sigma_AA + reg)
        K = Sigma_AA_inv @ (obs_vals - mu_A)

    mu_cond = mu_B + Sigma_BA @ K
    Sigma_cond = Sigma_BB - Sigma_BA @ Sigma_AA_inv @ Sigma_AB
    return mu_cond, Sigma_cond


def run_algorithm2(gh: GaussianHorizon, eps: float = 1e-6, seed: Optional[int] = None, simulate: bool = True, demands: Optional[np.ndarray] = None) -> dict:
    T = gh.T
    prices = gh.prices
    L = gh.L
    mu = gh.mu.copy()
    Sigma = gh.Sigma.copy()

    rng = np.random.default_rng(seed)

    allocations = np.zeros(T)
    realized = np.zeros(T)
    remaining = float(L)

    obs_idx = np.array([], dtype=int)
    obs_vals = np.array([])

    for t in range(T):
        if remaining <= 1e-12:
            break
        future_idx = np.arange(t, T)
        mu_cond, Sigma_cond = conditional_gaussian(mu, Sigma, obs_idx, obs_vals, future_idx)
        sigma_cond = np.sqrt(np.maximum(np.diag(Sigma_cond), 0.0))

        a_future, nu = algorithm1_bisection(prices[t:], remaining, mu_cond, sigma_cond, eps=eps)
        a_t = float(a_future[0])
        # Clip by remaining
        a_t = float(min(a_t, remaining))
        allocations[t] = a_t
        remaining -= a_t

        # Realize demand
        if demands is not None:
            d_t = float(demands[t])
        elif simulate:
            if t == 0:
                d_t = float(rng.normal(mu[0], np.sqrt(max(Sigma[0,0], 0.0))))
            else:
                # sample from conditional for the single next variable
                mu1, Sig1 = conditional_gaussian(mu, Sigma, np.arange(t), realized[:t], np.array([t]))
                d_t = float(rng.normal(mu1[0], np.sqrt(max(Sig1[0,0], 0.0))))
        else:
            d_t = np.nan
        realized[t] = d_t

        # Update conditioning sets
        obs_idx = np.arange(t+1)
        obs_vals = realized[:t+1]

    # Compute reward if realized available
    served = np.minimum(realized, allocations)
    reward = float(np.sum(prices * served))

    return {
        "allocations": allocations,
        "realized": realized,
        "served": served,
        "reward": reward,
        "remaining": remaining,
    }


def build_synthetic(T: int, seed: int = 0) -> GaussianHorizon:
    rng = np.random.default_rng(seed)
    prices = rng.uniform(0.8, 2.0, size=T)
    mu = rng.uniform(3.0, 8.0, size=T)
    A = rng.normal(scale=1.0, size=(T, T))
    Sigma = A @ A.T
    # Normalize to moderate variances and some correlation
    d = np.sqrt(np.diag(Sigma) + 1e-6)
    Sigma = Sigma / (np.outer(d, d))  # correlation matrix
    scales = rng.uniform(0.5, 2.0, size=T)
    Sigma = Sigma * np.outer(scales, scales)

    L = float(rng.uniform(0.3 * np.sum(mu), 0.6 * np.sum(mu)))
    return GaussianHorizon(T=T, prices=prices, L=L, mu=mu, Sigma=Sigma)


def main():
    ap = argparse.ArgumentParser(description="Algorithm 2 (receding horizon) with Gaussian demands, using Algorithm 1 for inner solves")
    ap.add_argument("--T", type=int, default=100, help="Horizon length")
    ap.add_argument("--L", type=float, default=None, help="Total resource budget. Default: random if not provided")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--simulate", type=bool, default=True, help="Simulate a demand path")
    args = ap.parse_args()

    gh = build_synthetic(args.T, args.seed)
    if args.L is not None:
        gh.L = float(args.L)

    res = run_algorithm2(gh, eps=1e-6, seed=args.seed, simulate=args.simulate)
    print(f"T={gh.T} L={gh.L:.2f} reward={res['reward']:.3f}")
    print("prices:", np.array2string(gh.prices, precision=3))
    print("allocations:", np.array2string(res["allocations"], precision=3))
    if args.simulate:
        print("realized:", np.array2string(res["realized"], precision=3))
        print("served:", np.array2string(res["served"], precision=3))
    print("reward:", res["reward"])

if __name__ == "__main__":
    main()
