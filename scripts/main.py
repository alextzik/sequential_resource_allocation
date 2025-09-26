#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.stats import norm, multivariate_normal

import cvxpy as cp

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class GaussianHorizon:
    T: int
    prices: np.ndarray  # shape (T,)
    L: float  # total resource budget
    mu: np.ndarray  # shape (T,)
    Sigma: np.ndarray  # shape (T,T)
    mu_logs: Optional[np.ndarray] = None  # shape (T,), optional, for reference
    sigma_logs: Optional[np.ndarray] = None  # shape (T,), optional, for


def survival_inverse_normal(mu: float, sigma: float, s: float) -> float:
    # S(u) = P(D >= u) for D ~ N(mu, sigma^2) = 1 - Phi((u-mu)/sigma)
    # For s in (0,1), S^{-1}(s) = mu + sigma * Phi^{-1}(1 - s)
    s = float(np.clip(s, 1e-12, 1 - 1e-12))
    if sigma <= 0:
        # Degenerate variance: D = mu a.s.
        # Then S(u) = 1 if u <= mu else 0; inverse picks u = mu for any s in (0,1)
        return max(0.0, mu)
    z = norm.ppf(1.0 - s)
    return mu + sigma * z


def algorithm1_bisection(prices: np.ndarray, L: float, mu_vec: np.ndarray, sigma_vec: np.ndarray, eps: float = 1e-7) -> Tuple[np.ndarray, float]:
    # Implements Algorithm 1 using bisection on nu* with Normal survival inverses
    T = len(prices)
    assert mu_vec.shape == (T,) and sigma_vec.shape == (T,)
    pmax = float(np.max(prices))
    nu_low, nu_up = 0.0, pmax
    # Number of iterations from paper: ceil(log(max{p}/eps))
    N = int(np.ceil(np.log(max(pmax/eps, 1 + 1e-9))))

    def alloc_for(nu: float) -> np.ndarray:
        s = nu / prices
        a = np.array([survival_inverse_normal(mu_vec[t], sigma_vec[t], s[t]) if nu/prices[t]<1. else 0. for t in range(T)])
        return np.exp(a)

    a_mid = None
    for _ in range(max(N, 50)):  # ensure enough iterations even if pmax/eps small
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
    if total > 0 and abs(total - L) / max(1.0, L) > 1e-4:
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


def run_algorithm2(gh: GaussianHorizon, eps: float = 1e-6, seed: Optional[int] = None, demands: Optional[np.ndarray] = None) -> dict:
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
        allocations[t] = a_t
        remaining -= a_t

        # Realize demand
        if demands is not None:
            d_t = float(demands[t])
            log_d_t = np.log(d_t)
        else:
            # sample from conditional for the single next variable
            mu1, Sig1 = conditional_gaussian(mu, Sigma, np.arange(t), realized[:t], np.array([t]))
            log_d_t = float(rng.normal(mu1[0], np.sqrt(max(Sig1[0,0], 0.0))))
        realized[t] = log_d_t

        # Update conditioning sets
        obs_idx = np.arange(t+1)
        obs_vals = realized[:t+1]

    # Compute reward if realized available
    served = np.minimum(np.exp(realized), allocations)
    reward = np.cumsum(prices * served)

    return {
        "allocations": allocations,
        "realized": np.exp(realized),
        "served": served,
        "reward": reward,
        "remaining": remaining,
    }


def run_baseline_static(gh: GaussianHorizon, eps: float = 1e-6) -> dict:
    """Static baseline: solve the joint problem once using the marginal distributions
    (unconditional) via Algorithm 1; no receding updates.
    """
    mu = gh.mu
    sigma = np.sqrt(np.maximum(np.diag(gh.Sigma), 0.0))
    allocations, nu = algorithm1_bisection(gh.prices, gh.L, mu, sigma, eps=eps)
    return {"allocations": allocations, "nu": nu}

def optimal_after_the_fact(prices: np.ndarray, L: float, demands: np.ndarray) -> dict:
    # Solve the offline optimal allocation problem given realized demands
    T = len(prices)
    assert demands.shape == (T,)
    # Sort by price/demand ratio in descending order
    allocations = cp.Variable(T, nonneg=True)
    objective = cp.Maximize(prices @ cp.minimum(allocations, demands))
    constraints = [cp.sum(allocations) <= L]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return {"allocations": allocations.value, "nu": None}

def evaluate_policy(prices: np.ndarray, allocations: np.ndarray, demands: np.ndarray) -> dict:
    served = np.minimum(allocations, demands)
    reward = np.cumsum(prices * served)
    return {"served": served, "reward": reward}

def sample_gaussian_path(mu: np.ndarray, Sigma: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Symmetrize and lightly regularize for numerical stability
    S = 0.5 * (Sigma + Sigma.T)
    S = S + 1e-10 * np.eye(S.shape[0])
    return multivariate_normal.rvs(mean=mu, cov=S, random_state=rng)

def get_normal_pars_from_lognormal(mu_log: np.ndarray, sigma_log: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Given mu_log and sigma_log for log-normal distribution, return mu and sigma for the underlying normal distribution
    sigma2_log = sigma_log ** 2
    mu = np.log(mu_log**2 / np.sqrt(sigma2_log + mu_log**2))
    sigma = np.sqrt(np.log(1 + sigma2_log / mu_log**2))
    return mu, sigma

def get_corr(T: int, rho: float = 0.7, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Correlation matrix with off-diagonals exactly in {+rho, -rho}.
    Corr = (1 - rho) I + rho * s s^T, where s_i ∈ {+1, -1}. This is SPD for 0 < rho < 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Use outer-product construction to ensure SPD and ±rho off-diagonals
    s = rng.choice([-1.0, 1.0], size=(T, T))
    s = np.triu(s, 1)
    s = (s + s.T)
    Corr = rho*s
    np.fill_diagonal(Corr, 1.0)

    # Use cvxpy to find nearest correlation matrix
    X = cp.Variable((T, T), PSD=True)
    constraints = [cp.diag(X) == 1]
    objective = cp.Minimize(cp.norm(X - Corr, 'fro'))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    Corr = X.value        
    return Corr

def build_synthetic(T: int, seed: int = 0, rho: float = 0.7) -> GaussianHorizon:
    rng = np.random.default_rng(seed)
    prices = rng.uniform(10, 100, size=T)
    mu_log = rng.uniform(20.0, 100.0, size=T)
    sigma_log = rng.uniform(10.0, 30.0, size=T)
    mu, sigma = get_normal_pars_from_lognormal(mu_log, sigma_log)
    # Correlation concentrated at ±rho
    Corr = get_corr(T, rho=rho, rng=rng)
    # Set marginal std scales and form covariance
    Sigma = np.outer(sigma, sigma) * Corr
    # Project to nearest PD matrix if needed
    X = cp.Variable((T, T), PSD=True)
    constraints = [cp.diag(X) == sigma**2]
    objective = cp.Minimize(cp.norm(X - Sigma, 'fro'))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    Sigma = X.value 
    # clip eigenvalues to be at least 1e-6
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.clip(eigvals, 1e-6, None)
    Sigma = (eigvecs * eigvals) @ eigvecs.T

    # Sigma = np.diag(sigma**2)

    L = float(rng.uniform(0.3 * np.sum(mu_log), 0.6 * np.sum(mu_log)))
    return GaussianHorizon(T=T, prices=prices, L=L, mu=mu, Sigma=Sigma, mu_logs=mu_log, sigma_logs=sigma_log)


def main():
    ap = argparse.ArgumentParser(description="Algorithm 2 (receding horizon) with Gaussian demands, using Algorithm 1 for inner solves")
    ap.add_argument("--T", type=int, default=100, help="Horizon length")
    ap.add_argument("--L", type=float, default=None, help="Total resource budget. Default: random if not provided")
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.7, help="Target absolute correlation; off-diagonals at ±rho")
    ap.add_argument("--compare-baseline", type=bool, default=True, help="Compare with static baseline using marginals")
    ap.add_argument("--trials", type=int, default=1, help="Number of independent demand paths to evaluate")
    args = ap.parse_args()

    gh = build_synthetic(args.T, args.seed, rho=args.rho)
    if args.L is not None:
        gh.L = float(args.L)

    if args.compare_baseline:
        # Baseline allocations are fixed for given (prices, mu, Sigma, L)
        base = run_baseline_static(gh, eps=1e-6)
        if args.trials <= 1:
            # Single trial detailed printout
            d = sample_gaussian_path(gh.mu, gh.Sigma, seed=args.seed)
            d = np.exp(d)  # convert to log-normal demands
            base_eval = evaluate_policy(gh.prices, base["allocations"], d)

            optimal_policy = optimal_after_the_fact(gh.prices, gh.L, d)
            optimal_policy_eval = evaluate_policy(gh.prices, optimal_policy["allocations"], d)

            rec = run_algorithm2(gh, eps=1e-6, seed=args.seed, demands=d)

            print(f"T={gh.T} L={gh.L:.2f}")
            print("prices:", np.array2string(gh.prices, precision=3))
            print("demands:", np.array2string(d, precision=3))
            print("baseline_alloc:", np.array2string(base["allocations"], precision=3))
            print("optimal_alloc:", np.array2string(optimal_policy["allocations"], precision=3))
            print("receding_alloc:", np.array2string(rec["allocations"], precision=3))
            print(f"baseline_reward: {base_eval['reward'][-1]:.3f}")
            print(f"optimal_reward: {optimal_policy_eval['reward'][-1]:.3f}")
            print(f"receding_reward: {rec['reward'][-1]:.3f}")
            # print sum of allocations compared to L
            print(f"sum_baseline_alloc: {np.sum(base['allocations']):.3f}  \
                  sum_optimal_alloc: {np.sum(optimal_policy['allocations']):.3f}  \
                    sum_receding_alloc: {np.sum(rec['allocations']):.3f}  L: {gh.L:.3f}")

            # set font to Times new roman and increase font size
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams.update({'font.size': 20})
            # Bar plot: realized demands vs mean demands
            idx = np.arange(gh.T)
            width = 0.5
            plt.figure(figsize=(10,6))
            plt.bar(idx - width/2, d, width=width, label='Realized Demand')
            plt.bar(idx + width/2, gh.mu_logs, width=width, label='Mean Demand')
            plt.xlabel('Time Period')
            plt.ylabel('Demand')
            # plt.title('Realized vs Mean Demand')
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.show()

            # make a bar plot of prices
            plt.figure(figsize=(10,6))
            plt.bar(np.arange(gh.T), gh.prices, width=0.5, color='orange')
            plt.xlabel('Time Period')
            plt.ylabel('Price')
            # plt.title('Prices Over Time')
            plt.grid(True, axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.show()

            # Bar plot: allocations (baseline vs sequential vs optimal)
            plt.figure(figsize=(10,6))
            width_alloc = 0.6
            offset = width_alloc
            plt.bar(idx - offset, base["allocations"], width=width_alloc, label='Static')
            plt.bar(idx, rec["allocations"], width=width_alloc, label='Sequential')
            plt.bar(idx + offset, optimal_policy["allocations"], width=width_alloc, label='Oracle', alpha=0.4)
            plt.xlabel('Time Period')
            plt.ylabel('Allocation')
            # plt.title('Allocations Over Time')
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.show()

            # make a line plot of the cumulative rewards over time
            plt.figure(figsize=(10,6))
            plt.plot(idx, base_eval['reward'], label='Static', marker='o')
            plt.plot(idx, rec['reward'], label='Sequential')
            plt.plot(idx, optimal_policy_eval['reward'], label='Oracle', marker='o')
            plt.xlabel('Time Period')
            plt.ylabel('Cumulative Revenue')
            # make y-axis be in scientific notation
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            # plt.title('Cumulative Rewards Over Time')
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.show()

        else:
            # Multiple independent trials with different demand paths
            base_rewards = []
            optimal_rewards = []
            rec_rewards = []
            for i in range(args.trials):
                print(f"Trial {i+1}/{args.trials}")
                d = sample_gaussian_path(gh.mu, gh.Sigma, seed=args.seed + i)
                d = np.exp(d)  # convert to log-normal demands
                base_eval = evaluate_policy(gh.prices, base["allocations"], d)

                optimal_policy = optimal_after_the_fact(gh.prices, gh.L, d)
                optimal_policy_eval = evaluate_policy(gh.prices, optimal_policy["allocations"], d)

                rec = run_algorithm2(gh, eps=1e-6, seed=args.seed + i, demands=d)
                base_rewards.append(base_eval["reward"][-1])
                optimal_rewards.append(optimal_policy_eval["reward"][-1])
                rec_rewards.append(rec["reward"][-1])

            base_rewards = np.array(base_rewards)
            optimal_rewards = np.array(optimal_rewards)
            rec_rewards = np.array(rec_rewards)
            improv = rec_rewards - base_rewards

            print(f"T={gh.T} L={gh.L:.2f} trials={args.trials} rho={args.rho}")
            print(f"baseline_mean: {base_rewards.mean():.3f}  std: {base_rewards.std(ddof=1):.3f}")
            print(f"receding_mean: {rec_rewards.mean():.3f}  std: {rec_rewards.std(ddof=1):.3f}")
            print(f"optimal_mean: {optimal_rewards.mean():.3f}  std: {optimal_rewards.std(ddof=1):.3f}")
            print(f"improvement_mean: {improv.mean():.3f}  std: {improv.std(ddof=1):.3f}")
    else:
        res = run_algorithm2(gh, eps=1e-6, seed=args.seed, demands=None)
        print(f"T={gh.T} L={gh.L:.2f} reward={res['reward']:.3f}")
        print("prices:", np.array2string(gh.prices, precision=3))
        print("allocations:", np.array2string(res["allocations"], precision=3))
        print("realized:", np.array2string(res["realized"], precision=3))
        print("served:", np.array2string(res["served"], precision=3))
        print("reward:", res["reward"]) 

if __name__ == "__main__":
    main()
