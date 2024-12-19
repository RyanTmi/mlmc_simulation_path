import numpy as np
from scipy.stats import norm


def black_scholes_call_price(s0, r, sigma, K, T):
    d1 = (np.log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = s0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def call_payoff(S, K):
    return np.maximum(S - K, 0.0)


# def sde_path(n, m, T, s, b, sigma, all_paths=True):
#     dt = T / m
#     dW = norm.rvs(scale=dt, size=(n, m))

#     if all_paths:
#         s = np.zeros((n, m + 1))
#         s[:, 0] = s
#     else:
#         s = np.full(n, s)

#     for i in range(m):
#         if all_paths:
#             s[:, i + 1] = s[:, i] + b(s[:, i]) * dt + sigma(s[:, i]) * dW[:, i]
#         else:
#             s = s + b(s) * dt + sigma(s) * dW[:, i]

#     return np.linspace(0, T, m + 1), s


# TODO: Add option to return all paths, no just the final point
def sde_path(s: float, r: float, sigma: float, M: int, dt: np.ndarray, dW: np.ndarray):
    S = s
    for i in range(M):
        S += r * S * dt + sigma * S * dW[:, i]

    return S


class MLMCEuropean:
    def __init__(self, max_level: int, M: int, T: float, rng: np.random.Generator):
        self.max_level = max_level
        self.M = M
        self.T = T
        self.rng = rng

    def estimate_call(self, s: float, r: float, sigma: float, strike: float, N: int):
        L = self.max_level
        M = self.M
        T = self.T

        estimators = []
        estimators_mean = np.zeros(L + 1)

        # ---- Level 0 ----
        S = s + r * s * T + sigma * s * self.rng.normal(scale=np.sqrt(T), size=N)

        estimators.append(np.exp(-r * T) * call_payoff(S, strike))
        estimators_mean[0] = np.mean(estimators[0])

        # ---- Levels 1..L ----
        for l in range(1, L + 1):
            m_fine, m_coarse = M**l, M ** (l - 1)
            dt_fine, dt_coarse = T / m_fine, T / m_coarse

            # Generate Brownian Motion increments for level l
            # In practice, faster than `np.random.randn`
            dW_fine = self.rng.normal(scale=np.sqrt(dt_fine), size=(N, m_fine))

            # Summation of Brownian increments in groups of 'M' to replicate the coarser increment
            dW_coarse = np.sum(dW_fine.reshape(N, m_coarse, M), axis=2)

            S_fine = sde_path(s, r, sigma, m_fine, dt_fine, dW_fine)
            S_coarse = sde_path(s, r, sigma, m_coarse, dt_coarse, dW_coarse)

            payoff_fine = call_payoff(S_fine, strike)
            payoff_coarse = call_payoff(S_coarse, strike)

            estimators.append(np.exp(-r * T) * (payoff_fine - payoff_coarse))
            estimators_mean[l] = np.mean(estimators[l])

        return estimators_mean


def main() -> None:
    r = 0.05
    sigma = 0.2
    s0 = 1.0
    K = 1.0
    T = 1.0

    # MLMC parameters
    L = 5
    N = 10_000
    M = 4

    # For reproducibility
    rng = np.random.default_rng(seed=42)

    # MLMC
    mlmc_european = MLMCEuropean(L, M, T, rng)
    mlmc_european.estimate_call(s0, r, sigma, K, N)

    # Classic Monte Carlo
    steps = M**L
    dt = T / steps
    dW = rng.normal(scale=np.sqrt(dt), size=(N, steps))

    S = sde_path(s0, r, sigma, steps, dt, dW)
    payoff = np.exp(-r * T) * call_payoff(S, K)
    p, v, s = np.mean(payoff), np.var(payoff, ddof=1), np.std(payoff, ddof=1)
    print(f"==================================================")
    print(f"Classic Monte Carlo: {p}")
    print(f"Variance: {v / N}")
    print(f"Standard deviation: {s / np.sqrt(N)}")
    print(f"Confidence interval 95%: [{p - 1.96 * s / np.sqrt(N)}, {p + 1.96 * s / np.sqrt(N)}]")
    print(f"Error: {100 * 1.96 * s / (p * np.sqrt(N))}%")

    # Real BS price
    print(f"==================================================")
    bs_price = black_scholes_call_price(s0, r, sigma, K, T)
    print(f"Exact Black-Scholes Price: {bs_price}")


if __name__ == "__main__":
    main()
