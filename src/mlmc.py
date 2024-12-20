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
    def __init__(self, eps: float, max_level: int, M: int, T: float, rng: np.random.Generator):
        self.eps = eps
        self.max_level = max_level
        self.M = M
        self.T = T
        self.rng = rng

    def estimate_call_adaptative(self, s: float, r: float, sigma: float, strike: float):
        estimator = {}
        estimator["values"] = []
        estimator["means"] = np.zeros(self.max_level + 1)
        estimator["vars"] = np.zeros(self.max_level + 1)

        h = self.T / self.M ** np.arange(self.max_level + 1)
        N = np.zeros(self.max_level + 1, dtype=int)
        initial_samples = 10_000

        # Step 1. Start with L = 0
        L = 0

        while L <= self.max_level:
            # Step 2. Estimate V_L using an initial number of N_L = 10^4 samples.
            N[L] = initial_samples

            if L == 0:
                S = s + r * s * self.T + sigma * s * self.rng.normal(scale=np.sqrt(self.T), size=N[0])
                estimator["values"].append(np.exp(-r * self.T) * call_payoff(S, strike))
                estimator["means"][0] = np.mean(estimator["values"][0])
                estimator["vars"][0] = np.var(estimator["values"][0], ddof=1)
            else:
                m_fine, m_coarse = self.M**L, self.M ** (L - 1)
                dt_fine, dt_coarse = self.T / m_fine, self.T / m_coarse

                # Generate Brownian Motion increments for level l
                # In practice, faster than `np.random.randn`
                dW_fine = self.rng.normal(scale=np.sqrt(dt_fine), size=(N[L], m_fine))

                # Summation of Brownian increments in groups of 'M' to replicate the coarser increment
                dW_coarse = np.sum(dW_fine.reshape(N[L], m_coarse, self.M), axis=2)

                S_fine = sde_path(s, r, sigma, m_fine, dt_fine, dW_fine)
                S_coarse = sde_path(s, r, sigma, m_coarse, dt_coarse, dW_coarse)

                payoff_fine = call_payoff(S_fine, strike)
                payoff_coarse = call_payoff(S_coarse, strike)

                estimator["values"].append(np.exp(-r * self.T) * (payoff_fine - payoff_coarse))
                estimator["means"][L] = np.mean(estimator["values"][L])
                estimator["vars"][L] = np.var(estimator["values"][L], ddof=1)

            # Step 3. Define optimal N_l, l = 0, 1, ..., L, using the Equation (12)
            new_N = np.zeros(self.max_level + 1, dtype=int)
            for l in range(L + 1):
                c1 = 2 * self.eps ** (-2) * np.sqrt(estimator["vars"][l] * h[l])
                c2 = np.sum(np.sqrt(estimator["vars"][: L + 1] / h[: L + 1]))
                new_N[l] = np.ceil(c1 * c2).astype(int)

            # print(f"L: {L} | New Nl: {new_N}")

            # Step 4. Evaluate extra samples at each level as needed for new N_l
            for l in range(L + 1):
                if N[L] > new_N[l]:
                    # Truncate
                    estimator["values"][l] = estimator["values"][l][:new_N[l]]
                    estimator["means"][l] = np.mean(estimator["values"][l])
                    estimator["vars"][l] = np.var(estimator["values"][l], ddof=1)
                    N[l] = new_N[l]
                    continue

                extra_samples = new_N[l] - N[l]
                N[l] = new_N[l]
                if l == 0:
                    S = s + r * s * self.T + sigma * s * self.rng.normal(scale=np.sqrt(self.T), size=extra_samples)
                    np.append(estimator["values"][0], np.exp(-r * self.T) * call_payoff(S, strike))
                    estimator["means"][0] = np.mean(estimator["values"][0])
                    estimator["vars"][0] = np.var(estimator["values"][0], ddof=1)
                else:
                    m_fine, m_coarse = self.M**l, self.M ** (l - 1)
                    dt_fine, dt_coarse = self.T / m_fine, self.T / m_coarse

                    # Generate Brownian Motion increments for level l
                    # In practice, faster than `np.random.randn`
                    dW_fine = self.rng.normal(scale=np.sqrt(dt_fine), size=(extra_samples, m_fine))

                    # Summation of Brownian increments in groups of 'M' to replicate the coarser increment
                    dW_coarse = np.sum(dW_fine.reshape(extra_samples, m_coarse, self.M), axis=2)

                    S_fine = sde_path(s, r, sigma, m_fine, dt_fine, dW_fine)
                    S_coarse = sde_path(s, r, sigma, m_coarse, dt_coarse, dW_coarse)

                    payoff_fine = call_payoff(S_fine, strike)
                    payoff_coarse = call_payoff(S_coarse, strike)

                    np.append(estimator["values"][L], np.exp(-r * self.T) * (payoff_fine - payoff_coarse))
                    estimator["means"][L] = np.mean(estimator["values"][L])
                    estimator["vars"][L] = np.var(estimator["values"][L], ddof=1)

            # Step 5. If L >= 2, test for convergence using Equation (10) or (11)
            if L >= 2:
                # Equation (10)
                test = max(np.abs(estimator["means"][L - 1]) / self.M, np.abs(estimator["means"][L]))
                if test < (self.M - 1) * self.eps / np.sqrt(2):
                    break

                # Equation (11)
                # diff = np.abs(estimator["means"][L] - estimator["means"][L - 1] / self.M)
                # if diff < (self.M**2 - 1) * self.eps / np.sqrt(2):
                #     break

            # Step 6. If L < 2, or it is not converged, increase L by 1 and go to Step 2
            L += 1

        return estimator, N, L

    def estimate_call(self, s: float, r: float, sigma: float, strike: float, N: int):
        L = self.max_level
        M = self.M
        T = self.T

        estimators = []
        estimators_mean = np.zeros(L + 1)
        estimators_var = np.zeros(L + 1)

        # ---- Level 0 ----
        S = s + r * s * T + sigma * s * self.rng.normal(scale=np.sqrt(T), size=N)

        estimators.append(np.exp(-r * T) * call_payoff(S, strike))
        estimators_mean[0] = np.mean(estimators[0])
        estimators_var[0] = np.var(estimators[0], ddof=1)

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
            estimators_var[l] = np.var(estimators[l], ddof=1)

        return estimators_mean, estimators_var


def main() -> None:
    r = 0.05
    sigma = 0.2
    s0 = 1.0
    K = 1.0
    T = 1.0

    # MLMC parameters
    L = 4
    N = 10_000
    M = 4

    # As in the paper
    Eps = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]

    # For reproducibility
    rng = np.random.default_rng()

    # MLMC
    for eps in Eps:
        print(f"Running MLMC with eps={eps:.5f}")

        mlmc_european = MLMCEuropean(eps, L, M, T, rng)
        estimator, N, maxL = mlmc_european.estimate_call_adaptative(s0, r, sigma, K)

        print(N)
        print(np.sum(estimator["means"]))
        print(np.sum(estimator["vars"]))
        print()
    return

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
