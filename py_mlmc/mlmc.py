import numpy as np

from contract import Contract
from model import Model
from utility import NormalGenerator

from time import perf_counter

# ======================================================================================================================
# MLMC
# ======================================================================================================================


class MLMC:
    """
    Multilevel Monte Carlo
    """

    def __init__(
        self,
        max_level: int,
        m: int,
        default_sample_count: int,
        ng: NormalGenerator,
        verbose: bool = False,
    ):
        self.max_level = max_level
        self.m = m
        self.default_sample_count = default_sample_count
        self.ng = ng
        self.verbose = verbose

    def computations_for_plots(self, model: Model, contract: Contract, sample_count: int, level: int) -> dict:
        self.ng.reset()

        estimator = {
            "payoffs": np.zeros((level, sample_count)),  # P_l - P_{l-1}
            "means": np.zeros(level),  # E[Y_l]
            "vars": np.zeros(level),  # Var(Y_l)
        }

        for i, l in enumerate(range(level)):
            if self.verbose:
                print(f"[computations_for_plots] level {l} - cost {sample_count * self.m**l:,}")

            payoffs = self._sample_payoffs(model, contract, sample_count, l)
            estimator["payoffs"][i] = payoffs
            estimator["means"][i] = np.mean(payoffs)
            estimator["vars"][i] = np.var(payoffs, ddof=1)

        estimator["means_richardson"] = np.abs(estimator["means"][1:] - estimator["means"][:-1] / self.m)
        return estimator

    def compute_multilevel_estimator(
        self, model: Model, contract: Contract, target_error: float, richardson_extrapolation: bool
    ) -> tuple[dict, np.ndarray]:
        self.ng.reset()

        estimator = {
            "means": np.array([]),  # E[Y_l]
            "vars": np.array([]),  # Var(Y_l)
            "payoffs": [],  # P_l
        }
        time_steps = contract.maturity / self.m ** np.arange(self.max_level + 1)

        samples = []
        level = 0
        while level <= self.max_level:
            samples.append(self.default_sample_count)
            payoffs = self._sample_payoffs(model, contract, self.default_sample_count, level)
            estimator["payoffs"].append(payoffs)
            estimator["means"] = np.append(estimator["means"], np.mean(payoffs))
            estimator["vars"] = np.append(estimator["vars"], np.var(payoffs, ddof=1))

            optimal_samples = self._compute_optimal_samples(level, estimator["vars"], time_steps, target_error)
            for l in range(len(optimal_samples)):
                if samples[l] > optimal_samples[l]:
                    samples[l] = optimal_samples[l]
                    continue

                extra_samples = optimal_samples[l] - samples[l]
                samples[l] += extra_samples

                payoffs = self._sample_payoffs(model, contract, extra_samples, l)
                estimator["payoffs"][l] = np.append(estimator["payoffs"][l], payoffs)
                estimator["means"][l] = np.mean(estimator["payoffs"][l])
                estimator["vars"][l] = np.var(estimator["payoffs"][l], ddof=1)

            if level >= 2 and self._has_converged(target_error, estimator["means"], level, richardson_extrapolation):
                break

            level += 1

        estimator["value"] = np.sum(estimator["means"])
        if richardson_extrapolation:
            estimator["value"] += estimator["means"][-1] / (self.m - 1)

        return estimator, samples

    def compute_standard_estimator(
        self, model: Model, contract: Contract, target_error: float, richardson_extrapolation: bool
    ) -> tuple[dict, np.ndarray]:
        self.ng.reset()

        estimator = {"means": np.array([])}

        samples = []
        level = 0
        while level < self.max_level:
            if self.verbose:
                start = perf_counter()

            dt = contract.maturity / self.m**level
            dw = self.ng.get(scale=np.sqrt(dt), size=(self.default_sample_count, self.m**level, model.dimension))

            s = self._build_sample_path(model, dt, dw)
            discount = np.exp(-model.interest_rate * contract.maturity)
            payoffs = discount * contract.payoff(s)
            estimator["means"] = np.append(estimator["means"], np.mean(payoffs))

            var = np.var(payoffs, ddof=1)
            optimal_samples = int(2 * var / target_error**2)
            samples.append(optimal_samples)
            if self.verbose:
                print(f"[compute_standard_estimator] samples {samples}")
            if optimal_samples > self.default_sample_count:
                extra_samples = optimal_samples - self.default_sample_count
                if self.verbose:
                    print(f"[compute_standard_estimator] extra_samples {extra_samples:,}")

                dw_extra = self.ng.get(scale=np.sqrt(dt), size=(extra_samples, self.m**level, model.dimension))
                s_extra = self._build_sample_path(model, dt, dw_extra)

                payoffs_extra = discount * contract.payoff(s_extra)
                estimator["means"][-1] = np.mean(np.concatenate((payoffs, payoffs_extra)))

            if level >= 2 and self._has_converged(target_error, estimator["means"], level, richardson_extrapolation):
                break

            if self.verbose:
                end = perf_counter()
                print(f"[compute_standard_estimator] level {level} ({end - start}s)")

            level += 1

        return estimator, samples

    def _sample_payoffs(self, model: Model, contract: Contract, sample_count: int, level: int) -> np.ndarray:
        maturity = contract.maturity
        discount = np.exp(-model.interest_rate * maturity)

        if level == 0:
            dw = self.ng.get(scale=np.sqrt(maturity), size=(sample_count, 1, model.dimension))
            s = self._build_sample_path(model, maturity, dw)
            payoff = contract.payoff(s)
            return discount * payoff
        else:
            m_fine, m_coarse = self.m**level, self.m ** (level - 1)
            dt_fine, dt_coarse = maturity / m_fine, maturity / m_coarse

            dw_fine = self.ng.get(scale=np.sqrt(dt_fine), size=(sample_count, m_fine, model.dimension))
            dw_coarse = np.sum(dw_fine.reshape(sample_count, m_coarse, self.m, model.dimension), axis=2)

            s_fine = self._build_sample_path(model, dt_fine, dw_fine)
            s_coarse = self._build_sample_path(model, dt_coarse, dw_coarse)

            payoff_fine = contract.payoff(s_fine)
            payoff_coarse = contract.payoff(s_coarse)
            return discount * (payoff_fine - payoff_coarse)

    def _build_sample_path(self, model: Model, dt: float, dw: np.ndarray) -> np.ndarray:
        if self.verbose:
            start = perf_counter()

        s = np.zeros((dw.shape[0], dw.shape[1] + 1, dw.shape[2]))
        s[:, 0] = model.initial_value

        for i in range(dw.shape[1]):
            if self.verbose:
                print(f"\r[_build_sample_path] size {dw.shape[0]:,} ({i + 1}/{dw.shape[1]})", flush=True, end=" ")

            # NOTE: Among these three computations, the final one is the most resource-intensive
            # and could benefit from optimization in the future.
            drift = model.drift(i * dt, s[:, i])
            diffusion = model.diffusion(i * dt, s[:, i])
            s[:, i + 1] = s[:, i] + drift * dt + np.einsum("ijk,ik->ij", diffusion, dw[:, i])

        if self.verbose:
            end = perf_counter()
            print(f"({end - start}s)")

        return s

    def _compute_optimal_samples(self, level: int, vars: np.ndarray, h: np.ndarray, target_error: float) -> np.ndarray:
        c1 = 2 * np.sqrt(vars[: level + 1] * h[: level + 1]) / target_error**2
        c2 = np.sum(np.sqrt(vars[: level + 1] / h[: level + 1]))
        optimal_samples_count = np.ceil(c1 * c2).astype(int)
        return optimal_samples_count

    def _has_converged(
        self, target_error: float, means: np.ndarray, level: int, richardson_extrapolation: bool
    ) -> bool:
        if richardson_extrapolation:
            # Equation (11)
            left = np.abs(means[level] - means[level - 1] / self.m)
            right = (self.m**2 - 1) * target_error / np.sqrt(2)
        else:
            # Equation (10)
            left = max(np.abs(means[level]), np.abs(means[level - 1]) / self.m)
            right = (self.m - 1) * target_error / np.sqrt(2)
        return left < right
