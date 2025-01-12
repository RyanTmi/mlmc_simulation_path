import numpy as np

from model import Model
from contract import Contract

# ======================================================================================================================
# MLMC
# ======================================================================================================================


class MLMC:
    def __init__(self, max_level: int, m: int, default_sample_count: int, rng: np.random.Generator):
        self.max_level = max_level
        self.m = m
        self.default_sample_count = default_sample_count
        self.rng = rng

    def estimate(self, target_error: float, model: Model, contract: Contract):
        estimator = {"values": [], "means": np.zeros(self.max_level + 1), "vars": np.zeros(self.max_level + 1)}

        samples_count = np.full(self.max_level + 1, self.default_sample_count, dtype=int)
        h = contract.maturity / self.m ** np.arange(self.max_level + 1)

        # Step 1. Start with L = 0
        level = 0
        while level <= self.max_level:
            # Step 2. Estimate V_L using an initial number of N_L = 10^4 samples.
            v = self._level_estimator(level, samples_count[level], model, contract)
            estimator["values"].append(v)
            estimator["means"][level] = np.mean(v)
            estimator["vars"][level] = np.var(v, ddof=1)

            # Step 3. Define optimal N_l, l = 0, 1, ..., L, using the Equation (12)
            optimal_samples_count = self._compute_optimal_samples(level, estimator["vars"], h, target_error)

            # Step 4. Evaluate extra samples at each level as needed for new N_l
            for l in range(len(optimal_samples_count)):
                if samples_count[l] > optimal_samples_count[l]:
                    samples_count[l] = optimal_samples_count[l]
                    continue

                extra_samples = optimal_samples_count[l] - samples_count[l]
                samples_count[l] += extra_samples
                v = self._level_estimator(l, extra_samples, model, contract)

                estimator["values"][l] = np.append(estimator["values"][l], v)
                estimator["means"][l] = np.mean(estimator["values"][l])
                estimator["vars"][l] = np.var(estimator["values"][l], ddof=1)

            # Step 5. If L >= 2, test for convergence using Equation (10) or (11)
            if level >= 2 and self._has_converged(target_error, estimator["means"], level):
                break

            # Step 6. If L < 2, or it is not converged, increase L by 1 and go to Step 2
            level += 1
        else:  # We have exited the loop without breaking
            # TODO: Algorithm has not converged
            level = self.max_level

        # NOTE: For the cost computation, `_level_estimator` may return it's cost (e.g. sample count, cpu time)
        costs_m = self.m ** np.arange(1, level + 1) + self.m ** np.arange(level)
        cost = samples_count[0] + np.dot(samples_count[1 : level + 1], costs_m)
        estimator["cost"] = cost

        # Truncate to the right size
        estimator["means"] = estimator["means"][: level + 1]
        estimator["vars"] = estimator["vars"][: level + 1]
        samples_count = samples_count[: level + 1]
        return estimator, samples_count, level

    def _level_estimator(self, level: int, sample_count: int, model: Model, contract: Contract):
        def build_sample_path(dt: float, dw: np.ndarray) -> np.ndarray:
            s = np.zeros((dw.shape[0], dw.shape[1] + 1))
            s[:, 0] = model.initial_value
            for i in range(dw.shape[1]):
                drift = model.drift(i * dt, s[:, i]) * dt
                diffusion = model.diffusion(i * dt, s[:, i]) * dw[:, i]
                s[:, i + 1] = s[:, i] + drift + diffusion

            return s

        maturity = contract.maturity
        discount = np.exp(-model.interest_rate * maturity)
        if level == 0:
            dw = self.rng.normal(scale=np.sqrt(maturity), size=(sample_count, 1))
            s = build_sample_path(maturity, dw)
            payoff = contract.payoff(s)
            return discount * payoff
        else:
            m_fine, m_coarse = self.m**level, self.m ** (level - 1)
            dt_fine, dt_coarse = maturity / m_fine, maturity / m_coarse

            dw_fine = self.rng.normal(scale=np.sqrt(dt_fine), size=(sample_count, m_fine))
            dw_coarse = np.sum(dw_fine.reshape(sample_count, m_coarse, self.m), axis=2)

            s_fine = build_sample_path(dt_fine, dw_fine)
            s_coarse = build_sample_path(dt_coarse, dw_coarse)

            payoff_fine = contract.payoff(s_fine)
            payoff_coarse = contract.payoff(s_coarse)
            return discount * (payoff_fine - payoff_coarse)

    def _compute_optimal_samples(self, level: int, vars: np.ndarray, h: np.ndarray, target_error: float) -> np.ndarray:
        c1 = 2 * np.sqrt(vars[: level + 1] * h[: level + 1]) / target_error**2
        c2 = np.sum(np.sqrt(vars[: level + 1] / h[: level + 1]))
        optimal_samples_count = np.ceil(c1 * c2).astype(int)
        return optimal_samples_count

    def _has_converged(self, target_error: float, means: np.ndarray, level: int) -> bool:
        if True:
            # Equation (10)
            left = max(np.abs(means[level]), np.abs(means[level - 1]) / self.m)
            right = (self.m - 1) * target_error / np.sqrt(2)
        else:
            # Equation (11)
            left = np.abs(means[level] - means[level - 1] / self.m)
            right = (self.m**2 - 1) * target_error / np.sqrt(2)
        return left < right


class GaussianGenerator:
    def __init__(self) -> None:
        self._index = 0

    def reset(self) -> None:
        self._index = 0


class MonteCarloEstimator:

    def __init__(self, max_level: int, m: int, default_sample_count: int, rng: np.random.Generator):
        self.max_level = max_level
        self.m = m
        self.default_sample_count = default_sample_count
        self.rng = rng

    def computations_for_plots(self, model: Model, contract: Contract, sample_count: int, level: int) -> dict:
        self.model = model
        self.contract = contract
        estimator = {
            "means": np.array([]),  # E[Y_l]
            "vars": np.array([]),  # Var(Y_l)
            "payoffs": np.array([]),  # P_l
        }

        for l in range(level):
            payoffs = self._sample_payoffs(sample_count, l)
            estimator["payoffs"] = np.append(estimator["payoffs"], payoffs)
            estimator["means"] = np.append(estimator["means"], np.mean(payoffs))
            estimator["vars"] = np.append(estimator["vars"], np.var(payoffs, ddof=1))

        means = estimator["means"]
        means_richardson = np.array([means[i + 1] - means[i] / self.m for i in range(level - 2)])
        estimator["means_richardson"] = means_richardson

        return estimator

    def compute_multilevel_estimator(
        self, model: Model, contract: Contract, target_error: float, richardson_extrapolation: bool
    ) -> tuple[dict, np.ndarray]:
        self.model = model
        self.contract = contract
        self.target_error = target_error
        self.richardson_extrapolation = richardson_extrapolation

        estimator = {
            "means": np.array([]),  # E[Y_l]
            "vars": np.array([]),  # Var(Y_l)
            "payoffs": np.array([]),  # P_l
        }
        h = contract.maturity / self.m ** np.arange(self.max_level + 1)

        samples = []
        level = 0
        while level <= self.max_level:
            payoffs = self._sample_payoffs(self.default_sample_count, level)
            estimator["payoffs"] = np.append(estimator["payoffs"], payoffs)
            estimator["means"] = np.append(estimator["means"], np.mean(payoffs))
            estimator["vars"] = np.append(estimator["vars"], np.var(payoffs, ddof=1))

            optimal_samples = self._compute_optimal_samples(level, estimator["vars"], h, target_error)
            for l in range(len(optimal_samples)):
                if samples[l] > optimal_samples[l]:
                    samples[l] = optimal_samples[l]
                    continue

                extra_samples = optimal_samples[l] - samples[l]
                samples[l] += extra_samples

                payoffs = self._sample_payoffs(extra_samples, l)
                estimator["payoffs"][l] = np.append(estimator["values"][l], payoffs)
                estimator["means"][l] = np.mean(estimator["values"][l])
                estimator["vars"][l] = np.var(estimator["values"][l], ddof=1)

            if level >= 2 and self._has_converged(target_error, estimator["means"], level):
                break

            level += 1

        estimator["value"] = np.sum(estimator["means"])
        if richardson_extrapolation:
            estimator["value"] += estimator["means"][-1] / (self.m - 1)

        return estimator, samples

    def compute_standard_estimator(
        self, model: Model, contract: Contract, target_error: float, richardson_extrapolation: bool
    ) -> tuple[dict, np.ndarray]:
        self.model = model
        self.contract = contract
        self.target_error = target_error
        self.richardson_extrapolation = richardson_extrapolation

        estimator = {"means": np.array([])}

        samples = []
        level = 0
        while level < self.max_level:
            dt = self.contract.maturity / self.m**level
            dw = self.rng.normal(scale=dt, size=(self.default_sample_count, self.m**level, self.model.dimension))

            s = self._build_sample_path(dt, dw)
            payoffs = self.contract.payoff(s)
            estimator["means"] = np.append(estimator["means"], np.mean(payoffs))

            var = np.var(payoffs)
            optimal_samples = int(np.ceil(2 * var / target_error**2))
            samples.append(optimal_samples)
            if optimal_samples > self.default_sample_count:
                extra_samples = optimal_samples - self.default_sample_count
                dw_extra = self.rng.normal(scale=dt, size=(extra_samples, self.m**level, self.model.dimension))
                s_extra = self._build_sample_path(dt, dw_extra)
                payoffs_extra = self.contract.payoff(s_extra)
                estimator["means"][-1] = np.mean(np.concatenate((payoffs, payoffs_extra)))

            if level >= 2 and self._test_convergence_std():
                break

            level += 1

        return estimator, samples

    def _sample_payoffs(self, sample_count: int, level: int):
        maturity = self.contract.maturity
        discount = np.exp(-self.model.interest_rate * maturity)

        if level == 0:
            dw = self.rng.normal(scale=np.sqrt(maturity), size=(sample_count, 1, self.model.dimension))
            s = self._build_sample_path(maturity, dw)
            payoff = self.contract.payoff(s)
            return discount * payoff
        else:
            m_fine, m_coarse = self.m**level, self.m ** (level - 1)
            dt_fine, dt_coarse = maturity / m_fine, maturity / m_coarse

            dw_fine = self.rng.normal(scale=np.sqrt(dt_fine), size=(sample_count, m_fine, self.model.dimension))
            dw_coarse = np.sum(dw_fine.reshape(sample_count, m_coarse, self.m, self.model.dimension), axis=2)

            s_fine = self._build_sample_path(dt_fine, dw_fine)
            s_coarse = self._build_sample_path(dt_coarse, dw_coarse)

            payoff_fine = self.contract.payoff(s_fine)
            payoff_coarse = self.contract.payoff(s_coarse)
            return discount * (payoff_fine - payoff_coarse)

    def _build_sample_path(self, dt: float, dw: np.ndarray) -> np.ndarray:
        s = np.zeros((dw.shape[0], dw.shape[1] + 1, dw.shape[2]))
        s[:, 0] = self.model.initial_value
        for i in range(dw.shape[1]):
            drift = np.apply_along_axis(lambda x: self.model.drift(i * dt, x), axis=1, arr=s[:, i])
            diffusion = np.apply_along_axis(lambda x: self.model.diffusion(i * dt, x), axis=1, arr=s[:, i])

            s[:, i + 1] = s[:, i] + drift * dt + diffusion * dw[:, i]
            # s[:, i + 1] = s[:, i] + drift * dt + np.einsum("ijk,ik->ij", diffusion, dw[:, i])

        return s

    def _compute_optimal_samples(self, level: int, vars: np.ndarray, h: np.ndarray, target_error: float) -> np.ndarray:
        c1 = 2 * np.sqrt(vars[: level + 1] * h[: level + 1]) / target_error**2
        c2 = np.sum(np.sqrt(vars[: level + 1] / h[: level + 1]))
        optimal_samples_count = np.ceil(c1 * c2).astype(int)
        return optimal_samples_count

    def _has_converged(self, target_error: float, means: np.ndarray, level: int) -> bool:
        if self.richardson_extrapolation:
            # Equation (11)
            left = np.abs(means[level] - means[level - 1] / self.m)
            right = (self.m**2 - 1) * target_error / np.sqrt(2)
        else:
            # Equation (10)
            left = max(np.abs(means[level]), np.abs(means[level - 1]) / self.m)
            right = (self.m - 1) * target_error / np.sqrt(2)
        return left < right

    def _test_convergence_std(self, target_error: float, means: np.ndarray, level: int) -> bool:
        if self.richardson_extrapolation:
            left = max(np.abs(means[level]), np.abs(means[level - 1]) / self.m)
            right = (self.m - 1) * target_error / np.sqrt(2)
        else:
            left = max(np.abs(means[level]), np.abs(means[level - 1]) / self.m)
            right = (self.m - 1) * target_error / np.sqrt(2)
        return left < right
