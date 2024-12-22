import numpy as np

from model import Model
from contract import Contract


class MLMC:
    def __init__(self, max_level: int, m: int, default_sample_count: int, rng: np.random.Generator):
        self.max_level = max_level
        self.m = m
        self.default_sample_count = default_sample_count
        self.rng = rng

    def estimate(self, target_error: float, model: Model, contract: Contract):
        estimator = {
            "values": [],  # List of np.ndarray
            "means": np.zeros(self.max_level + 1),
            "vars": np.zeros(self.max_level + 1),
        }
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
            if level >= 2 and self._has_converged(target_error, estimator, level):
                break

            # Step 6. If L < 2, or it is not converged, increase L by 1 and go to Step 2
            level += 1
        else:  # We have exited the loop without breaking
            # TODO: Algorithm has not converged
            level = self.max_level

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

    def _has_converged(self, target_error: float, estimator: dict, level: int) -> bool:
        if True:
            # Equation (10)
            left = max(np.abs(estimator["means"][level - 1]) / self.m, np.abs(estimator["means"][level]))
            right = (self.m - 1) * target_error / np.sqrt(2)
            return left < right
        else:
            # Equation (11)
            raise NotImplementedError("Equation (11) not implemented yet")
