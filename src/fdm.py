import numpy as np

from contract import EuropeanContract
from model import Model


# ======================================================================================================================
# FDM
# ======================================================================================================================


class FDM:
    def __init__(self, l: int, m: int) -> None:
        self.l = l
        self.m = m


# ======================================================================================================================
# European FDM
# ======================================================================================================================


class EuropeanFDM(FDM):
    def __init__(self, l: int, m: int) -> None:
        super().__init__(l, m)

    def solve(
        self,
        model: Model,
        contract: EuropeanContract,
        xmin: float,  # alpha
        xmax: float,  # beta
        boundary_min: float | np.ndarray = 0.0,
        boundary_max: float | np.ndarray = 0.0,
        theta: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx = (xmax - xmin) / (self.l + 1)
        dt = contract.maturity / self.m
        x = np.linspace(xmin, xmax, self.l + 2)
        t = np.linspace(0, contract.maturity, self.m + 1)

        u = np.zeros((self.l + 2, self.m + 1))
        u[:, -1] = contract.payoff(x, all_paths=False)  # Terminal condition
        u[0, :-1] = boundary_min  # Lower boundary
        u[-1, :-1] = boundary_max  # Upper boundary

        x_interior = x[1:-1]
        d1_next, _, d3_next, a_next = self._build_operator(model, t[-1], x_interior, dx)
        for i in reversed(range(self.m)):
            d1_curr, _, d3_curr, a_curr = self._build_operator(model, t[i], x_interior, dx)

            c1 = np.eye(self.l) - theta * dt * a_curr
            c2 = np.eye(self.l) + (1 - theta) * dt * a_next

            rhs = np.dot(c2, u[1:-1, i + 1])
            rhs[0] += dt * (theta * d1_curr[0] * u[0, i] + (1 - theta) * d1_next[0] * u[0, i + 1])
            rhs[-1] += dt * (theta * d3_curr[-1] * u[-1, i] + (1 - theta) * d3_next[-1] * u[-1, i + 1])

            u[1:-1, i] = np.linalg.solve(c1, rhs)
            d1_next, d3_next, a_next = d1_curr, d3_curr, a_curr

        return t, x, u

    def _build_operator(
        self, model: Model, time: float, x_interior: np.ndarray, dx: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        diffusion = model.diffusion(time, x_interior)
        drift = model.drift(time, x_interior)

        alpha = 0.5 * (diffusion / dx) ** 2
        beta = 0.5 * (drift / dx)

        d1 = alpha - beta
        d2 = -(2.0 * alpha + model.interest_rate)
        d3 = alpha + beta

        a = np.diag(d1[1:], k=-1) + np.diag(d2, k=0) + np.diag(d3[:-1], k=1)
        return d1, d2, d3, a
