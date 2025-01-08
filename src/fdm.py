import numpy as np

from contract import EuropeanContract
from model import Model


# ======================================================================================================================
# FDM
# ======================================================================================================================


class FDM:
    def __init__(self, l: int, m: int):
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
        xmin: float,
        xmax: float,
        boundary_min: float | np.ndarray = 0.0,
        boundary_max: float | np.ndarray = 0.0,
        theta: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        dx = (xmax - xmin) / (self.l + 1)
        dt = contract.maturity / self.m
        x = np.linspace(xmin, xmax, self.l + 2)
        t = np.linspace(0, contract.maturity, self.m + 1)

        u = np.zeros((self.l + 2, self.m + 1))
        # Terminal condition
        u[:, -1] = contract.payoff(x, all_paths=False)
        # Lower boundary
        u[0, :-1] = boundary_min
        # Upper boundary
        u[-1, :-1] = boundary_max

        f = np.zeros((self.l, self.m))
        f[0] = u[0, -1] * (model.diffusion(t, x[1]) ** 2 / (2 * dx**2) + model.drift(t, x[1]) / (2 * dx))
        f[-1] = u[-1, :-1] * (model.diffusion(t, x[-2]) ** 2 / (2 * dx**2) - model.drift(t, x[-2]) / (2 * dx))

        for i in reversed(range(self.m)):
            # sub-diagonal
            d1 = model.diffusion(i * dt, x[2:-1]) ** 2 / (2 * dx**2) - model.drift(i * dt, x[2:-1]) / (2 * dx)
            # diagonal
            d2 = -(model.interest_rate + (model.diffusion(i * dt, x[1:-1]) / dx) ** 2)
            # super-diagonal
            d3 = model.diffusion(i * dt, x[1:-2]) ** 2 / (2 * dx**2) + model.drift(i * dt, x[1:-2]) / (2 * dx)
            a = (
                np.diag(d1 * np.ones(self.l - 1), -1)
                + np.diag(d2 * np.ones(self.l))
                + np.diag(d3 * np.ones(self.l - 1), 1)
            )

            c1 = np.eye(self.l) - theta * dt * a
            c2 = np.eye(self.l) + (1 - theta) * dt * a

            u[1:-1, i] = np.dot(np.linalg.inv(c1), np.dot(c2, u[1:-1, i + 1]) + dt * f[:, i])

        return x, t, u
