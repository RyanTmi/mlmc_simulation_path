import numpy as np
from abc import ABC, abstractmethod

# ======================================================================================================================
# Model
# ======================================================================================================================


class Model(ABC):
    def __init__(self, interest_rate: float, initial_value: float | np.ndarray) -> None:
        self.interest_rate = interest_rate
        self.initial_value = initial_value
        self.dimension = np.size(initial_value)

    @abstractmethod
    def drift(self, t: float | np.ndarray, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def diffusion(self, t: float | np.ndarray, x: np.ndarray) -> np.ndarray:
        pass


# ======================================================================================================================
# Black-Scholes Model
# ======================================================================================================================


class BlackScholes(Model):
    def __init__(self, interest_rate: float, initial_value: float | np.ndarray, sigma: float) -> None:
        super().__init__(interest_rate, initial_value)
        self.sigma = sigma

    def drift(self, _, x: np.ndarray) -> np.ndarray:
        return self.interest_rate * x

    def diffusion(self, _, x: np.ndarray) -> np.ndarray:
        return self.sigma * x


# ======================================================================================================================
# Heston Model
# ======================================================================================================================


class Heston(Model):
    def __init__(
        self,
        interest_rate: float,
        initial_value: float | np.ndarray,
        sigma: float,
        lbd: float,
        xi: float,
        rho: float,
    ) -> None:
        super().__init__(interest_rate, initial_value)
        self.sigma = sigma
        self.lbd = lbd
        self.xi = xi
        self.rho = rho

    def drift(self, _, x: np.ndarray) -> np.ndarray:
        return np.array([self.interest_rate * x[0], self.lbd * (self.sigma**2 - x[1])])

    def diffusion(self, _, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(x[1], 0))
        return np.array([[x[0] * std, 0], [self.rho * self.xi * std, np.sqrt(1 - self.rho**2) * self.xi * std]])
