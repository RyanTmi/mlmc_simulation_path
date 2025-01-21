import numpy as np
from abc import ABC, abstractmethod

# ======================================================================================================================
# Model
# ======================================================================================================================


class Model(ABC):
    @abstractmethod
    def __init__(self, interest_rate: float, initial_value: np.ndarray) -> None:
        self.dimension = initial_value.shape[0]
        self.initial_value = initial_value
        self.interest_rate = interest_rate

    @abstractmethod
    def drift(self, t: float | np.ndarray, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def diffusion(self, t: float | np.ndarray, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


# ======================================================================================================================
# Black-Scholes Model
# ======================================================================================================================


class BlackScholes(Model):
    def __init__(self, interest_rate: float, initial_value: np.ndarray, sigma: float) -> None:
        super().__init__(interest_rate, initial_value)
        self.sigma = sigma

        if self.dimension > 1:
            raise ValueError("Black-Scholes model only support one dimension processes.")

    def drift(self, _, x: np.ndarray) -> np.ndarray:
        return self.interest_rate * x

    def diffusion(self, _, x: np.ndarray) -> np.ndarray:
        return self.sigma * x[:, np.newaxis]

    def name(self) -> str:
        return "Black-Scholes"


# ======================================================================================================================
# Heston Model
# ======================================================================================================================


class Heston(Model):
    def __init__(
        self,
        interest_rate: float,
        initial_value: np.ndarray,
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
        if len(x.shape) == 1:
            return np.array([self.interest_rate * x[0], self.lbd * (self.sigma**2 - x[1])])
        else:
            d = np.zeros(x.shape)
            d[:, 0] = self.interest_rate * x[:, 0]
            d[:, 1] = self.lbd * (self.sigma**2 - x[:, 1])
            return d

    def diffusion(self, _, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            std = np.sqrt(np.maximum(x[1], 0))
            return np.array([[x[0] * std, 0], [self.rho * self.xi * std, np.sqrt(1 - self.rho**2) * self.xi * std]])
        else:
            std = np.sqrt(np.maximum(x[:, 1], 0))
            d = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
            d[:, 0, 0] = x[:, 0] * std
            d[:, 1, 0] = self.rho * self.xi * std
            d[:, 1, 1] = np.sqrt(1 - self.rho**2) * self.xi * std
            return d

    def name(self) -> str:
        return "Heston"
