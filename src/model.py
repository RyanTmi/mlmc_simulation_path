import numpy as np
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, interest_rate: float, initial_value: float | np.ndarray) -> None:
        self.interest_rate = interest_rate
        self.initial_value = initial_value
        self.dimension = np.size(initial_value)
        if self.dimension > 1:
            raise ValueError("Only scalar processes are supported")

    @abstractmethod
    def drift(self, t: float, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def diffusion(self, t: float, x: np.ndarray) -> np.ndarray:
        pass


class BlackScholes(Model):
    def __init__(self, interest_rate: float, initial_value: float | np.ndarray, sigma: float) -> None:
        super().__init__(interest_rate, initial_value)
        self.sigma = sigma

    def drift(self, _, x: np.ndarray) -> np.ndarray:
        return self.interest_rate * x

    def diffusion(self, _, x: np.ndarray) -> np.ndarray:
        return self.sigma * x
