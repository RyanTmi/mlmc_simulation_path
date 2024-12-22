import numpy as np
from abc import ABC, abstractmethod


class Contract(ABC):
    def __init__(self, maturity: float) -> None:
        self.maturity = maturity

    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def type(self) -> str:
        pass


class EuropeanCall(Contract):
    def __init__(self, maturity: float, strike: float) -> None:
        super().__init__(maturity)
        self.strike = strike

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        return np.maximum(paths[:, -1] - self.strike, 0.0)

    def type(self) -> str:
        return "European Call"


class EuropeanPut(Contract):
    def __init__(self, maturity: float, strike: float) -> None:
        super().__init__(maturity)
        self.strike = strike

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - paths[:, -1], 0.0)

    def type(self) -> str:
        return "European Put"


class AsianCall(Contract):
    def __init__(self, maturity: float, strike: float) -> None:
        super().__init__(maturity)
        self.strike = strike

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        return np.maximum(np.mean(paths, axis=1) - self.strike, 0)

    def type(self) -> str:
        return "Asian Call"


class AsianPut(Contract):
    def __init__(self, maturity: float, strike: float) -> None:
        super().__init__(maturity)
        self.strike = strike

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - np.mean(paths, axis=1), 0)

    def type(self) -> str:
        return "Asian Put"


class Lookback(Contract):
    def __init__(self, maturity: float, vol: float):
        super().__init__(maturity)
        self.vol = vol

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        beta = 0.5826
        dt = self.maturity / (paths.shape[1] - 1)
        correction = 1.0 - beta * self.vol * np.sqrt(dt)
        return paths[:, -1] - np.min(paths, axis=1) * correction

    def type(self) -> str:
        return "Lookback"


class Digital(Contract):
    def __init__(self, maturity: float):
        super().__init__(maturity)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        return np.heaviside(paths[:, -1] - 1, 0)

    def type(self) -> str:
        return "Digital"
