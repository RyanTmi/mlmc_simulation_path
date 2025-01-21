import numpy as np
from abc import ABC, abstractmethod


# ======================================================================================================================
# Contracts
# ======================================================================================================================


class Contract(ABC):
    def __init__(self, maturity: float) -> None:
        self.maturity = maturity

    @abstractmethod
    def payoff(self, paths: np.ndarray, all_paths: bool) -> np.ndarray:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


# ======================================================================================================================
# European Contracts
# ======================================================================================================================


class EuropeanContract(Contract):
    @abstractmethod
    def __init__(self, maturity: float, strike: float):
        super().__init__(maturity)
        self.strike = strike


# ----------------------------------------------------------------------------------------------------------------------
# European Call
# ----------------------------------------------------------------------------------------------------------------------


class EuropeanCall(EuropeanContract):
    def __init__(self, maturity: float, strike: float) -> None:
        super().__init__(maturity, strike)

    def payoff(self, paths: np.ndarray, all_paths: bool = True) -> np.ndarray:
        if all_paths:
            return np.maximum(paths[:, -1, 0] - self.strike, 0.0)
        else:
            return np.maximum(paths - self.strike, 0.0)

    def name(self) -> str:
        return "European Call"


# ----------------------------------------------------------------------------------------------------------------------
# European Put
# ----------------------------------------------------------------------------------------------------------------------


class EuropeanPut(EuropeanContract):
    def __init__(self, maturity: float, strike: float) -> None:
        super().__init__(maturity, strike)

    def payoff(self, paths: np.ndarray, all_paths: bool = True) -> np.ndarray:
        if all_paths:
            return np.maximum(self.strike - paths[:, -1, 0], 0.0)
        else:
            return np.maximum(self.strike - paths, 0.0)

    def name(self) -> str:
        return "European Put"


# ======================================================================================================================
# Asian Contracts
# ======================================================================================================================


class AsianContract(Contract):
    @abstractmethod
    def __init__(self, maturity: float, strike: float):
        super().__init__(maturity)
        self.strike = strike


# ----------------------------------------------------------------------------------------------------------------------
# Asian Call
# ----------------------------------------------------------------------------------------------------------------------


class AsianCall(AsianContract):
    def __init__(self, maturity: float, strike: float, approximation_type: str = "t") -> None:
        super().__init__(maturity, strike)
        self.approximation_type = approximation_type

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        dt = self.maturity / (paths.shape[1] - 1)
        if self.approximation_type == "t":
            return np.maximum(dt * np.sum(paths[:, 1:, 0] + paths[:, :-1, 0], axis=1) / 2.0 - self.strike, 0.0)
        else:
            return np.maximum(np.mean(paths[:, :, 0], axis=1) - self.strike, 0)

    def name(self) -> str:
        return "Asian Call"


# ----------------------------------------------------------------------------------------------------------------------
# Asian Put
# ----------------------------------------------------------------------------------------------------------------------


class AsianPut(AsianContract):
    def __init__(self, maturity: float, strike: float) -> None:
        super().__init__(maturity, strike)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - np.mean(paths[:, :, 0], axis=1), 0)

    def name(self) -> str:
        return "Asian Put"


# ======================================================================================================================
# Lookback Contract
# ======================================================================================================================


class Lookback(Contract):
    def __init__(self, maturity: float, vol: float):
        super().__init__(maturity)
        self.vol = vol

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        beta = 0.5826
        dt = self.maturity / (paths.shape[1] - 1)
        correction = 1.0 - beta * self.vol * np.sqrt(dt)
        return paths[:, -1] - np.min(paths, axis=1) * correction

    def name(self) -> str:
        return "Lookback"


# ======================================================================================================================
# Digital Contract
# ======================================================================================================================


class Digital(Contract):
    def __init__(self, maturity: float):
        super().__init__(maturity)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        return np.heaviside(paths[:, -1, 0] - 1, 0)

    def name(self) -> str:
        return "Digital"
