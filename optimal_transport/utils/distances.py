import numpy as np
from typing import Optional, Union


class Distance(type):
    def __new__(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        pass


class SquaredEuclidean(Distance):
    def __new__(
        cls,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        return np.expand_dims((x**2).sum(axis=1),1) + np.expand_dims((y**2).sum(axis=1),0) - 2 * x@y.T
    
class KLDiv(Distance):
    def __new__(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        eps: 1e-10
    ) -> np.ndarray:
        return np.sum(x * np.log(x + eps) - x * np.log(y + eps), axis=-1)
    
class JSDiv(Distance):
    def __new__(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        eps: int = 1e-10
    ) -> np.ndarray:
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=0)
        return 0.5 * (KLDiv(x, (x + y) / 2, eps) + KLDiv(y, (x + y) / 2, eps))