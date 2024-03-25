from ._ot import _OT
from ..utils import Distance, SquaredEuclidean

import numpy as np
from typing import Optional
import ot

class EMD(_OT):
    def __init__(
        self,
        distance: Distance = SquaredEuclidean,
        max_iters: int = 10000,
        div_term : int = 1e-10
    ):
        super().__init__(distance)
        self.P_: Optional[np.ndarray] = None
        self.max_iters = max_iters
        self.div_term = div_term

    
    def fit(
        self, 
        xs: np.ndarray, 
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        **kwargs,
    ) -> "EMD":
        C = self.dist_fn(xs, xt)
        C = C / (C.max() + self.div_term)
        
        self.P_ = ot.emd(a, b, C, numItermax=self.max_iters)
        return self

    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        m = xt.shape[0]
        assert self.P_ is not None, "Should run fit() before mapping"

        return self.P_.dot(xt) / (self.P_.dot(np.ones((m, 1))) + self.div_term)

