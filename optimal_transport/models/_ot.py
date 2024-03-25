from ..utils import Distance

import numpy as np
from typing import Tuple, Optional, Any


class _OT:
    def __init__(
        self,
        distance: Distance
    ):
        self.dist_fn = distance
    
    def fit(
        self, 
        xs: np.ndarray, 
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        **kwargs,
    ) -> "_OT":
        pass

    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        pass