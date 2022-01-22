"""
Corresponds to Directed Acyclic Graph X0 -> X1 <- X2.
"""
import numpy as np
import pandas as pd

adjacency_matrix = np.asarray(
    [
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ]
)
