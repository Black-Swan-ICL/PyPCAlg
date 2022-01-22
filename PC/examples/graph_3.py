"""
Corresponds to the Directed Acyclic Graph given in Figure 5.1 in 'Causation,
Prediction, and Search' (P. Spirtes, C. Glymour and R. Scheines ; 2nd
edition, 2000)
"""
import numpy as np

adjacency_matrix_example_3 = np.asarray(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]
    ]
)