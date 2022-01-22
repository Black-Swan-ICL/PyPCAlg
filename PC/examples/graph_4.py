"""
Corresponds to the Directed Acyclic Graph given in Figure 5.3 in 'Causation,
Prediction, and Search' (P. Spirtes, C. Glymour and R. Scheines ; 2nd
edition, 2000)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pingouin as pg


def return_adjacency_matrix():
    adjacency_matrix = np.asarray(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ]
    )

    return adjacency_matrix


def return_default_coefficients():
    coefficients = dict()
    coefficients['u0_mean'] = 0
    coefficients['u0_std'] = 1
    coefficients['u1_mean'] = 0
    coefficients['u1_std'] = 1
    coefficients['u2_mean'] = 0
    coefficients['u2_std'] = 1
    coefficients['u3_mean'] = 0
    coefficients['u3_std'] = 1
    coefficients['u4_mean'] = 0
    coefficients['u4_std'] = 1
    coefficients['u0_to_x0_coeff'] = 1
    coefficients['u1_to_x1_coeff'] = 1
    coefficients['u2_to_x2_coeff'] = 1
    coefficients['u3_to_x3_coeff'] = 1
    coefficients['u4_to_x4_coeff'] = 1
    coefficients['x0_to_x1_coeff'] = 1
    coefficients['x1_to_x2_coeff'] = 1
    coefficients['x2_to_x3_coeff'] = 1
    coefficients['x4_to_x2_coeff'] = 1
    coefficients['x4_to_x3_coeff'] = 1

    return coefficients


def generate_data(sample_size: int, coefficients: dict = None,
                  random_coefficients: bool = False,
                  rng: np.random.Generator = None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(182100268991040947606686875891677251809)

    if coefficients is None:
        if random_coefficients:
            pass
        else:
            coefficients = return_default_coefficients()

    u0 = rng.normal(
        loc=coefficients['u0_mean'],
        scale=coefficients['u0_std'],
        size=sample_size
    )
    u1 = rng.normal(
        loc=coefficients['u1_mean'],
        scale=coefficients['u1_std'],
        size=sample_size
    )
    u2 = rng.normal(
        loc=coefficients['u2_mean'],
        scale=coefficients['u2_std'],
        size=sample_size
    )
    u3 = rng.normal(
        loc=coefficients['u1_mean'],
        scale=coefficients['u1_std'],
        size=sample_size
    )
    u4 = rng.normal(
        loc=coefficients['u2_mean'],
        scale=coefficients['u2_std'],
        size=sample_size
    )

    x0 = coefficients['u0_to_x0_coeff'] * u0
    x1 = (
            coefficients['u1_to_x1_coeff'] * u1 +
            coefficients['x0_to_x1_coeff'] * x0
    )
    x4 = coefficients['u4_to_x4_coeff'] * u4
    x2 = (
            coefficients['u2_to_x2_coeff'] * u2 +
            coefficients['x1_to_x2_coeff'] * x1 +
            coefficients['x4_to_x2_coeff'] * x4
    )
    x3 = (
            coefficients['u3_to_x3_coeff'] * u3 +
            coefficients['x2_to_x3_coeff'] * x2 +
            coefficients['x4_to_x3_coeff'] * x4
    )

    data = pd.DataFrame()
    data['x0'] = x0
    data['x1'] = x1
    data['x2'] = x2
    data['x3'] = x3
    data['x4'] = x4

    return data


if __name__ == '__main__':
    n = 10000

    data = generate_data(n)
    print(data.head())

    data.hist(density=True)
    plt.savefig('test.png')

    print(pg.partial_corr(
        data=data,
        x='x0',
        y='x4',
        covar=['x2']
    ))
