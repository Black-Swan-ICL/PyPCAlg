"""
Corresponds to Directed Acyclic Graph X0 -> X1 -> X2.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pingouin as pg


def return_adjacency_matrix():
    adjacency_matrix = np.asarray(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
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
    coefficients['u0_to_x0_coeff'] = 1
    coefficients['u1_to_x1_coeff'] = 1
    coefficients['u2_to_x2_coeff'] = 1
    coefficients['x0_to_x1_coeff'] = 1
    coefficients['x1_to_x2_coeff'] = 1

    return coefficients


def generate_data(sample_size: int, coefficients: dict = None,
                  random_coefficients: bool = False,
                  rng: np.random.Generator = None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(278634375013283489033619102949855519832)

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

    x0 = coefficients['u0_to_x0_coeff'] * u0
    x1 = (
            coefficients['u1_to_x1_coeff'] * u1 +
            coefficients['x0_to_x1_coeff'] * x0
    )
    x2 = (
            coefficients['u2_to_x2_coeff'] * u2 +
            coefficients['x1_to_x2_coeff'] * x1
    )

    data = pd.DataFrame()
    data['x0'] = x0
    data['x1'] = x1
    data['x2'] = x2

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
        y='x2',
        covar='x1'
    ))