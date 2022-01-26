"""
Corresponds to Directed Acyclic Graph X0 -> X1 <- X2.
"""
import os
import numpy as np
import pandas as pd

from PC.utlities.independence_relationships import \
    do_test_linear_independence, \
    do_test_linear_conditional_independence, \
    produce_independence_relationships, render_independence_relationships


def return_adjacency_matrix():
    adjacency_matrix = np.asarray(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0]
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
    coefficients['x2_to_x1_coeff'] = 1

    return coefficients


def generate_data(sample_size: int, coefficients: dict = None,
                  random_coefficients: bool = False,
                  rng: np.random.Generator = None) -> pd.DataFrame:

    if rng is None:
        rng = np.random.default_rng(764218522134391065607584502462157823)

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
    x2 = coefficients['u2_to_x2_coeff'] * u2
    x1 = (
        coefficients['x0_to_x1_coeff'] * x0 +
        coefficients['x2_to_x1_coeff'] * x2 +
        coefficients['u1_to_x1_coeff'] * u1
    )

    data = pd.DataFrame()
    data['x0'] = x0
    data['x1'] = x1
    data['x2'] = x2

    return data


def run_example_with_linear_tests(sample_size: int, csv_filename: str):
    data = generate_data(sample_size)
    dic_independence_relationships = produce_independence_relationships(
        data=data,
        independence_test=do_test_linear_independence,
        conditional_independence_test=do_test_linear_conditional_independence,
    )
    df_independence_relationships = render_independence_relationships(
        dic_independence_relationships,
        level=0.05
    )

    df_independence_relationships.to_csv(
        csv_filename,
        sep=';',
        index=False
    )


if __name__ == '__main__':

    sample_size = 10000
    csv_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'empirical_graph_1_independence_relationships_with_linear_tests.csv'
    )
    run_example_with_linear_tests(
        sample_size=sample_size,
        csv_filename=csv_filename
    )
