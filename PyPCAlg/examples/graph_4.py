"""
Corresponds to the Directed Acyclic Graph given in Figure 5.3 in 'Causation,
Prediction, and Search' (P. Spirtes, C. Glymour and R. Scheines ; 2nd
edition, 2000)
"""
import os
import numpy as np
import pandas as pd

from PyPCAlg.examples.oracle_tools import \
    generate_oracle_independence_relationships, \
    oracle_independence_test, oracle_conditional_independence_test
from PyPCAlg.utilities.independence_relationships import \
    do_test_linear_independence, \
    do_test_linear_conditional_independence, \
    produce_independence_relationships, render_independence_relationships


def get_oracle_independence_relationships() -> dict:

    filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'true_independence_relationships_graph_4.csv'
    )

    oracle = generate_oracle_independence_relationships(filename)

    return oracle


def oracle_indep_test() -> callable:

    def res(data, x, y, level):

        oracle = get_oracle_independence_relationships()

        return oracle_independence_test(
            oracle=oracle,
            data=data,
            x=x,
            y=y,
            level=level
        )

    return res


def oracle_cond_indep_test() -> callable:

    def res(data, x, y, z, level):
        oracle = get_oracle_independence_relationships()

        return oracle_conditional_independence_test(
            oracle=oracle,
            data=data,
            x=x,
            y=y,
            z=z,
            level=level
        )

    return res


def get_graph_skeleton():
    """
    Returns the skeleton of the graph corresponding to example 4.

    Returns
    -------
    array_like
        The skeleton of the graph corresponding to example 4.
    """
    skeleton = np.asarray(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0]
        ]
    )

    return skeleton


def get_cpdag():
    """
    Returns the Markov equivalence class of the true graph as a Completed
    Partially Directed Acyclic Graph (CPDAG) for example 4.

    Returns
    -------
    array_like
        The CPDAG for example 4.
    """
    cpdag = np.asarray(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ]
    )

    return cpdag


def get_separation_sets():
    """
    Returns the separation sets as they should be returned by the PC
    algorithm for example 4.

    Returns
    -------
    dict
        A dictionary returning the separation sets as they should be returned
        by the PC algorithm for example 4.
    """
    nb_var = get_adjacency_matrix().shape[0]

    separation_sets = dict()
    for i in range(nb_var):
        for j in range(i + 1, nb_var):
            separation_sets[(i, j)] = set()
            separation_sets[(j, i)] = set()

    separation_sets[(0, 2)].add((1,))
    separation_sets[(2, 0)].add((1,))
    separation_sets[(0, 3)].add((1,))
    separation_sets[(3, 0)].add((1,))
    separation_sets[(0, 4)].add(tuple())
    separation_sets[(4, 0)].add(tuple())
    separation_sets[(1, 3)].add((2, 4))
    separation_sets[(3, 1)].add((2, 4))
    separation_sets[(1, 4)].add(tuple())
    separation_sets[(4, 1)].add(tuple())

    return separation_sets


def get_adjacency_matrix():
    """
    Returns the adjacency matrix of the graph corresponding to example 4.

    Returns
    -------
    array_like
        The adjacency matrix of the graph corresponding to example 4.
    """
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


def get_default_coefficients():
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
            coefficients = get_default_coefficients()

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
        'empirical_graph_4_independence_relationships_with_linear_tests.csv'
    )
    run_example_with_linear_tests(
        sample_size=sample_size,
        csv_filename=csv_filename
    )
