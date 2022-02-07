from itertools import combinations

import logging
import os

import numpy as np
import pandas as pd

from PC.utlities.pc_algorithm import find_adjacent_vertices, \
    find_adjacent_vertices_to

field_pc_cpdag = 'CPDAG'
field_separation_sets = 'SeparationSets'


def run_pc_adjacency_phase(data: pd.DataFrame, indep_test_func: callable,
                           cond_indep_test_func: callable,
                           level: float,
                           log_file: str = '') -> tuple[np.ndarray, dict]:

    # To deal with matters of logging
    logging_active = False
    if log_file != '':
        logging_active = True
        logger = logging.getLogger('pc_alg_adjacency_phase')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S'
        )
        file_handler = logging.FileHandler(
            log_file,
            mode='a'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    nb_obs, nb_var = data.shape

    causal_skeleton = np.ones((nb_var, nb_var)) - np.identity(nb_var)
    separation_sets = dict()

    depth = 0

    while True:

        adjacent_vertices = find_adjacent_vertices(causal_skeleton)

        if logging_active:
            logger.info('\n\n\n\n')  # just for greater readability of the log
            logger.info(f'Depth == {depth}')
            logger.info(f'Causal Skeleton :\n{causal_skeleton}')
            logger.info(
                f'Adjacent Vertices :\n{sorted(list(adjacent_vertices))}\n'
            )

        stop_condition = True
        for (x, y) in adjacent_vertices:
            adj_to_x = find_adjacent_vertices_to(x, causal_skeleton)
            adj_to_x_excl_y = [elt for elt in adj_to_x if elt != y]
            stop_condition = stop_condition and (len(adj_to_x_excl_y) < depth)

        if logging_active:
            logger.info(f'Stop condition == {stop_condition}')

        for (x, y) in adjacent_vertices:

            if logging_active:
                logger.info(f'Pair considered == {(x,y)}')

            adj_to_x = find_adjacent_vertices_to(x, causal_skeleton)
            adj_to_x_excl_y = [elt for elt in adj_to_x if elt != y]

            if logging_active:
                logger.info(f'Adjacent to {x} == {adj_to_x}')
                logger.info(f'Adjacent to {x} except {y} == {adj_to_x_excl_y}')

            if len(adj_to_x_excl_y) >= depth:

                if depth == 0:

                    if logging_active:
                        logger.info('Conditioning set considered == []')

                    x_indep_y = indep_test_func(
                        data=data,
                        x=x,
                        y=y,
                        level=level
                    )

                    if x_indep_y:

                        if logging_active:
                            logger.info(f'INDEPENDENCE FOUND : {x} _||_ {y}')

                        causal_skeleton[x, y] = 0
                        causal_skeleton[y, x] = 0

                        if (x, y) in separation_sets:
                            separation_sets[(x, y)].append([])
                        else:
                            separation_sets[(x, y)] = [[]]

                        if (y, x) in separation_sets:
                            separation_sets[(y, x)].append([])
                        else:
                            separation_sets[(y, x)] = [[]]

                else:

                    for z in combinations(adj_to_x_excl_y, depth):

                        if logging_active:
                            logger.info(f'Conditioning set considered == {z}')

                        x_indep_y_given_z = cond_indep_test_func(
                            data=data,
                            x=x,
                            y=y,
                            z=list(z),
                            level=level
                        )

                        if x_indep_y_given_z:

                            if logging_active:
                                logger.info(
                                    f'INDEPENDENCE FOUND == {x} _||_ {y} | {z}'
                                )

                            causal_skeleton[x, y] = 0
                            causal_skeleton[y, x] = 0

                            if (x, y) in separation_sets:
                                separation_sets[(x, y)].append(list(z))
                            else:
                                separation_sets[(x, y)] = [list(z)]

                            if (y, x) in separation_sets:
                                separation_sets[(y, x)].append(list(z))
                            else:
                                separation_sets[(y, x)] = [list(z)]

        depth += 1

        if stop_condition:
            break

    return causal_skeleton, separation_sets


# TODO implement
def run_pc_orientation_phase(causal_skeleton: np.ndarray,
                             separation_sets: dict,
                             log_file: str = '') -> np.ndarray:

    return np.nan * np.ones_like(causal_skeleton)


def run_pc_algorithm(data: pd.DataFrame, indep_test_func: callable,
                     cond_indep_test_func: callable, level: float,
                     log_file: str = '') -> dict:
    """
    Runs the original PC algorithm.

    Parameters
    ----------
    data : pandas.DataFrame
        The observations.
    indep_test_func : callable
        A function to perform unconditional independence testing.
    cond_indep_test_func : callable
        A function to perform conditional independence testing.
    level : float
        The level for the tests
    log_file : str, optional
        The path to a file in which to store the log. No log will be generated
        if the empty string is provided.

    Returns
    -------
    dict
        A dictionary containing the CPDAG obtained by running the PC algorithm
        as well as the separation sets determined on the way.
    """

    causal_skeleton, separation_sets = run_pc_adjacency_phase(
        data=data,
        indep_test_func=indep_test_func,
        cond_indep_test_func=cond_indep_test_func,
        level=level,
        log_file=log_file
    )

    cpdag = run_pc_orientation_phase(
        causal_skeleton=causal_skeleton,
        separation_sets=separation_sets,
        log_file=log_file
    )

    res = dict()
    res[field_pc_cpdag] = cpdag
    res[field_separation_sets] = separation_sets

    return res


if __name__ == '__main__':

    log_file = 'log_pc.log'
    # os.remove(log_file)

    from PC.examples.graph_4 import generate_data as generate_data_example_4
    from PC.examples.graph_4 import oracle_indep_test as \
        oracle_indep_test_example_4
    from PC.examples.graph_4 import oracle_cond_indep_test as \
        oracle_cond_indep_test_example_4

    skeleton, separation_sets = run_pc_adjacency_phase(
        data=generate_data_example_4(10),
        indep_test_func=oracle_indep_test_example_4(),
        cond_indep_test_func=oracle_cond_indep_test_example_4(),
        level=0.05,
        log_file=''
    )
    print(skeleton)
    print(separation_sets)
