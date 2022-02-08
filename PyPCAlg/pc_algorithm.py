from itertools import combinations

import copy

import numpy as np
import pandas as pd

from PyPCAlg.utilities.logs import create_logger
from PyPCAlg.utilities.pc_algorithm import find_adjacent_vertices, \
    find_adjacent_vertices_to, find_unshielded_triples
from PyPCAlg.meeks_rules import apply_Meeks_rules

field_pc_cpdag = 'CPDAG'
field_separation_sets = 'SeparationSets'


def run_pc_adjacency_phase(data: pd.DataFrame, indep_test_func: callable,
                           cond_indep_test_func: callable,
                           level: float,
                           log_file: str = '') -> tuple[np.ndarray, dict]:
    """
    Runs the adjacency phase of the PC algorithm, producing the causal
    skeleton and the separation sets.

    Parameters
    ----------
    data : pandas.DataFrame
        The observations.
    indep_test_func : callable
        A function to perform unconditional independence testing.
    cond_indep_test_func : callable
        A function to perform conditional independence testing.
    level : float
        The level for the tests.
    log_file : str, optional
        The path to a file in which to store the log. No log will be generated
        if the empty string is provided.

    Returns
    -------
    tuple
        A tuple containing the causal skeleton as first element, and a
        dictionary of the separation sets as second element.
    """

    # To deal with matters of logging
    logging_active = False
    if log_file != '':
        logging_active = True
        logger = create_logger(
            logger_name='pc_alg_adjacency_phase',
            log_file=log_file
        )

    nb_obs, nb_var = data.shape

    causal_skeleton = np.ones((nb_var, nb_var)) - np.identity(nb_var)
    separation_sets = dict()
    for x in range(nb_var):
        for y in range(x + 1, nb_var):
            separation_sets[(x, y)] = set()
            separation_sets[(y, x)] = set()

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
                        separation_sets[(x, y)].add(tuple())
                        separation_sets[(y, x)].add(tuple())

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
                            separation_sets[(x, y)].add(tuple(sorted(z)))
                            separation_sets[(y, x)].add(tuple(sorted(z)))

        depth += 1

        if stop_condition:
            break

    return causal_skeleton, separation_sets


def run_pc_orientation_phase(causal_skeleton: np.ndarray,
                             separation_sets: dict,
                             log_file: str = '') -> np.ndarray:
    """
    Runs the adjacency phase of the PC algorithm, producing the Completed
    Partially Directed Acyclic Graph (CPDAG) of the true causal graph (i.e.
    the graphical representation of the Markov equivalence class of the true
    graph).

    Parameters
    ----------
    causal_skeleton : array_like
        The causal skeleton of the true causal graph.
    separation_sets : dict
        The separation sets.
    log_file : str, optional
        The path to a file in which to store the log. No log will be generated
        if the empty string is provided.

    Returns
    -------
    array_like
        The CPDAG.
    """

    # To deal with matters of logging
    logging_active = False
    if log_file != '':
        logging_active = True
        logger = create_logger(
            logger_name='pc_alg_orientation_phase',
            log_file=log_file
        )

    cpdag = copy.deepcopy(causal_skeleton)

    # Orient the unshielded triples if any
    unshielded_triples = find_unshielded_triples(
        adjacency_matrix=causal_skeleton
    )

    if logging_active:
        logger.info(f'The unshielded triples are : {unshielded_triples}')

    for (a, b, c) in unshielded_triples:
        not_in_sepset = all([b not in sep for sep in separation_sets[(a, c)]])
        if not_in_sepset:
            cpdag[b, a] = 0
            cpdag[b, c] = 0

            if logging_active:
                logger.info(f'Unshielded triple {(a, b, c)}')
                logger.info(f'{[b]} not in SepSet[{(a, c)}] = '
                            f'{separation_sets[(a, c)]}')
                logger.info(f'Removing {b} -> {a} and {b} -> {c} from graph')

    # Apply Meek's rules repeatedly until the CPDAG no longer changes
    current_cpdag = copy.deepcopy(cpdag)
    while True:

        new_cpdag = apply_Meeks_rules(
            pdag=current_cpdag,
            apply_R4=False  # Rule R4 is not necessary for the PC algorithm
        )

        if np.array_equal(new_cpdag, current_cpdag):
            break

        current_cpdag = new_cpdag

    return new_cpdag


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
        The level for the tests.
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
