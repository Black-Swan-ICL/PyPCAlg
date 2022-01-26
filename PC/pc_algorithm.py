from itertools import combinations

import numpy as np
import pandas as pd

from PC.utlities.pc_algorithm import find_adjacent_vertices, \
    find_adjacent_vertices_to

field_pc_cpdag = 'CPDAG'
field_separation_sets = 'SeparationSets'


# TODO give an option to log the run ?
# TODO debug, does not work
def run_pc_adjacency_phase(data: pd.DataFrame, indep_test_func: callable,
                           cond_indep_test_func: callable,
                           level: float) -> tuple[np.ndarray, dict]:

    nb_obs, nb_var = data.shape

    causal_skeleton = np.ones((nb_var, nb_var)) - np.identity(nb_var)
    separation_sets = dict()

    n = 0

    while True:

        adjacent_vertices = find_adjacent_vertices(causal_skeleton)
        print("\n------------------------------------------------------------")
        print(f"n == {n}")
        print("-------------------------------------------------------------")
        print(causal_skeleton)
        l = list(adjacent_vertices)
        l.sort()
        print(l)

        stop_condition = True
        for (x, y) in adjacent_vertices:
            adjacent_to_x = find_adjacent_vertices_to(x, causal_skeleton)
            adjacent_to_x_excl_y = [
                elt for elt in adjacent_to_x if elt != y
            ]
            stop_condition = stop_condition and (len(adjacent_to_x_excl_y) < n)

        print(f'\nStop condition == {stop_condition}')

        for (x, y) in adjacent_vertices:

            print(f"\nPair considered == {(x, y)}")

            adjacent_to_x = find_adjacent_vertices_to(x, causal_skeleton)
            adjacent_to_x_excl_y = [
                elt for elt in adjacent_to_x if elt != y
            ]
            print(f"Adjacent to {x} == {adjacent_to_x}")
            print(f"Adjacent to {x} except {y} == {adjacent_to_x_excl_y}")

            if len(adjacent_to_x_excl_y) >= n:

                if n == 0:

                    print("Conditioning set considered == []")

                    x_indep_y = indep_test_func(
                        data=data,
                        x=x,
                        y=y,
                        level=level
                    )

                    if x_indep_y:

                        print(f"INDEPENDENCE FOUND : {x} _||_ {y}")

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

                    for z in combinations(adjacent_to_x_excl_y, n):

                        print(f"Conditioning set considered == {z}")

                        x_indep_y_given_z = cond_indep_test_func(
                            data=data,
                            x=x,
                            y=y,
                            z=list(z),
                            level=level
                        )

                        if x_indep_y_given_z:

                            print(f"INDEPENDENCE FOUND == {x} _||_ {y} | {z}")

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

        n += 1

        if stop_condition:
            break

    return causal_skeleton, separation_sets


# TODO given an option to log the run ?
# TODO implement
def run_pc_orientation_phase(causal_skeleton, separation_sets):

    return np.nan * np.ones_like(causal_skeleton)


# TODO give an option to log the run ?
def run_pc_algorithm(data, indep_test_func, cond_indep_test_func, level):
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
        level=level
    )

    cpdag = run_pc_orientation_phase(
        causal_skeleton=causal_skeleton,
        separation_sets=separation_sets
    )

    res = dict()
    res[field_pc_cpdag] = cpdag
    res[field_separation_sets] = separation_sets

    return res
