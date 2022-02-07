"""

"""
import copy

import numpy as np

from PC.utilities.pdag import find_children, find_parents, \
    find_undirected_neighbours, find_undirected_adjacent_pairs, \
    find_undirected_non_adjacent_pairs


def apply_rule_R1(pdag: np.ndarray) -> np.ndarray:

    new_pdag = copy.deepcopy(pdag)

    non_adjacent_vertices = find_undirected_non_adjacent_pairs(pdag=pdag)

    for a, c in non_adjacent_vertices:

        children_a = find_children(pdag=new_pdag,
                                   node=a)

        undirected_neighbours_c = find_undirected_neighbours(pdag=new_pdag,
                                                             node=c)

        eligible = children_a.intersection(undirected_neighbours_c)
        for b in eligible:
            new_pdag[c, b] = 0

    return new_pdag


def apply_rule_R2(pdag: np.ndarray) -> np.ndarray:

    new_pdag = copy.deepcopy(pdag)

    adjacent_vertices = find_undirected_adjacent_pairs(new_pdag)

    for a, b in adjacent_vertices:

        children_a = find_children(pdag=new_pdag,
                                   node=a)

        parents_b = find_parents(pdag=new_pdag,
                                 node=b)

        eligible = children_a.intersection(parents_b)
        if len(eligible) > 0:
            new_pdag[b, a] = 0

    return new_pdag


def apply_rule_R3(pdag: np.ndarray) -> np.ndarray:

    new_pdag = copy.deepcopy(pdag)

    adjacent_vertices = find_undirected_adjacent_pairs(new_pdag)

    for a, b in adjacent_vertices:

        undirected_neighbours_a = find_undirected_neighbours(pdag=new_pdag,
                                                             node=a)
        nb_neighbours_a = len(undirected_neighbours_a)
        if nb_neighbours_a < 2:
            continue
        else:
            non_adjacent_neighbours_a = set()
            for i in range(nb_neighbours_a):
                for j in range(i + 1, nb_neighbours_a):
                    if new_pdag[i, j] == 0 and new_pdag[j, i] == 0:
                        non_adjacent_neighbours_a.add((i, j))
            if len(non_adjacent_neighbours_a) < 1:
                continue
            else:
                parents_b = find_parents(pdag=new_pdag,
                                         node=b)
                for c, d in non_adjacent_neighbours_a:
                    if c in parents_b and d in parents_b:
                        new_pdag[b, a] = 0
                        break

    return new_pdag


def apply_rule_R4(pdag: np.ndarray) -> np.ndarray:

    new_pdag = copy.deepcopy(pdag)

    adjacent_vertices = find_undirected_adjacent_pairs(new_pdag)

    for a, b in adjacent_vertices:

        undirected_neighbours_a = find_undirected_neighbours(pdag=new_pdag,
                                                             node=a)

        parents_b = find_parents(pdag=new_pdag,
                                 node=b)

        eligible_ds = undirected_neighbours_a.intersection(parents_b)
        stop = False
        for d in eligible_ds:

            parents_d = find_parents(pdag=new_pdag,
                                     node=d)

            proto_eligible_cs = parents_d.intersection(undirected_neighbours_a)
            eligible_cs = set()
            for c in proto_eligible_cs:
                if new_pdag[c, b] == 0 and new_pdag[b, c] == 0:
                    eligible_cs.add(c)
                    new_pdag[b, a] = 0
                    stop = True
                    break

            if stop:
                break

    return new_pdag


def apply_Meeks_rules(pdag: np.ndarray, apply_R4: bool) -> np.ndarray:

    new_pdag = apply_rule_R1(pdag=pdag)
    new_pdag = apply_rule_R2(pdag=new_pdag)
    new_pdag = apply_rule_R3(pdag=new_pdag)

    if apply_R4:
        new_pdag = apply_rule_R4(pdag=new_pdag)

    return new_pdag
