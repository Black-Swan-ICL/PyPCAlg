"""

"""
import copy

import numpy as np


def apply_rule_R1(cpdag: np.ndarray) -> np.ndarray:

    new_cpdag = copy.deepcopy(cpdag)

    n = new_cpdag.shape[0]
    non_adjacent_vertices = set()
    for a in range(n):
        for c in range(a + 1, n):
            if new_cpdag[a, c] == 0 and new_cpdag[c, a] == 0:
                non_adjacent_vertices.add((a, c))

    for a, c in non_adjacent_vertices:

        children_a = set()
        for b in range(n):
            if new_cpdag[a, b] == 1 and new_cpdag[b, a] == 0:
                children_a.add(b)

        adjacent_c = set()
        for b in range(n):
            if new_cpdag[b, c] == 1 and new_cpdag[c, b] == 1:
                adjacent_c.add(b)

        eligible = children_a.intersection(adjacent_c)
        for b in eligible:
            new_cpdag[c, b] = 0

    return new_cpdag


def apply_rule_R2(cpdag: np.ndarray) -> np.ndarray:

    new_cpdag = copy.deepcopy(cpdag)

    n = new_cpdag.shape[0]
    adjacent_vertices = set()
    for a in range(n):
        for b in range(a + 1, n):
            if new_cpdag[a, b] == 1 and new_cpdag[b, a] == 1:
                adjacent_vertices.add((a, b))

    for a, b in adjacent_vertices:

        children_a = set()
        for c in range(n):
            if new_cpdag[a, c] == 1 and new_cpdag[c, a] == 0:
                children_a.add(c)

        parents_b = set()
        for c in range(n):
            if new_cpdag[c, b] == 1 and new_cpdag[b, c] == 0:
                parents_b.add(c)

        eligible = children_a.intersection(parents_b)
        if len(eligible) > 0:
            new_cpdag[b, a] = 0

    return new_cpdag


def apply_rule_R3(cpdag: np.ndarray) -> np.ndarray:

    new_cpdag = copy.deepcopy(cpdag)

    n = new_cpdag.shape[0]
    adjacent_vertices = set()
    for a in range(n):
        for b in range(a + 1, n):
            if new_cpdag[a, b] == 1 and new_cpdag[b, a] == 1:
                adjacent_vertices.add((a, b))

    for a, b in adjacent_vertices:

        neighbours_a = set()
        for i in range(n):
            if new_cpdag[a, i] == 1 and new_cpdag[i, a] == 1:
                neighbours_a.add(i)
        nb_neighbours_a = len(neighbours_a)
        if nb_neighbours_a < 2:
            continue
        else:
            non_adjacent_neighbours_a = set()
            for i in range(nb_neighbours_a):
                for j in range(i + 1, nb_neighbours_a):
                    if new_cpdag[i, j] == 0 and new_cpdag[j, i] == 0:
                        non_adjacent_neighbours_a.add((i, j))
            if len(non_adjacent_neighbours_a) < 1:
                continue
            else:
                parents_b = set()
                for k in range(n):
                    if new_cpdag[k, b] == 1 and new_cpdag[b, k] == 0:
                        parents_b.add(k)
                for c, d in non_adjacent_neighbours_a:
                    if c in parents_b and d in parents_b:
                        new_cpdag[b, a] = 0
                        break

    return new_cpdag


def apply_rule_R4(cpdag: np.ndarray) -> np.ndarray:

    new_cpdag = copy.deepcopy(cpdag)

    n = new_cpdag.shape[0]
    adjacent_vertices = set()
    for a in range(n):
        for b in range(a + 1, n):
            if new_cpdag[a, b] == 1 and new_cpdag[b, a] == 1:
                adjacent_vertices.add((a, b))

    for a, b in adjacent_vertices:

        neighbours_a = set()
        for i in range(n):
            if new_cpdag[a, i] == 1 and new_cpdag[i, a] == 1:
                neighbours_a.add(i)

        parents_b = set()
        for k in range(n):
            if new_cpdag[k, b] == 1 and new_cpdag[b, k] == 0:
                parents_b.add(k)

        eligible_ds = neighbours_a.intersection(parents_b)
        stop = False
        for d in eligible_ds:

            parents_d = set()
            for k in range(n):
                if new_cpdag[k, d] == 1 and new_cpdag[d, k] == 0:
                    parents_d.add(k)

            proto_eligible_cs = parents_d.intersection(neighbours_a)
            eligible_cs = set()
            for c in proto_eligible_cs:
                if new_cpdag[c, b] == 0 and new_cpdag[b, c] == 0:
                    eligible_cs.add(c)
                    new_cpdag[b, a] = 0
                    stop = True
                    break

            if stop:
                break

    return new_cpdag


def apply_Meeks_rules(cpdag: np.ndarray, apply_R4: bool) -> np.ndarray:

    new_cpdag = apply_rule_R1(cpdag=cpdag)
    new_cpdag = apply_rule_R2(cpdag=new_cpdag)
    new_cpdag = apply_rule_R3(cpdag=new_cpdag)

    if apply_R4:
        new_cpdag = apply_rule_R4(cpdag=new_cpdag)

    return new_cpdag
