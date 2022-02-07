import numpy as np


def find_children(pdag: np.ndarray, node: int) -> set:

    n = pdag.shape[0]

    children = set()

    for i in range(n):
        if pdag[node, i] == 1 and pdag[i, node] == 0:
            children.add(i)

    return children


def find_parents(pdag: np.ndarray, node: int) -> set:

    n = pdag.shape[0]

    parents = set()

    for i in range(n):
        if pdag[node, i] == 0 and pdag[i, node] == 1:
            parents.add(i)

    return parents


def find_undirected_neighbours(pdag: np.ndarray, node: int) -> set:

    n = pdag.shape[0]

    undirected_neighbours = set()

    for i in range(n):
        if pdag[i, node] == 1 and pdag[node, i] == 1:
            undirected_neighbours.add(i)

    return undirected_neighbours


def find_undirected_adjacent_pairs(pdag: np.ndarray) -> set:

    n = pdag.shape[0]

    adjacent_vertices = set()

    for i in range(n):
        for j in range(i + 1, n):
            if pdag[i, j] == 1 and pdag[j, i] == 1:
                adjacent_vertices.add((i, j))

    return adjacent_vertices


def find_undirected_non_adjacent_pairs(pdag: np.ndarray) -> set:

    n = pdag.shape[0]

    non_adjacent_vertices = set()

    for i in range(n):
        for j in range(i + 1, n):
            if pdag[i, j] == 0 and pdag[j, i] == 0:
                non_adjacent_vertices.add((i, j))

    return non_adjacent_vertices
