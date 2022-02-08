import numpy as np


def find_children(pdag: np.ndarray, node: int) -> set[int]:
    """
    Finds the children of a node in a Partially Directed Acyclic Graph (PDAG).

    Parameters
    ----------
    pdag : array_like
        The PDAG.
    node : int
        The node.

    Returns
    -------
    set
        The children of the node in the PDAG.
    """

    n = pdag.shape[0]

    children = set()

    for i in range(n):
        if pdag[node, i] == 1 and pdag[i, node] == 0:
            children.add(i)

    return children


def find_parents(pdag: np.ndarray, node: int) -> set[int]:
    """
    Finds the parents of a node in a Partially Directed Acyclic Graph (PDAG).

    Parameters
    ----------
    pdag : array_like
        The PDAG.
    node : int
        The node.

    Returns
    -------
    set
        The parents of the node in the PDAG.
    """

    n = pdag.shape[0]

    parents = set()

    for i in range(n):
        if pdag[node, i] == 0 and pdag[i, node] == 1:
            parents.add(i)

    return parents


def find_undirected_neighbours(pdag: np.ndarray, node: int) -> set[int]:
    """
    Finds the neighbours of a node a in a Partially Directed Acyclic Graph (
    PDAG) i.e. the nodes b such a - b in the PDAG.

    Parameters
    ----------
    pdag : array_like
        The PDAG.
    node : int
        The node.

    Returns
    -------
    set
        The neighbours of the node in the PDAG.
    """

    n = pdag.shape[0]

    undirected_neighbours = set()

    for i in range(n):
        if pdag[i, node] == 1 and pdag[node, i] == 1:
            undirected_neighbours.add(i)

    return undirected_neighbours


def find_undirected_adjacent_pairs(pdag: np.ndarray) -> set[tuple]:
    """
    Finds the pairs of nodes (a, b) in the Partially Directed Acyclic Graph
    (PDAG) such that there is an undirected edge between a and b, a - b, in
    the graph.

    Note that a - b will contribute both (a, b) and (b, a) to the set of pairs.

    Parameters
    ----------
    pdag : array_like
        The PDAG.

    Returns
    -------
    set
        The nodes that are adjacent in the graph.
    """

    n = pdag.shape[0]

    adjacent_vertices = set()

    for i in range(n):
        for j in range(i + 1, n):
            if pdag[i, j] == 1 and pdag[j, i] == 1:
                adjacent_vertices.add((i, j))
                adjacent_vertices.add((j, i))

    return adjacent_vertices


def find_undirected_non_adjacent_pairs(pdag: np.ndarray) -> set[tuple]:
    """
    Finds the pairs of nodes (a, b) in the Partially Directed Acyclic Graph
    (PDAG) such that there is no edge between a and b in the graph.

    Note that a <no edge> b will contribute both (a, b) and (b, a) to the
    set of pairs.

    Parameters
    ----------
    pdag : array_like
        The PDAG.

    Returns
    -------
    set
        The nodes that are not adjacent in the graph.
    """
    n = pdag.shape[0]

    non_adjacent_vertices = set()

    for i in range(n):
        for j in range(i + 1, n):
            if pdag[i, j] == 0 and pdag[j, i] == 0:
                non_adjacent_vertices.add((i, j))
                non_adjacent_vertices.add((j, i))

    return non_adjacent_vertices
