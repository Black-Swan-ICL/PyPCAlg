import numpy as np

from numpy import typing as npt


def find_adjacent_vertices(adjacency_matrix: npt.ArrayLike) -> set[tuple]:
    """
    Finds the pairs of vertices that are adjacent in the graph.

    In the case of an undirected edge x -- y in the graph, BOTH (x, y) and
    (y, x) will appear in the adjacent vertices returned.

    Parameters
    ----------
    adjacency_matrix : array_like
        The adjacency matrix of the graph

    Returns
    -------
    set
        A set of tuples. Each tuple represents an edge in the graph.
    """

    n = adjacency_matrix.shape[0]
    adjacent_vertices = set()
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i, j] != 0:
                adjacent_vertices.add((i, j))

    return adjacent_vertices


def find_adjacent_vertices_to(x: int, adjacency_matrix: npt.ArrayLike) -> list:
    """
    Finds vertices adjacent to vertex the index of which is x.

    Parameters
    ----------
    x : int
        The index of the vertex of interest.
    adjacency_matrix : array_like
        The adjacency matrix of the graph.

    Returns
    -------
    list
        The list of vertices (their indices) adjacent in the graph to vertex
        the index of which is x.
    """

    return list(np.where(adjacency_matrix[x, :] != 0)[0])


def find_unshielded_triples(adjacency_matrix: npt.ArrayLike) -> set[tuple]:
    """
    Finds unshielded colliders in an undirected graph.

    In an undirected graph, unshielded triples are of the form a -- b -- c
    with a and c NOT adjacent in the graph. They are represented by tuples
    (a, b, c).

    Parameters
    ----------
    adjacency_matrix : array_like
        The adjacency matrix of the graph.

    Returns
    -------
    set
        The set of unshielded triples in the graph.
    """
    n = adjacency_matrix.shape[0]
    non_adjacent_vertices = set()
    for i in range(n):
        for j in range(i+1, n):
            if adjacency_matrix[i, j] == 0:
                non_adjacent_vertices.add((i, j))

    unshielded_triples = set()
    for (a, c) in non_adjacent_vertices:
        adjacent_to_a = find_adjacent_vertices_to(
            x=a,
            adjacency_matrix=adjacency_matrix
        )
        adjacent_to_c = find_adjacent_vertices_to(
            x=c,
            adjacency_matrix=adjacency_matrix
        )
        adjacent_a_and_c = set(adjacent_to_a).intersection(set(adjacent_to_c))
        for b in adjacent_a_and_c:
            unshielded_triples.add((a, b, c))

    return unshielded_triples
