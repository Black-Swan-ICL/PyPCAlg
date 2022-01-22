import numpy as np

from numpy import typing as npt


def find_adjacent_vertices(adjacency_matrix: npt.ArrayLike) -> set[tuple]:
    """
    Finds the pairs of vertices that are adjacent in the graph.

    Parameters
    ----------
    adjacency_matrix : array_like
        The adjacency matrix of the graph

    Returns
    -------
    set
        A set of tuples. Each tuple represents a pair of vertices adjacent in
        the graph.
    """

    n = adjacency_matrix.shape[0]
    adjacent_vertices = set()
    for i in range(n):
        for j in range(i + 1, n):
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