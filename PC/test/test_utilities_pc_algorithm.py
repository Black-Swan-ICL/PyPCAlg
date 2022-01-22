import pytest

import numpy as np

from PC.utlities.pc_algorithm import find_adjacent_vertices, \
    find_adjacent_vertices_to

# Graph with no edges
adjacency_matrix_1 = np.asarray(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
)
# Complete graph
adjacency_matrix_2 = np.ones((4, 4)) - np.identity(4)
# Unshielded triple
adjacency_matrix_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]
)


@pytest.mark.parametrize(
    "adjacency_matrix, expected_result",
    [
        (
            adjacency_matrix_1,
            set()
        ),
        (
            adjacency_matrix_2,
            {
                (0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3),
                (2, 3)
            }
        ),
        (
            adjacency_matrix_3,
            {
                (0, 1),
                (1, 2)
            }
        ),
    ]
)
def test_find_adjacent_vertices(adjacency_matrix, expected_result):

    actual_result = find_adjacent_vertices(adjacency_matrix)

    assert actual_result == expected_result


@pytest.mark.parametrize(
    "x, adjacency_matrix, expected",
    [
        (
            0,
            adjacency_matrix_1,
            []
        ),
        (
            1,
            adjacency_matrix_2,
            [0, 2, 3]
        ),
        (
            0,
            adjacency_matrix_3,
            [1]
        ),
        (
            1,
            adjacency_matrix_3,
            [0, 2]
        ),
        (
            2,
            adjacency_matrix_3,
            [1]
        )
    ]
)
def test_find_adjacent_vertices_to(x, adjacency_matrix, expected):

    pass
