import numpy as np
import pytest

from PC.pc_algorithm import run_pc_adjacency_phase, run_pc_orientation_phase, \
    run_pc_algorithm

from PC.examples.graph_1 import generate_data as generate_data_example_1
from PC.examples.graph_1 import get_graph_skeleton as skeleton_example_1
from PC.examples.graph_1 import oracle_indep_test as \
    oracle_indep_test_example_1
from PC.examples.graph_1 import oracle_cond_indep_test as \
    oracle_cond_indep_test_example_1

from PC.examples.graph_4 import generate_data as generate_data_example_4
from PC.examples.graph_4 import get_graph_skeleton as skeleton_example_4
from PC.examples.graph_4 import oracle_indep_test as \
    oracle_indep_test_example_4
from PC.examples.graph_4 import oracle_cond_indep_test as \
    oracle_cond_indep_test_example_4


@pytest.mark.parametrize(
    'data, indep_test_func, cond_indep_test_func, level, expected_skeleton, '
    'expected_separation_sets',
    [
        (
            generate_data_example_1(10),
            oracle_indep_test_example_1(),
            oracle_cond_indep_test_example_1(),
            0.05,
            skeleton_example_1(),
            set()  # TODO change when ready
        ),
        (
                generate_data_example_4(10),
                oracle_indep_test_example_4(),
                oracle_cond_indep_test_example_4(),
                0.05,
                skeleton_example_4(),
                set()  # TODO change when ready
        ),
    ]
)
def test_run_pc_adjacency_phase(data, indep_test_func, cond_indep_test_func,
                                level, expected_skeleton,
                                expected_separation_sets):

    skeleton, separation_sets = run_pc_adjacency_phase(
        data=data,
        indep_test_func=indep_test_func,
        cond_indep_test_func=cond_indep_test_func,
        level=level
    )

    assert np.allclose(skeleton, expected_skeleton)
    # TODO uncomment when ready
    # assert separation_sets == expected_separation_sets
