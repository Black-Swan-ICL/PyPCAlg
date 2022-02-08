import numpy as np
import pytest

from PyPCAlg.pc_algorithm import run_pc_adjacency_phase, \
    run_pc_orientation_phase, run_pc_algorithm, field_pc_cpdag

from PyPCAlg.examples.graph_1 import generate_data as generate_data_example_1
from PyPCAlg.examples.graph_1 import get_graph_skeleton as skeleton_example_1
from PyPCAlg.examples.graph_1 import oracle_indep_test as \
    oracle_indep_test_example_1
from PyPCAlg.examples.graph_1 import oracle_cond_indep_test as \
    oracle_cond_indep_test_example_1
from PyPCAlg.examples.graph_1 import get_cpdag as cpdag_example_1
from PyPCAlg.examples.graph_1 import get_separation_sets as \
    separation_sets_example_1

from PyPCAlg.examples.graph_2 import generate_data as generate_data_example_2
from PyPCAlg.examples.graph_2 import get_graph_skeleton as skeleton_example_2
from PyPCAlg.examples.graph_2 import oracle_indep_test as \
    oracle_indep_test_example_2
from PyPCAlg.examples.graph_2 import oracle_cond_indep_test as \
    oracle_cond_indep_test_example_2
from PyPCAlg.examples.graph_2 import get_cpdag as cpdag_example_2
from PyPCAlg.examples.graph_2 import get_separation_sets as \
    separation_sets_example_2

from PyPCAlg.examples.graph_3 import generate_data as generate_data_example_3
from PyPCAlg.examples.graph_3 import get_graph_skeleton as skeleton_example_3
from PyPCAlg.examples.graph_3 import oracle_indep_test as \
    oracle_indep_test_example_3
from PyPCAlg.examples.graph_3 import oracle_cond_indep_test as \
    oracle_cond_indep_test_example_3
from PyPCAlg.examples.graph_3 import get_cpdag as cpdag_example_3
from PyPCAlg.examples.graph_3 import get_separation_sets as \
    separation_sets_example_3

from PyPCAlg.examples.graph_4 import generate_data as generate_data_example_4
from PyPCAlg.examples.graph_4 import get_graph_skeleton as skeleton_example_4
from PyPCAlg.examples.graph_4 import oracle_indep_test as \
    oracle_indep_test_example_4
from PyPCAlg.examples.graph_4 import oracle_cond_indep_test as \
    oracle_cond_indep_test_example_4
from PyPCAlg.examples.graph_4 import get_cpdag as cpdag_example_4
from PyPCAlg.examples.graph_4 import get_separation_sets as \
    separation_sets_example_4


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
                separation_sets_example_1()
        ),
        (
                generate_data_example_2(10),
                oracle_indep_test_example_2(),
                oracle_cond_indep_test_example_2(),
                0.05,
                skeleton_example_2(),
                separation_sets_example_2()
        ),
        (
                generate_data_example_3(10),
                oracle_indep_test_example_3(),
                oracle_cond_indep_test_example_3(),
                0.05,
                skeleton_example_3(),
                separation_sets_example_3()
        ),
        (
                generate_data_example_4(10),
                oracle_indep_test_example_4(),
                oracle_cond_indep_test_example_4(),
                0.05,
                skeleton_example_4(),
                separation_sets_example_4()
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

    assert np.array_equal(skeleton, expected_skeleton)
    assert separation_sets == expected_separation_sets


@pytest.mark.parametrize(
    'causal_skeleton, separation_sets, expected_cpdag',
    [
        (
                skeleton_example_1(),
                separation_sets_example_1(),
                cpdag_example_1()
        ),
        (
                skeleton_example_2(),
                separation_sets_example_2(),
                cpdag_example_2()
        ),
        (
                skeleton_example_3(),
                separation_sets_example_3(),
                cpdag_example_3()
        ),
        (
                skeleton_example_4(),
                separation_sets_example_4(),
                cpdag_example_4()
        ),
    ]
)
def test_run_pc_orientation_phase(causal_skeleton, separation_sets,
                                  expected_cpdag):

    actual_cpdag = run_pc_orientation_phase(
        causal_skeleton=causal_skeleton,
        separation_sets=separation_sets
    )

    assert np.array_equal(actual_cpdag, expected_cpdag)


@pytest.mark.parametrize(
    'data, indep_test_func, cond_indep_test_func, level, expected_cpdag',
    [
        (
                generate_data_example_1(10),
                oracle_indep_test_example_1(),
                oracle_cond_indep_test_example_1(),
                0.05,
                cpdag_example_1()
        ),
        (
                generate_data_example_2(10),
                oracle_indep_test_example_2(),
                oracle_cond_indep_test_example_2(),
                0.05,
                cpdag_example_2()
        ),
        (
                generate_data_example_3(10),
                oracle_indep_test_example_3(),
                oracle_cond_indep_test_example_3(),
                0.05,
                cpdag_example_3()
        ),
        (
                generate_data_example_4(10),
                oracle_indep_test_example_4(),
                oracle_cond_indep_test_example_4(),
                0.05,
                cpdag_example_4()
        ),
    ]
)
def test_run_pc_algorithm(data, indep_test_func, cond_indep_test_func, level,
                          expected_cpdag):

    actual_cpdag = run_pc_algorithm(
        data=data,
        indep_test_func=indep_test_func,
        cond_indep_test_func=cond_indep_test_func,
        level=level
    )[field_pc_cpdag]

    assert np.array_equal(actual_cpdag, expected_cpdag)
