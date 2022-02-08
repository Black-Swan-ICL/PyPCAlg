import numpy as np
import pytest

from PyPCAlg.meeks_rules import apply_rule_R1, apply_rule_R2, apply_rule_R3, \
    apply_rule_R4


@pytest.mark.parametrize(
    'input_pdag, expected',
    [
        (
            np.asarray([
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 0]
            ]),
            np.asarray([
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0]
            ]),
        ),
        (
            np.asarray([
                [0, 1, 1],
                [0, 0, 1],
                [1, 1, 0]
            ]),
            np.asarray([
                [0, 1, 1],
                [0, 0, 1],
                [1, 1, 0]
            ]),
        ),
    ]
)
def test_apply_rule_R1(input_pdag, expected):

    actual = apply_rule_R1(input_pdag)

    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    'input_pdag, expected',
    [
        (
            np.asarray([
                [0, 1, 1],
                [0, 0, 1],
                [1, 0, 0]
            ]),
            np.asarray([
                [0, 1, 1],
                [0, 0, 1],
                [0, 0, 0]
            ]),
        ),
    ]
)
def test_apply_rule_R2(input_pdag, expected):

    actual = apply_rule_R2(input_pdag)

    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    'input_pdag, expected',
    [
        (
            np.asarray([
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0]
            ]),
            np.asarray([
                [0, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0]
            ]),
        ),
        (
            np.asarray([
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 1, 0, 1],
                [1, 1, 1, 0]
            ]),
            np.asarray([
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 1, 0, 1],
                [1, 1, 1, 0]
            ]),
        ),
    ]
)
def test_apply_rule_R3(input_pdag, expected):

    actual = apply_rule_R3(input_pdag)

    print(actual)

    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    'input_pdag, expected',
    [
        (
            np.asarray([
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 1, 0, 0]
            ]),
            np.asarray([
                [0, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 1, 0, 0]
            ]),
        ),
        (
            np.asarray([
                [0, 1, 1, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [1, 1, 0, 0]
            ]),
            np.asarray([
                [0, 1, 1, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
                [1, 1, 0, 0]
            ]),
        ),
    ]
)
def test_apply_rule_R4(input_pdag, expected):

    actual = apply_rule_R4(input_pdag)

    print(actual)

    assert np.array_equal(actual, expected)
