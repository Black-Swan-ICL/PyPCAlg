from collections.abc import Callable
from itertools import combinations

import copy

import numpy as np
import pandas as pd
import pingouin as pg


def do_test_linear_independence(data: pd.DataFrame, x: str, y: str,
                                level: float):
    """
    Tests for linear independence between variables x and y.

    Can be used ONLY IF the variables x and y taken jointly follow a
    multivariate Gaussian distribution.

    Parameters
    ----------
    data : pandas.DataFrame
        The observations.
    x : str
        The name of the column containing the observations for variable x.
    y : str
        The name of the column containing the observations for variable y.
    level : float
        The level of the test.

    Returns
    -------
    float
        The p-value of the test x _||_ y (where the null hypothesis is that
        independence holds).
    """

    correlation_stats = pg.corr(
        data.loc[:, x],
        data.loc[:, y],
        alternative='two-sided',
        method='pearson'
    )

    pval = correlation_stats['p-val'].values[0]

    return pval


def do_test_linear_conditional_independence(data: pd.DataFrame, x: str, y: str,
                                            z: list[str], level: float):
    """
    Tests for linear independence between variables x and y given the set of
    variables z.

    Can be used ONLY IF the variables x, y and z taken jointly follow a
    multivariate Gaussian distribution.

    Parameters
    ----------
    data : pandas.DataFrame
        The observations.
    x : str
        The name of the column containing the observations for variable x.
    y : str
        The name of the column containing the observations for variable y.
    z : list
        A list of the names of the column containing the observations for
        the variables in z.
    level : float
        The level of the test.

    Returns
    -------
    float
        The p-value of the test x _||_ y | z (where the null hypothesis is that
        conditional independence holds).
    """

    correlation_stats = pg.partial_corr(
        data=data,
        x=x,
        y=y,
        covar=z,
        alternative='two-sided',
        method='pearson'
    )

    pval = correlation_stats['p-val'].values[0]

    return pval


def produce_independence_relationships(
        data: pd.DataFrame, independence_test: Callable,
        conditional_independence_test: Callable,
        level: float = 0.05):
    """
    Performs all possible conditional independence tests for the variables in
    the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The observations.
    independence_test : callable
        A function to perform unconditional independence testing.
    conditional_independence_test : callable
        A function to perform conditional independence testing.
    level : float
        The level of the test.

    Returns
    -------
    dict
        A dictionary. Each key is a string 'x _||_ y | z' where x is any
        variable in the dataset (represented by its column name), y is any
        variable in the dataset distinct from x and z is any subset of the
        other variables (including the empty set). The values are the p-values
        for the (conditional) independence tests.
    """

    variables = list(data.columns)
    p = len(variables)

    res = dict()

    for i in range(p):
        x = variables[i]
        for j in range(i + 1, p):
            y = variables[j]

            # Test unconditional independence
            x_indep_y = independence_test(
                data=data,
                x=x,
                y=y,
                level=level
            )
            res[f'{x} _||_ {y}'] = x_indep_y

            # Test conditional independence
            other_variables = copy.deepcopy(variables)
            other_variables.remove(x)
            other_variables.remove(y)
            for r in range(1, len(other_variables) + 1):
                for conditioning_set in combinations(other_variables, r):
                    z = list(conditioning_set)
                    z.sort()
                    x_indep_y_given_z = conditional_independence_test(
                        data=data,
                        x=x,
                        y=y,
                        z=z,
                        level=level
                    )
                    res[f'{x} _||_ {y} | {z}'] = x_indep_y_given_z

    return res


def render_independence_relationships(dic: dict, level: float):
    """
    Renders the conditional independence relationships in dataframe format.

    Parameters
    ----------
    dic : dict
        A dictionary the keys of which are strings 'x _||_ y | z' and the
        values of which are the corresponding p-values in the (conditional)
        independence tests of the relationships.
    level : float
        The level of the tests.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing information on all (conditional) independence
        relationships.
    """

    keys = list(dic.keys())
    n = len(keys)

    df = pd.DataFrame()
    df['Relation'] = np.repeat(np.nan, n)
    df['p-value'] = np.repeat(np.nan, n)
    df['(Conditional) Independence Holds'] = np.repeat(np.nan, n)

    for i in range(n):
        key = keys[i]
        value = dic[key]
        df.loc[i, 'Relation'] = key
        df.loc[i, 'p-value'] = value

    df.loc[:, '(Conditional) Independence Holds'] = (
            df.loc[:, 'p-value'] >= level
    )

    df.sort_values(
        by='Relation',
        axis=0,
        ascending=True,
        inplace=True
    )

    return df
