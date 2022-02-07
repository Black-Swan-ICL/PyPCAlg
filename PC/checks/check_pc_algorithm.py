"""
The checks conducted here are conducted with some examples where the variables
have multivariate Gaussian distribution. It makes it easier because we can then
rely on correlation/partial correlation to test for independence.
"""
import pandas as pd

from PC.pc_algorithm import *

from PC.utilities.independence_relationships import \
    do_test_linear_independence, do_test_linear_conditional_independence

from PC.examples.graph_1 import generate_data as generate_data_example_1
from PC.examples.graph_2 import generate_data as generate_data_example_2
from PC.examples.graph_3 import generate_data as generate_data_example_3

from PC.examples.graph_4 import generate_data as generate_data_example_4
from PC.examples.graph_4 import oracle_indep_test as \
    oracle_indep_test_example_4
from PC.examples.graph_4 import oracle_cond_indep_test as \
    oracle_cond_indep_test_example_4

if __name__ == '__main__':

    data = generate_data_example_4(1000)

    def indep_test_func(data: pd.DataFrame, x: int, y: int, level: float):

        column_names = list(data.columns)
        
        pval = do_test_linear_independence(
            data=data,
            x=column_names[x],
            y=column_names[y],
            level=level
        )

        return pval >= level, pval

    def cond_indep_test_func(data: pd.DataFrame, x: int, y: int, z: list[int],
                             level: float):

        column_names = list(data.columns)

        pval = do_test_linear_conditional_independence(
            data,
            x=column_names[x],
            y=column_names[y],
            z=[column_names[i] for i in z],
            level=level
        )

        return pval >= level, pval

    causal_skeleton, separation_sets = run_pc_adjacency_phase(
        data=data,
        # indep_test_func=indep_test_func,
        # cond_indep_test_func=cond_indep_test_func,
        indep_test_func=oracle_indep_test_example_4(),
        cond_indep_test_func=oracle_cond_indep_test_example_4(),
        level=0.1
    )

    print(causal_skeleton)
