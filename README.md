# PyPCAlg

This repository contains a Python implementation of the original PC 
algorithm, as described in *Causation, Prediction, and Search* by P. Spirtes,
C. Glymour and R. Scheines (2nd edition, MIT Press, 2000).

## Installation
The package is available from PyPi : [PyPCAlg](https://pypi.org/project/PyPCAlg/).

To install, run 
> pip install PyPCAlg


## Note on the structure of the package

Folder **examples** contains examples of small dimensional graphs (i.e. with 
a low number of nodes) to test the PC algorithm on. 

The exhaustive lists of the (conditional) independence relationships 
satisfied by these examples (assuming both the causal Markov condition and 
causal Faithfulness) have been worked out. They are contained in files :
- examples/true_independence_relationships_graph_1.csv,
- examples/true_independence_relationships_graph_2.csv,
- examples/true_independence_relationships_graph_3.csv, and
- examples/true_independence_relationships_graph_4.csv.

In practice, the results of the PC algorithm depend on the statistical 
tests of (conditional) independence that we use. Considering the high 
number of statistical (conditional) independence tests carried out by the PC 
algorithm (even on graphs of moderate sizes), it is inevitable that some of 
these statistical tests will be erroneous (that is the whole problem of 
Multiple Hypothesis Testing). 

By providing the lists of (conditional) independence relationships satisfied 
by the examples, we make it possible to check whether the implementation of 
the PC algorithm itself is correct (indeed, things are as if we had at our 
disposal statistical tests of unconditional/conditional dependence that 
always return a correct result : no type I error, no type II error).

## Example of use

```python
from PyPCAlg.pc_algorithm import run_pc_algorithm, field_pc_cpdag, \
    field_separation_sets
from PyPCAlg.examples.graph_4 import generate_data
from PyPCAlg.examples.graph_4 import oracle_indep_test
from PyPCAlg.examples.graph_4 import oracle_cond_indep_test
from PyPCAlg.examples.graph_4 import get_adjacency_matrix


df = generate_data(sample_size=10)
independence_test_func = oracle_indep_test()
conditional_independence_test_func = oracle_cond_indep_test()

dic = run_pc_algorithm(
    data=df,
    indep_test_func=independence_test_func,
    cond_indep_test_func=conditional_independence_test_func,
    level=0.05
)
cpdag = dic[field_pc_cpdag]
separation_sets = dic[field_separation_sets]

print(f'The true causal graph is \n{get_adjacency_matrix()}')
print(f'\nThe CPDAG retrieved by PC is \n{cpdag}')
```

The example above demonstrates the use of the PC algorithm on one of the 
examples provided, using oracle independence and conditional independence
tests. The user can provide their own tests of independence / conditional 
independence ; they need only have the following signatures :
```python
def user_provided_independence_test(data: pandas.DataFrame, x: int, y: int, 
    level: float) -> bool:
    """
    Tests whether the variables X and Y with respective observations 
    data.iloc[:, x] and data.iloc[:, y] are statistically independent at 
    the level considered
    """
    # code for the independence test provided by the user goes here...
    
def user_provided_conditional_independence_test(data: pandas.DataFrame, x: int,
    y: int, z: list[int], level: float) -> bool:
    """
    Tests whether the variables X and Y with respective observations 
    data.iloc[:, x] and data.iloc[:, y] are statistically independent 
    conditionally on the variables z with observations data.iloc[:, z] at 
    the level considered.
    """
    # code for the conditional independence test provided by the user goes here...
```

## References
- *Causation, Prediction, and Search* P. Spirtes, C. Glymour and R. Scheines
(2nd edition, MIT Press, 2000)
```
@book{SpirtesGlymourScheines2000,
	author = {Spirtes, Peter and Glymour, Clark N and Scheines, Richard},
	title = {{Causation, Prediction, and Search}},
	publisher = {MIT press},
	year = {2000},
	edition = {2nd},
	series = {Adaptive Computation and Machine Learning}
}
```
