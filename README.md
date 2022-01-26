# PyPCAlg

This repository contains Python implementation of the original PC algorithm.

# Structure of the package

Folder **examples** contains small dimensional (i.e. the number of nodes is
low) of graphs to test the algorithm on. 

The exhaustive lists of the (conditional) independence relationships 
satisfied by these examples (assuming both the causal Markov condition and 
causal Faithfulness) have been worked out. They are contained in files :
- examples/true_independence_relationships_graph_1.csv,
- examples/true_independence_relationships_graph_2.csv,
- examples/true_independence_relationships_graph_3.csv, and
- examples/true_independence_relationships_graph_4.csv.

This make it possible to test the algorithm with oracles of the (conditional) 
independence relationships (essentially, it is exactly as if we had 
infallible tests of independence and conditional independence). That way we
avoid problems of erroneous test results (unavoidable considering the high 
number of tests to perform, that's the problem of multiple hypothesis 
testing), and problems caused by tests the implementation of which we do 
not trust.
