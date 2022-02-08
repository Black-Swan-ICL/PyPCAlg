# PyPCAlg

This repository contains a Python implementation of the original PC algorithm.

# Structure of the package

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
