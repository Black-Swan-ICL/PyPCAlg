"""
This module contains functions to apply Meek's rules, as stated in Judea
Pearl's 'Causality - Models, Reasoning, and Inference' (2009 ; 2nd edition) on
page 51, to Partially Directed Acyclic Graphs (PDAGs).
"""
import copy

import numpy as np

from PyPCAlg.utilities.pdag import find_children, find_parents, \
    find_undirected_neighbours, find_undirected_adjacent_pairs, \
    find_undirected_non_adjacent_pairs


def apply_rule_R1(pdag: np.ndarray) -> np.ndarray:
    """
    Applies Meek's rule R1 to a Partially Directed Acyclic Graph (PDAG).

    In full : 'orient b - c into b -> c whenever there is an arrow a -> b such
    that a and c are nonadjacent'.

    Parameters
    ----------
    pdag : array_like
        The PDAG.

    Returns
    -------
    array_like
        The PDAG after applying rule R1.
    """

    new_pdag = copy.deepcopy(pdag)

    non_adjacent_vertices = find_undirected_non_adjacent_pairs(pdag=pdag)

    for a, c in non_adjacent_vertices:

        children_a = find_children(pdag=new_pdag,
                                   node=a)

        undirected_neighbours_c = find_undirected_neighbours(pdag=new_pdag,
                                                             node=c)

        eligible = children_a.intersection(undirected_neighbours_c)
        for b in eligible:
            new_pdag[c, b] = 0

    return new_pdag


def apply_rule_R2(pdag: np.ndarray) -> np.ndarray:
    """
    Applies Meek's rule R2 to a Partially Directed Acyclic Graph (PDAG).

    In full : 'orient a — b into a -> b whenever there is chain a -> c -> b'.

    Parameters
    ----------
    pdag : array_like
        The PDAG.

    Returns
    -------
    array_like
        The PDAG after applying rule R2.
    """

    new_pdag = copy.deepcopy(pdag)

    adjacent_vertices = find_undirected_adjacent_pairs(new_pdag)

    for a, b in adjacent_vertices:

        children_a = find_children(pdag=new_pdag,
                                   node=a)

        parents_b = find_parents(pdag=new_pdag,
                                 node=b)

        eligible = children_a.intersection(parents_b)

        if len(eligible) > 0:
            new_pdag[b, a] = 0

    return new_pdag


def apply_rule_R3(pdag: np.ndarray) -> np.ndarray:
    """
    Applies Meek's rule R3 to a Partially Directed Acyclic Graph (PDAG).

    In full : 'orient a — b into a -> b whenever there are two chains
    a — c -> b and a — d -> b such that c and d are nonadjacent'.

    Parameters
    ----------
    pdag : array_like
        The PDAG.

    Returns
    -------
    array_like
        The PDAG after applying rule R3.
    """

    new_pdag = copy.deepcopy(pdag)

    adjacent_vertices = find_undirected_adjacent_pairs(new_pdag)

    for a, b in adjacent_vertices:

        undirected_neighbours_a = find_undirected_neighbours(pdag=new_pdag,
                                                             node=a)
        if b in undirected_neighbours_a:
            undirected_neighbours_a.remove(b)
        undirected_neighbours_a = tuple(undirected_neighbours_a)
        nb_neighbours_a = len(undirected_neighbours_a)
        if nb_neighbours_a < 2:
            continue
        else:
            non_adjacent_neighbours_a = set()
            for i in range(nb_neighbours_a):
                for j in range(i + 1, nb_neighbours_a):
                    c = undirected_neighbours_a[i]
                    d = undirected_neighbours_a[j]
                    if new_pdag[c, d] == 0 and new_pdag[d, c] == 0:
                        non_adjacent_neighbours_a.add((c, d))
            if len(non_adjacent_neighbours_a) < 1:
                continue
            else:
                parents_b = find_parents(pdag=new_pdag,
                                         node=b)
                for c, d in non_adjacent_neighbours_a:
                    if c in parents_b and d in parents_b:
                        new_pdag[b, a] = 0
                        break

    return new_pdag


def apply_rule_R4(pdag: np.ndarray) -> np.ndarray:
    """
    Applies Meek's rule R4 to a Partially Directed Acyclic Graph (PDAG).

    In full : 'orient a — b into a -> b whenever there are two chains
    a — c -> d and c -> d -> b such that c and b are nonadjacent and a and d
    are adjacent'.

    Parameters
    ----------
    pdag : array_like
        The PDAG.

    Returns
    -------
    array_like
        The PDAG after applying rule R4.
    """

    new_pdag = copy.deepcopy(pdag)

    adjacent_vertices = find_undirected_adjacent_pairs(new_pdag)

    for a, b in adjacent_vertices:

        undirected_neighbours_a = find_undirected_neighbours(pdag=new_pdag,
                                                             node=a)

        parents_b = find_parents(pdag=new_pdag,
                                 node=b)

        eligible_ds = undirected_neighbours_a.intersection(parents_b)
        stop = False
        for d in eligible_ds:

            parents_d = find_parents(pdag=new_pdag,
                                     node=d)

            proto_eligible_cs = parents_d.intersection(undirected_neighbours_a)
            eligible_cs = set()
            for c in proto_eligible_cs:
                if new_pdag[c, b] == 0 and new_pdag[b, c] == 0:
                    eligible_cs.add(c)
                    new_pdag[b, a] = 0
                    stop = True
                    break

            if stop:
                break

    return new_pdag


def apply_Meeks_rules(pdag: np.ndarray, apply_R4: bool) -> np.ndarray:
    """
    Applies Meek's rules to a Partially Directed Acyclic Graph (PDAG).

    Parameters
    ----------
    pdag : array_like
        The PDAG.
    apply_R4 : bool, optional
        Whether to apply Meek's rule R4 (not necessary for the PC algorithm).

    Returns
    -------
    array_like
        The PDAG after applying Meek's rules.
    """

    new_pdag = apply_rule_R1(pdag=pdag)
    new_pdag = apply_rule_R2(pdag=new_pdag)
    new_pdag = apply_rule_R3(pdag=new_pdag)

    if apply_R4:
        new_pdag = apply_rule_R4(pdag=new_pdag)

    return new_pdag
