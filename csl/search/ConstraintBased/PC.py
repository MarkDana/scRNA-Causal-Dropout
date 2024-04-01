from __future__ import annotations

import time
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from numpy import ndarray

from csl.graph.GraphClass import CausalGraph
from csl.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from csl.utils.cit import *
from csl.utils.PCUtils import Helper, Meek, SkeletonDiscovery, UCSepset
from csl.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge


def pc(
    data: ndarray, 
    alpha=0.05, 
    indep_test=None,
    stable: bool = True, 
    uc_rule: int = 0, 
    uc_priority: int = 2,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False, 
    show_progress: bool = True,
    node_names: List[str] | None = None,
    **kwargs
):
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    return pc_alg(data=data, node_names=node_names, alpha=alpha, indep_test=indep_test, stable=stable, uc_rule=uc_rule,
                      uc_priority=uc_priority, background_knowledge=background_knowledge, verbose=verbose,
                      show_progress=show_progress, **kwargs)


def pc_alg(
    data: ndarray,
    node_names: List[str] | None,
    alpha: float,
    indep_test,
    stable: bool,
    uc_rule: int,
    uc_priority: int,
    background_knowledge: BackgroundKnowledge | None = None,
    verbose: bool = False,
    show_progress: bool = True,
    **kwargs
) -> CausalGraph:
    """
    Perform Peter-Clark (PC) algorithm for causal discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)
    alpha : float, desired significance level of independence tests (p_value) in (0, 1)
    indep_test : str, the name of the independence test being used
            ["fisherz", "chisq", "gsq", "kci"]
           - "fisherz": Fisher's Z conditional independence test
           - "chisq": Chi-squared conditional independence test
           - "gsq": G-squared conditional independence test
           - "kci": Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    uc_rule : how unshielded colliders are oriented
           0: run uc_sepset
           1: run maxP
           2: run definiteMaxP
    uc_priority : rule of resolving conflicts between unshielded colliders
           -1: whatever is default in uc_rule
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.

    Returns
    -------
    cg : a CausalGraph object, where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    start = time.time()
    # indep_test = CIT(data, indep_test, **kwargs)
    cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, stable,
                                                background_knowledge=background_knowledge, verbose=verbose,
                                                show_progress=show_progress, node_names=node_names)

    if background_knowledge is not None:
        orient_by_background_knowledge(cg_1, background_knowledge)

    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.uc_sepset(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_1, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.maxp(cg_1, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority, background_knowledge=background_knowledge)
        else:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, background_knowledge=background_knowledge)
        cg_before = Meek.definite_meek(cg_2, background_knowledge=background_knowledge)
        cg = Meek.meek(cg_before, background_knowledge=background_knowledge)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    end = time.time()

    cg.PC_elapsed = end - start

    return cg
