#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Haoyue
@file: main.py
@time: 3/31/23 19:54
"""
import argparse
import random, pickle
import json, os
import numpy as np
import pandas as pd
from itertools import combinations
import networkx as nx
from scipy.stats import norm, logistic
from csl.graph.SHD import SHD
from csl.utils.DAG2CPDAG import dag2cpdag
from csl.utils.TXT2GeneralGraph import dagadjmat2generalgraph
from csl.search.ConstraintBased.PC import pc
from csl.search.ScoreBased.GES import ges
from csl.utils import cit
np.random.seed(42)
random.seed(42)
import warnings
warnings.filterwarnings("ignore")

# comment out the following lines if you just want to run the algorithms with existing simulated data
import igraph as ig
import magic
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set_theme()

def sample_linear_weighted_adjmat(adjmat_binary, linear_weight_ranges):
    range_ids_flatten = np.random.randint(0, len(linear_weight_ranges), size=adjmat_binary.size)
    weight_mask = np.array([np.random.uniform(*linear_weight_ranges[range_id]) for range_id in range_ids_flatten]).reshape(adjmat_binary.shape)
    return adjmat_binary * weight_mask
def encode_ijS(i, j, S):
    less, more = min(i, j), max(i, j)
    return f'{less};{more}|{".".join(map(str, sorted(S)))}'
def decode_ijS(ijS):
    ijstr, Sstr = ijS.split('|')
    i, j = map(int, ijstr.split(';'))
    S = set(map(int, Sstr.split('.'))) if Sstr != '' else set()
    return i, j, S
rand_funcs = {
    'gaussian': np.random.normal,
    'uniform': np.random.uniform,
    'laplace': np.random.laplace,
    'exponential': np.random.exponential,
}


class DAG_Simulator(object):
    def __init__(self,
                 n_nodes=10,  # （int) number of nodes
                 graph_type='ER', # (str) 'ER', 'SF'
                 avg_degree=2,  # (float) average degree = (sum_in_degree + sum_out_degree) / n_nodes = 2 * n_edges / n_nodes
                 linear_weight_ranges=((-1, -0.25), (0.25, 1)), # (tuple) range of linear weights. uniformly sampled from
                 load_dir=None,  # (str Pathlike) load the presampled weighted adjacency matrix, noise params, and d-separation oracles
                 save_dir=None,  # (str Pathlike) save the generated graph weights and noise params
                 ):
        # graph structure and edges weights
        if load_dir is not None:
            self.adjmat_weighted = np.load(os.path.join(load_dir, 'adjmat_weighted.npy'))
            self.n_nodes = self.adjmat_weighted.shape[0]
            self.adjmat_binary = (~np.isclose(self.adjmat_weighted, 0)).astype(int)
            self.edgeslist = list(zip(*np.where(self.adjmat_binary.T)))
        else:
            self.n_nodes = n_nodes
            assert graph_type in ['ER', 'SF']
            self.edgeslist = list(random.sample(list(combinations(range(self.n_nodes), 2)),
                                min(int(round(self.n_nodes * avg_degree / 2)), self.n_nodes * (self.n_nodes - 1) // 2))) \
                                    if graph_type == 'ER' else \
                             list(zip(*np.where(np.array(ig.Graph.Barabasi(     # note: int(round(avg_degree / 2)) might be too coarse for avg_degree=1,2,3,4,5,6. no SF for now.
                                 n=self.n_nodes, m=int(round(avg_degree / 2)), directed=True).get_adjacency().data).T))) #outpref=True for ig.Graph.Barabasi?
            self.edgesarray = np.array(self.edgeslist).reshape(-1, 2)
            self.adjmat_binary = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
            self.adjmat_binary[self.edgesarray[:, 1], self.edgesarray[:, 0]] = 1
            self.adjmat_weighted = sample_linear_weighted_adjmat(self.adjmat_binary, linear_weight_ranges)
        self.parents = {x: np.where(self.adjmat_binary[x])[0] for x in range(self.n_nodes)}
        self.children = {x: np.where(self.adjmat_binary[:, x])[0] for x in range(self.n_nodes)}
        self.nxgraph = nx.DiGraph(self.adjmat_binary.T)

        # d-separation oracles
        if load_dir is not None:
            with open(os.path.join(load_dir, 'd_separation_oracles.json'), 'r') as fin:
                self.d_separation_oracles = json.load(fin)
        else:
            self.max_cond_size = min(3, n_nodes - 2)
            self.max_num_tests_per_class = 50   # i.e., need to do 2 * (1 + self.max_cond_size) * self.max_num_tests_per_class tests
            self.d_separation_oracles = {cond_size: {True: [], False: []} for cond_size in range(self.max_cond_size + 1)}
            all_XY_cond_candidates = [(x, y, S) for x in range(n_nodes)
                                                for y in range(x + 1, n_nodes)
                                                for cond_size in range(self.max_cond_size + 1)
                                                for S in combinations(set(range(n_nodes)) - {x, y}, cond_size)]
            random.shuffle(all_XY_cond_candidates)
            for x, y, S in all_XY_cond_candidates:
                d_sep_enough, d_conn_enough = len(self.d_separation_oracles[len(S)][True]) >= self.max_num_tests_per_class, \
                                              len(self.d_separation_oracles[len(S)][False]) >= self.max_num_tests_per_class
                if d_sep_enough and d_conn_enough: continue
                is_d_sep = nx.d_separated(self.nxgraph, {x}, {y}, set(S))
                if (is_d_sep and not d_sep_enough) or (not is_d_sep and not d_conn_enough):
                    self.d_separation_oracles[len(S)][is_d_sep].append(encode_ijS(x, y, S))

        if save_dir is not None and load_dir is None:
            # save the params for further preloading
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, 'adjmat_weighted.npy'), self.adjmat_weighted)

            # draw the graph structure
            figwidth = self.n_nodes * 0.5 + 2
            plt.figure(figsize=(figwidth, figwidth))
            pos = graphviz_layout(self.nxgraph, prog='dot')
            nx.draw_networkx_nodes(self.nxgraph, pos=pos, node_color='lightblue')
            nx.draw_networkx_labels(self.nxgraph, pos=pos, font_size=8)
            nx.draw_networkx_edges(self.nxgraph, pos=pos)
            nx.draw_networkx_edge_labels(self.nxgraph, pos=pos, edge_labels={e: f'{self.adjmat_weighted[e[1], e[0]]:.2f}' for e in self.edgeslist})
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'graph.pdf'), bbox_inches='tight')
            plt.close()

            # save the d-separation examples for further comparison
            with open(os.path.join(save_dir, 'd_separation_oracles.json'), 'w') as f:
                json.dump(self.d_separation_oracles, f, indent=2)


class Data_Simulator(object):
    def __init__(self,
                 dag,                           # (DAG_Simulator) simulated_dag_object
                 n_samples=1000,                # (int) number of samples
                 true_functional_form='linear', # (str) in ['linear', 'poisson']
                 exo_noise_type='gaussian',     # (str) exogenous noise type, 'gaussian', 'uniform', 'laplace', 'exponential'
                 exo_noise_params_range=((-1, 1), (1., 2.)),  # denotes gaussian's mean and stdd sampled from
                                                              # or, ((-2., -1.), (1., 2.)), uniform's low and high sampled from
                                                              # or, ((-1., 1.), (1., 2.)), laplace's mean and scale sampled from
                                                              # or, ((1., 2.),), exponential's scale sampled from
                 nonnegative_type='raw',        # (str) in ['nodewise_relu', 'final_relu', 'final_exp', 'raw']
                                                #   to describe how the gene expressions are made nonnegative. e.g., final_exp yields a lognormal.
                                                #   nodewise_* is not very useful, and is difficult to characterize its properties
                                                #   (comparing to gaussian copula) -- so it's not recommended
                 dropout_mechanism=None,        # (str) in ['latent', 'CAR', 'CAR_by_mean_Zi', 'truncated', 'probit', 'logistic',
                                                # 'post_poisson', 'post_poisson_CAR_by_mean_Zi', 'choi_ni_nodewise_ZI', None]
                 **kwargs,
                 ):
        self.dag = dag
        self.dag_generalgraph = dagadjmat2generalgraph(self.dag.adjmat_binary)
        self.cpdag_generalgraph = dag2cpdag(self.dag_generalgraph)
        self.n_samples = n_samples
        self.true_functional_form = true_functional_form
        self.exo_noise_type = exo_noise_type
        self.exo_noise_params_range = exo_noise_params_range
        self.nonnegative_type = nonnegative_type
        self.dropout_mechanism = dropout_mechanism
        self.kwargs = kwargs
        self.load_dir = kwargs.get('load_dir', None)
        self.save_dir = kwargs.get('save_dir', None)
        self.dropout_param_1 = self.kwargs.get('dropout_param_1', 0.4)
        self.dropout_param_2 = self.kwargs.get('dropout_param_2', 0.7)

        if self.load_dir is not None:
            self.data = np.load(os.path.join(self.load_dir, self.dropout_mechanism, 'data.npy'))
            self.n_samples = self.data.shape[0] # reset check?
            assert self.data.shape == (self.n_samples, self.dag.n_nodes)
            if self.dropout_mechanism != 'latent':
                self.data_imputed = np.load(os.path.join(self.load_dir, self.dropout_mechanism, 'data_imputed.npy'))
                assert self.data_imputed.shape == (self.n_samples, self.dag.n_nodes)
        else:
            if os.path.exists(os.path.join(self.save_dir, self.dropout_mechanism)):
                # remove the old data and sample the new ones
                os.system('rm -rf %s' % os.path.join(self.save_dir, self.dropout_mechanism))
            if self.dropout_mechanism == 'latent': self.simulate_true_latent_data()
            else: self.simulate_dropout_observed_data()
            if self.save_dir is not None:
                os.makedirs(os.path.join(self.save_dir, self.dropout_mechanism), exist_ok=True)
                np.save(os.path.join(self.save_dir, self.dropout_mechanism, 'data.npy'), self.data)
                if self.dropout_mechanism != 'latent':
                    np.save(os.path.join(self.save_dir, self.dropout_mechanism, 'data_imputed.npy'), self.data_imputed)
                else:
                    noise_params = (self.exo_noise_type, self.nonnegative_type, self.noise_params) \
                        if self.true_functional_form == 'linear' else (self.beta, self.gamma)
                    with open(os.path.join(self.save_dir, self.dropout_mechanism, 'noise_params.pkl'), 'wb') as f:
                        pickle.dump(noise_params, f)  # just to keep record

    ######################################################################################
    ################## 1. SIMULATE DATA ##################################################
    ######################################################################################
    def simulate_true_latent_data(self):
        # first a linear weighted adjmat, which both linear models and poisson models can use
        # some other nonlinear functional models have not been implemented yet
        assert self.dropout_mechanism == 'latent'
        def simulate_linear():
            self.noise_func = rand_funcs[self.exo_noise_type]
            self.noise_params = np.random.uniform(*self.exo_noise_params_range,
                                                  size=(self.dag.n_nodes, len(self.exo_noise_params_range)))
            def _generate_exogenous_noise(node_i, n_samples):
                return self.noise_func(*self.noise_params[node_i], size=n_samples)
            self.data = np.zeros((self.n_samples, self.dag.n_nodes))    # do not use nan. as nan * 0 = nan
            for i in list(nx.topological_sort(self.dag.nxgraph)):
                self.data[:, i] = \
                    self.data @ self.dag.adjmat_weighted[i] + _generate_exogenous_noise(i, self.n_samples)
                if self.nonnegative_type == 'nodewise_relu':
                    # here relu is to model the gene data (when a gene is inhibited, it is 0 instead of negative)
                    # note that this postprocessing of relu is different from nonnegative truncation at the final step
                    # the essential difference is that, nodewise relu still follows the DAG, while finalall relu introduces measurement error
                    self.data[:, i] = np.maximum(self.data[:, i], 0)
            if self.nonnegative_type == 'final_relu':
                self.data = np.maximum(self.data, 0)
            elif self.nonnegative_type == 'final_exp':  # i.e., the lognormal copula
                # standardize the data to prevent from numerical issues (exp of too big numbers)
                # TODO: but is it good for performance (as edge params are also rescaled)? e.g., should we do nothing, or just center the data?
                self.data = (self.data - self.data.mean(axis=0)) / self.data.std(axis=0)
                self.data = np.exp(self.data)
                # ideally, here self.noise_params should also be changed to satisfy the rescaled params.

        def simulate_poisson():
            # (ZIPBN) Choi, Junsouk, Robert Chapkin, and Yang Ni. "Bayesian causal structural learning with zero-inflated poisson bayesian networks." 2020.
            # parameters according to the paper and https://github.com/junsoukchoi/zipbn
            # There are two main differences between the followings in `simulate_dropout_observed_data` (intuitive) models and this one:
            #   1. in all settings below the dropout switch is either isolated or having only one parent (respective Zi).
            #      but in this model, the dropout switch's parents are all parents of Zi -- which may make testwise deletion CI tests biased.
            #   2. in all above settings, dropout is measurement error, i.e., a bipartite graph from Z to observed X.
            #      but in this model, dropout is iteratively inserted into the data generating process. Not sample X after each Zi is sampled.
            # The additional zero (dropout rate) is also logistic, i.e., log(eta/1-eta) = alpha^T * Z + delta.
            # also, as the 'all inhibitory parents' above, here alpha>0, i.e., also inhibitory, higher parents -> higher dropout rate. (TODO: why?)
            beta_ranges = self.kwargs.get('poisson_beta_ranges', ((-1, -0.3),))  # or use self.dag.adjmat_weighted instead?
            gamma_range = self.kwargs.get('poisson_gamma_range', (1, 3))
            self.beta = sample_linear_weighted_adjmat(self.dag.adjmat_binary, beta_ranges)
            self.gamma = np.zeros(self.dag.n_nodes)

            self.data = np.zeros((self.n_samples, self.dag.n_nodes))
            for i in list(nx.topological_sort(self.dag.nxgraph)):
                while True:
                    gamma_i = np.random.uniform(*gamma_range)
                    lamdas = np.exp(self.data @ self.beta[i] + gamma_i) + 0.5
                    # this +0.1 is to prevent from too small lamda and variable almost constant to 0 (also the var>0.1 check below)
                    # however this is just for later calculation's convenience. but the fundamental issue is not addressed: the edge weights can be too small,
                    #   i.e., a case of unfaithfulness, where the cause variables' effect is not strong enough (exp of very negative) to be detected.
                    data_i = np.random.poisson(lam=lamdas)
                    if np.var(data_i) > 0.1:
                        self.data[:, i] = data_i
                        self.gamma[i] = gamma_i
                        break
            means = self.data.mean(axis=0)
            vars = self.data.var(axis=0)
            print('mean of data:', ' '.join([f'{m:.2f}' for m in means]))
            print('var of data:', ' '.join([f'{v:.2f}' for v in vars]))

        if self.true_functional_form == 'linear': simulate_linear()
        elif self.true_functional_form == 'poisson': simulate_poisson()
        else: raise NotImplementedError

    def simulate_dropout_observed_data(self):
        true_latent_data_pth = os.path.join(self.save_dir, 'latent', 'data.npy')
        assert os.path.exists(true_latent_data_pth), f'please first sample true latent data.'
        true_latent_data = np.load(true_latent_data_pth)

        def get_CAR_dropout_mask():
            # a simple model of dropout.
            # Example 3.2 and sec 5.1 in Saeed, Basil, et al. "Anchored causal inference in the presence of measurement error." 2020.
            # Pierson, Emma, and Christopher Yau. "ZIFA: Dimensionality reduction for zero-inflated single-cell gene expression analysis." 2015.
            probs = np.random.uniform(self.dropout_param_1, self.dropout_param_2, size=self.dag.n_nodes)
            return 1. - np.random.binomial(n=1, p=probs, size=(self.n_samples, self.dag.n_nodes))

        def get_CAR_by_mean_Zi_dropout_mask():
            # Example 3.3 in Saeed, Basil, et al. "Anchored causal inference in the presence of measurement error." 2020.
            # i.e., the Michaelis-Menten model as in Andrews, Tallulah S., and Martin Hemberg. "M3Drop: dropout-based feature selection for scRNASeq." 2019.
            # though dropout rate is dependent to E[Zi], it's a constant and still independent to Zi, so still CAR.
            mean_Z = np.mean(true_latent_data, axis=0)
            assert np.all(mean_Z >= 0)
            c_ = np.mean(mean_Z)    # c_ as a constant; here just use the mean of mean_Z
            probs = 1. - mean_Z / (mean_Z + c_) # the higher the mean_Z, the lower the dropout rate
            return 1. - np.random.binomial(n=1, p=probs, size=(self.n_samples, self.dag.n_nodes))

        def get_truncated_dropout_mask():
            # definition 3 in Yoon, Grace, Raymond J. Carroll, and Irina Gaynanova. "Sparse semiparametric canonical correlation analysis for data of mixed types." 2020.
            # originated from binary indicators in Fan, Jianqing, et al. "High dimensional semiparametric latent graphical model for mixed data." 2017.
            # for more discussions on truncated (Gaussian copula model), also see
            #     Chung, Hee Cheol, Yang Ni, and Irina Gaynanova. "Sparse semiparametric discriminant analysis for high-dimensional zero-inflated data." 2022.
            #     Yang, Eunho, et al. "On Poisson graphical models." 2013.
            # to choose the threshold c_i, similar to random probs in CAR, we use percentiles 30% to 60% of ALL values (i.e., further cutoff this percentage).
            # TODO: finetune this threshold 30%~60% for a best performance gain (not dropout too less or too much)
            thresholds = np.array([np.percentile(true_latent_data[:, i], np.random.uniform(self.dropout_param_1, self.dropout_param_2)) for i in range(self.dag.n_nodes)])
            return true_latent_data > thresholds

        def get_monotone_cdf_dropout_mask():
            # sec 3.1 in Miao, Wang, Peng Ding, and Zhi Geng. "Identifiability of normal and normal mixture models with nonignorable missing data." 2016.
            # P(R = 1|y) = F(α + βy), where F is a known and strictly monotone distribution function with support on (−∞,+∞).
            # for now we choose beta=-1 (the higher value of Zi, the less likely it is dropped out) and alpha=0.
            # TODO: still we need to findtune the mono_alpha and mono_beta here for best performance gain.
            mono_alpha = self.kwargs.get('mono_alpha', self.dropout_param_1)  # in range (-inf, inf). the larger, the more dropout
            mono_beta = self.kwargs.get('mono_beta', self.dropout_param_2)  # assert mono_beta < 0, the larger abs, the more dependence between dropout and Zi value (the more distinct from CAR)
            mono_func = {'probit': norm.cdf, 'logistic': logistic.cdf}[self.dropout_mechanism]
            probs = mono_func(mono_alpha + mono_beta * true_latent_data)
            return 1. - np.random.binomial(n=1, p=probs)

        def get_post_poisson_observed_data():
            # (PLN) Xiao, Feiyi, et al. "Estimating graphical models for count data with applications to single-cell gene network." 2022.
            # also similar to Example 3.3 in Saeed, Basil, et al. "Anchored causal inference in the presence of measurement error." 2020.
            # this is not dropout actually -- it is measurement error.
            # i.e., the i-th sample(cell) of variable j, Xij|Zij ~ Poisson(Si * Zij), where Si is constants (scaling factor related to sum of counts, library size)
            # and according to the paper, Z follows lognormal.
            scaing_factors = np.ones(self.n_samples) * 1.    # just for now
            return np.random.poisson(lam=scaing_factors[:, None] * true_latent_data)

        def get_choi_ni_nodewise_ZI_dropout_data():
            alpha_ranges = self.kwargs.get('poisson_alpha_ranges', ((0.3, 1),))  # default parameters by junsoukchoi's code
            delta_range = self.kwargs.get('poisson_delta_range', (-2, -1))
            alpha = sample_linear_weighted_adjmat(self.dag.adjmat_binary, alpha_ranges)
            delta = np.random.uniform(*delta_range, size=self.dag.n_nodes)

            self.data = np.zeros((self.n_samples, self.dag.n_nodes))
            for i in list(nx.topological_sort(self.dag.nxgraph)):
                probs = logistic.cdf(self.data @ alpha[i] + delta[i])
                # i.e., etas = np.exp(dropout_data @ alpha[i] + delta[i]); etas = etas / (1 + etas).
                # here it's dropout_observed_data; the dropout zero percentage will affect later nodes.
                self.data[:, i] = true_latent_data[:, i] * (1 - np.random.binomial(n=1, p=probs))

        if self.dropout_mechanism == 'CAR': self.data = true_latent_data * get_CAR_dropout_mask()
        elif self.dropout_mechanism == 'CAR_by_mean': self.data = true_latent_data * get_CAR_by_mean_Zi_dropout_mask()
        elif self.dropout_mechanism == 'truncated': self.data = true_latent_data * get_truncated_dropout_mask()
        elif self.dropout_mechanism in ['probit', 'logistic']: self.data = true_latent_data * get_monotone_cdf_dropout_mask()
        elif self.dropout_mechanism == 'post_poisson': self.data = get_post_poisson_observed_data()
        elif self.dropout_mechanism == 'post_poisson_CAR_by_mean': self.data = get_post_poisson_observed_data() * get_CAR_by_mean_Zi_dropout_mask()
        elif self.dropout_mechanism == 'choi_ni_nodewise_ZI': self.data = get_choi_ni_nodewise_ZI_dropout_data()

        # use magic for imputation. TODO: for MAGIC we need to input the raw counts, not float data?
        magic_op = magic.MAGIC(n_jobs=-1, t='auto')
        input_to_magic = pd.DataFrame(self.data)
        self.data_imputed = magic_op.fit_transform(input_to_magic, genes='all_genes').to_numpy()

    ######################################################################################
    ################## 2. PLOT UTILITIES #################################################
    ######################################################################################
    def plot_data(self):
        os.makedirs(os.path.join(self.save_dir, self.dropout_mechanism), exist_ok=True)
        plot_title = self.kwargs.get('plot_title', 'plot_data_scatter')

        true_latent_data_pth = os.path.join(self.save_dir, 'latent', 'data.npy')
        assert os.path.exists(true_latent_data_pth), f'please first sample true latent data.'
        true_latent_data = np.load(true_latent_data_pth)

        latent_zero_rate = np.mean(np.isclose(true_latent_data, 0), axis=0)
        observed_zero_rate = np.mean(np.isclose(self.data, 0), axis=0)
        magic_imputed_zero_rate = np.mean(np.isclose(self.data_imputed, 0), axis=0)
        latent_vairable_names = [f'Z{i} zero%={100*latent_zero_rate[i]:.1f}' for i in range(self.dag.n_nodes)]
        observed_vairable_names = [f'X{i} zero%={100*observed_zero_rate[i]:.1f}' for i in range(self.dag.n_nodes)]
        magic_imputed_vairable_names = [f'X{i} zero%={100*magic_imputed_zero_rate[i]:.1f}' for i in range(self.dag.n_nodes)]

        gdatas = {
            'True Latent Data': pd.DataFrame(true_latent_data, columns=latent_vairable_names),
            'Observed Dropout Data': pd.DataFrame(self.data, columns=observed_vairable_names),
            'MAGIC Imputed Data': pd.DataFrame(self.data_imputed, columns=magic_imputed_vairable_names),
        }
        for gid, (gtitle, gdata) in enumerate(gdatas.items()):
            g = sns.pairplot(gdata, plot_kws={'alpha': 0.8})
            g.fig.suptitle(gtitle)
            g.fig.tight_layout()
            g.fig.subplots_adjust(top=0.94)
            g.savefig(f"g{gid}.jpg", dpi=300, bbox_inches='tight')
            plt.close(g.fig)

        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        for gid in range(3):
            axs[gid].imshow(mpimg.imread(f'g{gid}.jpg'))
            axs[gid].set_axis_off()

        fig.suptitle(plot_title, fontsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, self.dropout_mechanism, f'{plot_title}.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
        os.system(f'rm -rf g0.jpg g1.jpg g2.jpg')

    def plot_nonzero_samplesize(self):
        os.makedirs(os.path.join(self.save_dir, self.dropout_mechanism), exist_ok=True)
        plot_title = 'plot_data_nonzero_samplesize'

        nonzero_mask = self.data != 0
        # nonzero_size_of_vars = np.sum(nonzero_mask, axis=0)
        # nonzero_mask = nonzero_mask[:, nonzero_mask > 500]  # to directly remove genes with too few expressions.

        all_combs_pd = pd.DataFrame({'combination_size': [], 'nonzero_samplesize': []})
        for combsize in range(1, min(nonzero_mask.shape[1], 9)):
            # this can consume very large memory and is slow when #vars is large. checkout t-cell-challenge/overview/check_missingness.py for solution.
            all_subsets_of_size = list(combinations(range(nonzero_mask.shape[1]), combsize))
            all_subsets_of_size = random.sample(all_subsets_of_size, min(10000, len(all_subsets_of_size)))
            non_zero_comb_samplesizes = [np.sum(np.all(nonzero_mask[:, vids], axis=1)) for vids in all_subsets_of_size]
            all_combs_pd = all_combs_pd.append(pd.DataFrame({'combination_size': [combsize]*len(non_zero_comb_samplesizes), 'nonzero_samplesize': non_zero_comb_samplesizes}))

        # violin plot
        fig, ax = plt.subplots(figsize=(16, 7))
        sns.violinplot(x='combination_size', y='nonzero_samplesize', data=all_combs_pd, ax=ax)
        ax.set_title(plot_title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, self.dropout_mechanism, f'{plot_title}.jpg'), dpi=300, bbox_inches='tight')
        plt.close()

    ######################################################################################
    ################## 3. RUN METHODS (CITEST, PC, AND GES) ##############################
    ######################################################################################
    def run_ci_pvalues_comparison(self, method='fisherz', dname='data', pval_alpha=0.05):
        this_data_dir = os.path.join(self.load_dir, self.dropout_mechanism)
        res_pth = os.path.join(this_data_dir, f'{method}{pval_alpha}_{dname}_ci_pvalues_comparison.json')
        try:
            with open(res_pth) as f: return json.load(f)
        except Exception as e: print('==== RUN CI ERROR ====' , e)

        max_kci_sample_size = self.kwargs.get('max_kci_sample_size', 2000)
        assert dname in ['data', 'data_imputed']
        data_to_run = self.data if dname == 'data' else self.data_imputed
        tester = cit.CIT(data_to_run, method=method,
                         cache_path=os.path.join(this_data_dir, f'{method}_{dname}_cache.json'),
                         max_sample_size=max_kci_sample_size)  # max_sample_size is used only for kci

        res_dict = {condsize: {
            'dsep_precision': None,
            'dsep_recall': None,
            'dsep_f1': None,
            True: [],   # just store the empirical True or False, or we could also store samplesize, pval?
            False: []
        } for condsize in self.dag.d_separation_oracles.keys()}

        for condsize, orcs in self.dag.d_separation_oracles.items():
            for true_or_false, ijs_list in orcs.items():
                for ijS_str in ijs_list:
                    i, j, S = decode_ijS(ijS_str)
                    pval, rho, N_spsz = tester(i, j, S, return_rho=True, return_sample_size=True)
                    res_dict[condsize][true_or_false == 'true'].append((pval, rho, N_spsz))
            TP = len([1 for pval, _, _ in res_dict[condsize][True] if pval > pval_alpha])
            FP = len([1 for pval, _, _ in res_dict[condsize][False] if pval > pval_alpha])
            FN = len(res_dict[condsize][True]) - TP
            precision = TP / (TP + FP) if TP + FP > 0 else 0.
            recall = TP / (TP + FN) if TP + FN > 0 else 0.
            F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.
            res_dict[condsize]['dsep_precision'], res_dict[condsize]['dsep_recall'], res_dict[condsize]['dsep_f1'] = precision, recall, F1
        with open(res_pth, 'w') as f: json.dump(res_dict, f, indent=4)
        return res_dict

    def run_pc(self, method='fisherz', dname='data', pval_alpha=0.05):
        # dname: 'data' or 'data_imputed'. only effective for data with dropout.
        this_data_dir = os.path.join(self.load_dir, self.dropout_mechanism)
        adjmat_pth = os.path.join(this_data_dir, f'pc_{method}{pval_alpha}_{dname}_adjmat_coltorow.txt')
        res_pth = os.path.join(this_data_dir, f'pc_{method}{pval_alpha}_{dname}_res.txt')
        try:
            adjmat_est = np.loadtxt(adjmat_pth)
            cpdag_shd, skel_f1 = np.loadtxt(res_pth)
            return adjmat_est, cpdag_shd, skel_f1
        except Exception as e: print('==== RUN PC ERROR ====' , e)
        max_kci_sample_size = self.kwargs.get('max_kci_sample_size', 2000)
        assert dname in ['data', 'data_imputed']
        data_to_run = self.data if dname == 'data' else self.data_imputed
        tester = cit.CIT(data_to_run, method=method, cache_path=os.path.join(this_data_dir, f'{method}_{dname}_cache.json'),
                           max_sample_size=max_kci_sample_size) # max_sample_size is used only for kci
        cg = pc(self.data, pval_alpha, tester, verbose=False, show_progress=False)
        cpdag_shd = int(SHD(self.cpdag_generalgraph, cg.G).get_shd())
        skel_edges_true = {(i, j) if i < j else (j, i) for (i, j) in self.dag.edgeslist}
        skel_edges_estm = {(i, j) if i < j else (j, i) for (i, j) in list(zip(*np.where(cg.G.graph.T != 0)))}
        try:
            skel_true_positive_num = len(skel_edges_true.intersection(skel_edges_estm))
            skel_precision = skel_true_positive_num / len(skel_edges_estm)
            skel_recall = skel_true_positive_num / len(skel_edges_true)
            skel_f1 = 2 * skel_precision * skel_recall / (skel_precision + skel_recall)
        except ZeroDivisionError:
            skel_f1 = 0.
        np.savetxt(adjmat_pth, cg.G.graph, fmt='%d')
        np.savetxt(res_pth, np.array([cpdag_shd, skel_f1]))
        return cg.G.graph, cpdag_shd, skel_f1

    def run_ges(self, method='fisherz', dname='data', bic_l0_penal=1.):
        this_data_dir = os.path.join(self.load_dir, self.dropout_mechanism)
        adjmat_pth = os.path.join(this_data_dir, f'ges_{method}{bic_l0_penal}_{dname}_adjmat_coltorow.txt')
        res_pth = os.path.join(this_data_dir, f'ges_{method}{bic_l0_penal}_{dname}_res.txt')
        try:
            adjmat_est = np.loadtxt(adjmat_pth)
            cpdag_shd, skel_f1 = np.loadtxt(res_pth)
            return adjmat_est, cpdag_shd, skel_f1
        except Exception as e: print('==== RUN GES ERROR ====' , e)
        max_kci_sample_size = self.kwargs.get('max_kci_sample_size', 2000)
        assert dname in ['data', 'data_imputed']
        data_to_run = self.data if dname == 'data' else self.data_imputed
        tester = cit.CIT(data_to_run, method=method, cache_path=os.path.join(this_data_dir, f'{method}_{dname}_cache.json'),
                                                max_sample_size=max_kci_sample_size) # max_sample_size is used only for kci
        _, cg, _ = ges(self.data, tester, bic_l0_penal=bic_l0_penal, phases=['forward', 'backward'], iterate=False)
        cpdag_shd = int(SHD(self.cpdag_generalgraph, cg).get_shd())
        skel_edges_true = {(i, j) if i < j else (j, i) for (i, j) in self.dag.edgeslist}
        skel_edges_estm = {(i, j) if i < j else (j, i) for (i, j) in list(zip(*np.where(cg.graph.T != 0)))}
        try:
            skel_true_positive_num = len(skel_edges_true.intersection(skel_edges_estm))
            skel_precision = skel_true_positive_num / len(skel_edges_estm)
            skel_recall = skel_true_positive_num / len(skel_edges_true)
            skel_f1 = 2 * skel_precision * skel_recall / (skel_precision + skel_recall)
        except ZeroDivisionError:
            skel_f1 = 0.
        np.savetxt(adjmat_pth, cg.graph, fmt='%d')
        np.savetxt(res_pth, np.array([cpdag_shd, skel_f1]))
        return cg.graph, cpdag_shd, skel_f1


if __name__ == '__main__':
    GRAPHTYPES = ['ER']
    NODENUMS = [5, 10, 20, 30]  # nodenum=5 just for visualization and examples.
    DEGREES = lambda x: [1, 2, 3, 4, 5, 6] if x != 5 else [1.5, 3]
    DAGIDS = range(5)

    #### 1. generate DAGs #######       ##### already done #####
    for gtype in GRAPHTYPES:
        for nodenum in NODENUMS:
            for degree in DEGREES(nodenum):
                for dagid in DAGIDS:
                    dag_dir = f'./graphs/{gtype}/{nodenum}_nodes_{degree}_degree/dag_{dagid}/'
                    dag = DAG_Simulator(n_nodes=nodenum,  # （int) number of nodes
                                        graph_type=gtype,
                                        avg_degree=degree,
                                        save_dir=dag_dir)

    # ####### 2. generate data #######       ##### already done #####
    def get_dropout_params(true_functional_form, exo_noise_type, nonnegative_type, dropout_mechanism):
        if (true_functional_form, exo_noise_type, nonnegative_type) == ('linear', 'gaussian', 'raw'):
            if dropout_mechanism in ['latent', 'CAR']: return 0.4, 0.7
            elif dropout_mechanism == 'truncated': return 20, 50
            elif dropout_mechanism == 'logistic': return -0.5, -1.5
        elif (true_functional_form, exo_noise_type, nonnegative_type) in [('linear', 'gaussian', 'final_exp'), ('poisson', 'na', 'na')]:
            if dropout_mechanism in ['latent', 'CAR']: return 0.3, 0.6
            elif dropout_mechanism == 'truncated': return 20, 40
            elif dropout_mechanism == 'logistic': return 0.5, -0.5
    for gtype in GRAPHTYPES:
        for nodenum in NODENUMS:
            for degree in DEGREES(nodenum):
                for dagid in DAGIDS:
                    dag_dir = f'./graphs/{gtype}/{nodenum}_nodes_{degree}_degree/dag_{dagid}/'
                    dag = DAG_Simulator(n_nodes=nodenum,  # （int) number of nodes
                                        graph_type=gtype,
                                        avg_degree=degree,
                                        load_dir=dag_dir)

                    for true_functional_form, exo_noise_type, nonnegative_type in [
                        ('linear', 'gaussian', 'raw'),
                        ('linear', 'gaussian', 'final_exp'),
                        ('poisson', 'na', 'na')
                    ]:
                        for dropout_mechanism in ['latent', 'CAR', 'truncated', 'logistic']:
                            data_dir = os.path.join(dag_dir.replace('./graphs/', './data/'),
                                                    f'{true_functional_form}_{exo_noise_type}_{nonnegative_type}')
                            dropout_param_1, dropout_param_2 = \
                                get_dropout_params(true_functional_form, exo_noise_type, nonnegative_type, dropout_mechanism)

                            ds = Data_Simulator(dag=dag,
                                                n_samples=10000,
                                                true_functional_form=true_functional_form,
                                                exo_noise_type=exo_noise_type,
                                                nonnegative_type=nonnegative_type,
                                                dropout_mechanism=dropout_mechanism,
                                                save_dir=data_dir,
                                                dropout_param_1=dropout_param_1,
                                                dropout_param_2=dropout_param_2,
                                                )
                            if nodenum <= 5 and dagid == 0 and dropout_mechanism != 'latent': ds.plot_data()
                            ds.plot_nonzero_samplesize()


    # ####### 3. run algorithms #######
    parser = argparse.ArgumentParser()
    parser.add_argument('--dagid', type=int, default=0)    # in range(5)
    parser.add_argument('--gtype', type=str, default='ER') # ER, SF
    parser.add_argument('--funcform', type=str, default='linear')   # fixed for now
    parser.add_argument('--exonoise', type=str, default='gaussian') # fixed for now
    parser.add_argument('--nonnegative', type=str, default='raw') # raw, final_exp
    parser.add_argument('--dropouts', type=str, default='latent.CAR.truncated.logistic') # 'latent', 'CAR', 'truncated', 'logistic' or comb of them
    parser.add_argument('--pvalalpha', type=float, default=0.05)
    parser.add_argument('--bicl0penal', type=float, default=1.)

    args = parser.parse_args()
    dagid = args.dagid
    gtype = args.gtype
    true_functional_form = args.funcform
    exo_noise_type = args.exonoise
    nonnegative_type = args.nonnegative
    dropout_mechanism_lists = args.dropouts.split('.')
    pval_alpha = args.pvalalpha
    bic_l0_penal = args.bicl0penal

    # added many try pass to skip the unknown errors on slurm
    for nodenum in NODENUMS:
        for degree in DEGREES(nodenum):
                dag_dir = f'./graphs/{gtype}/{nodenum}_nodes_{degree}_degree/dag_{dagid}/'
                try:
                    dag = DAG_Simulator(n_nodes=nodenum,  # （int) number of nodes
                                    graph_type=gtype,
                                    avg_degree=degree,
                                    load_dir=dag_dir)
                except Exception as e: print('==== DAG_LOAD ERROR ====', e); continue

                for dropout_mechanism in dropout_mechanism_lists:
                    data_dir = os.path.join(dag_dir.replace('./graphs/', './data/'),
                                            f'{true_functional_form}_{exo_noise_type}_{nonnegative_type}')
                    try:
                        ds = Data_Simulator(dag=dag,
                                            n_samples=10000,
                                            true_functional_form=true_functional_form,
                                            exo_noise_type=exo_noise_type,
                                            nonnegative_type=nonnegative_type,
                                            dropout_mechanism=dropout_mechanism,
                                            load_dir=data_dir,
                                            )
                    except Exception as e: print('==== DATA_LOAD ERROR ====', e); continue

                    # obs_ci = {int(condsize): float(f"{res['dsep_f1']:.3f}") for condsize, res in
                    #                 ds.run_ci_pvalues_comparison('fisherz', 'data', pval_alpha).items()}
                    try: _, obs_pc_shd, obs_pc_f1 = ds.run_pc('fisherz', 'data', pval_alpha)
                    except Exception as e: print('==== OBS_PC ERROR ====' , e)
                    try: _, obs_ges_shd, obs_ges_f1 = ds.run_ges('fisherz', 'data', bic_l0_penal)
                    except Exception as e: print('==== OBS_GES ERROR ====' , e)

                    # we'll notice that the f1-score of citests-accuracies may be very similar acorss latent, obs, obs-0del, obs-imputed
                    # this is because 1) fisherz is already good on gaussian data (than e.g., kci), and 2) the oracle ijS pairs are randomly chosen
                    # while in terms of PC and GES, there are citests "critical" to predict edges. and 0-deletion's better performance is highlighted.
                    if dropout_mechanism == 'latent':
                        try:
                            print(f'\n\n===== {data_dir} =====')
                            print(f'  - Latent:\tPC={int(obs_pc_shd)}\tGES={int(obs_ges_shd)}') # \tCIs={obs_ci}
                        except Exception as e: print('==== PRINTLATENT ERROR ====' , e)
                    else:
                        # obs_zerodel_ci = {int(condsize): float(f"{res['dsep_f1']:.3f}") for condsize, res in
                        #           ds.run_ci_pvalues_comparison('zerodel_fisherz', 'data', pval_alpha).items()}
                        try: _, obs_zerodel_pc_shd, obs_zerodel_pc_f1 = ds.run_pc('zerodel_fisherz', 'data', pval_alpha)
                        except Exception as e: print('==== OBS_0DEL_PC RROR ====' , e)
                        try: _, obs_zerodel_ges_shd, obs_zerodel_ges_f1 = ds.run_ges('zerodel_fisherz', 'data', bic_l0_penal)
                        except Exception as e: print('==== OBS_0DEL_GES ERROR ====' , e)
                        # imputed_ci = {int(condsize): float(f"{res['dsep_f1']:.3f}") for condsize, res in
                        #                   ds.run_ci_pvalues_comparison('fisherz', 'data_imputed', pval_alpha).items()}
                        try: _, imputed_pc_shd, imputed_pc_f1 = ds.run_pc('fisherz', 'data_imputed', pval_alpha)
                        except Exception as e: print('==== IMPUTED_PC ERROR ====' , e)
                        try: _, imputed_ges_shd, imputed_ges_f1 = ds.run_ges('fisherz', 'data_imputed', bic_l0_penal)
                        except Exception as e: print('==== IMPUTED_GES ERROR ====' , e)
                        try:
                            print(f'\n  - {dropout_mechanism}:\tPC={int(obs_pc_shd)}\tGES={int(obs_ges_shd)}')  # \tCIs={obs_ci}
                            print(f'  - {dropout_mechanism}-0del:\tPC={int(obs_zerodel_pc_shd)}\tGES={int(obs_zerodel_ges_shd)}')   # \tCIs={obs_zerodel_ci}
                            print(f'  - {dropout_mechanism}-imputed:\tPC={int(imputed_pc_shd)}\tGES={int(imputed_ges_shd)}')    # \tCIs={imputed_ci}
                        except Exception as e: print('==== PRINT_OBS ERROR ====' , e)
