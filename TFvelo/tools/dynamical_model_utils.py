import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats.distributions import chi2, norm

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as pl
from matplotlib import rcParams

from .. import logging as logg
from ..core import clipped_log, invert, SplicingDynamics
from ..preprocessing.moments import get_connectivities
from .utils import make_dense, round

exp = np.exp


def get_norm_std(data1, data2, par=1):
    #norm_term = 1
    norm_term = np.sqrt(np.std(data1) * np.std(data2))
    #norm_term = data.max()-data.min()
    norm_term = norm_term/par #+ 1e-6
    return norm_term

def log(x, eps=1e-6):  # to avoid invalid values for log.
    warnings.warn(
        "`clipped_log` is deprecated since scVelo v0.2.4 and will be removed in a "
        "future version. Please use `clipped_log(x, eps=1e-6)` from `scvelo/core/`"
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return clipped_log(x, lb=0, ub=1, eps=eps)


def inv(x):
    warnings.warn(
        "`inv` is deprecated since scVelo v0.2.4 and will be removed in a future "
        "version. Please use `invert(x)` from `scvelo/core/` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return invert(x)


def normalize(X, axis=0, min_confidence=None):
    X_sum = np.sum(X, axis=axis)
    if min_confidence:
        X_sum += min_confidence
    X_sum += X_sum == 0
    return X / X_sum

def use_model(WX_model, in_data, method, bias=0):
    if method == 'constant':
        out_data = np.dot(in_data, WX_model)
    elif method == 'NN':
        import torch
        in_data = torch.Tensor(in_data)
        out_data = WX_model(in_data)
        out_data = out_data.detach().numpy()
    elif method == 'lsq_linear':
        out_data = np.dot(in_data, WX_model)
    elif method=='Gaussian_Kernel':    
        out_data = WX_model.predict(in_data)
    elif method in ['LS','LASSO', 'Ridge']: 
        out_data = WX_model.predict(in_data)
    elif method == 'LS_constrant':
        weight = WX_model.coef_.reshape(-1,1)
        out_data = np.ravel(np.dot(in_data, weight))
    return np.ravel(out_data) + bias


def convolve(x, weights=None):
    if weights is None:
        return x
    else:
        return weights.multiply(x).tocsr() if issparse(weights) else weights * x


def linreg(u, s):  # linear regression fit
    ss_ = s.multiply(s).sum(0) if issparse(s) else (s ** 2).sum(0)
    us_ = s.multiply(u).sum(0) if issparse(s) else (s * u).sum(0)
    return us_ / ss_


def compute_dt(t, clipped=True, axis=0):
    prepend = np.min(t, axis=axis)[None, :]
    dt = np.diff(np.sort(t, axis=axis), prepend=prepend, axis=axis)
    m_dt = np.max([np.mean(dt, axis=axis), np.max(t, axis=axis) / len(t)], axis=axis)
    m_dt = np.clip(m_dt, 0, None)
    if clipped:  # Poisson upper bound
        ub = m_dt + 3 * np.sqrt(m_dt)
        dt = np.clip(dt, 0, ub)
    return dt


def root_time(t, root=None):
    nans = np.isnan(np.sum(t, axis=0))
    if np.any(nans):
        t = t[:, ~nans]

    t_root = 0 if root is None else t[root]
    o = np.array(t >= t_root, dtype=int)
    t_after = (t - t_root) * o
    t_origin = np.max(t_after, axis=0)
    t_before = (t + t_origin) * (1 - o)

    t_switch = np.min(t_before, axis=0)
    t_rooted = t_after + t_before
    return t_rooted, t_switch


def compute_shared_time(t, perc=None, norm=True):
    nans = np.isnan(np.sum(t, axis=0))
    if np.any(nans):
        t = np.array(t[:, ~nans])
    t -= np.min(t)

    tx_list = np.percentile(t, [15, 20, 25, 30, 35] if perc is None else perc, axis=1)
    tx_max = np.max(tx_list, axis=1)
    tx_max += tx_max == 0
    tx_list /= tx_max[:, None]

    mse = []
    for tx in tx_list:
        tx_ = np.sort(tx)
        linx = np.linspace(0, 1, num=len(tx_))
        mse.append(np.sum((tx_ - linx) ** 2))
    idx_best = np.argsort(mse)[:2]

    t_shared = tx_list[idx_best].sum(0)
    if norm:
        t_shared /= t_shared.max()

    return t_shared


"""Dynamics delineation"""



def assign_t(
    x, y, alpha, beta, omega, theta, gamma, delta, WX_model, WX_method, filtered, 
    norm_std=False
):
    num = np.clip(len(x), 500, 1000)
    tpoints = np.linspace(0, 1, num=num)
    data_model = SplicingDynamics(alpha=alpha, beta=beta, omega=omega, theta=theta, 
        gamma=gamma).get_solution(tpoints)

    WX = use_model(WX_model, x, method=WX_method, bias=delta)
    if norm_std:
        std_WX = get_norm_std(data_model[:,0], WX[filtered]) # WX[filtered], data_model[:,0]
        std_y = get_norm_std(data_model[:,1], y[filtered], par=1.5) # y[filtered] data_model[:,1]
    else:
        std_WX, std_y = 1, 1
    data_model = data_model/np.array([std_WX, std_y])
    data_obs = np.vstack([WX/std_WX, np.ravel(y)/std_y]).T
    t = tpoints[
        ((data_model[None, :, :] - data_obs[:, None, :]) ** 2).sum(axis=2).argmin(axis=1)
    ]
    return t


def compute_divergence(
    x=None,
    y=None,
    alpha=None, 
    beta=None, 
    omega=None,
    theta=None,
    gamma=None,
    delta=None,
    WX_model=None,
    WX_method=None,
    WX=None,
    filtered=None,
    t=None,
    normalized=False,
    mode="distance",
    var_scale=False,  
    kernel_width=None,
    connectivities=None,
    reg_time=None,
    reg_par=None,
    min_confidence=None,
    pval_steady=None,
    steady_u=None,
    steady_s=None,
    noise_model="chi",
    time_connectivities=None,
    clusters=None,
    **kwargs,
):
    """Estimates the divergence of ODE to observations
    (avaiable metrics: distance, mse, likelihood, loglikelihood)

    Arguments
    ---------
    mode: `'distance'`, `'mse'`, `'likelihood'` (default: `'distance'`)

    """
    if mode in {"assign_timepoints", "time", "t"}:
        t = assign_t(
            x, y, alpha, beta, omega, theta, gamma, delta, WX_model, WX_method, filtered
        )
        return t


    # compute induction/repression state distances
    tpoints = np.linspace(0, 1, num=100)
    WX_t, y_t = SplicingDynamics(alpha=alpha, beta=beta, omega=omega, theta=theta, gamma=gamma, array_flag=True).get_solution(
        tpoints, stacked=False
    )
    if WX is None:
        WX = use_model(WX_model, x, method=WX_method, bias=delta)
    WX = np.array(WX, dtype=np.float32)
    WX_t = np.array(WX_t, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    y_t = np.array(y_t, dtype=np.float32)
    dist_WX = np.expand_dims(WX, axis=2) - np.expand_dims(WX_t, axis=0)
    dist_y = np.expand_dims(y, axis=2) - np.expand_dims(y_t, axis=0)

    #if mode == "unspliced_dists":
    #    return distu, distu_

    #elif mode == "outside_of_trajectory":
    #    return np.sign(distu) * np.sign(distu_) == 1

    dist_data = dist_WX ** 2 + dist_y ** 2
    res, var_data = dist_data, 1  # default vals;

    if connectivities is not None and connectivities is not False:
        res = (
            np.array([connectivities.dot(r) for r in res])
            if res.ndim > 2
            else connectivities.dot(res.T).T
        )

    # compute variances
    if noise_model == "chi":
        if var_scale:
            var_data = np.mean(dist_data, axis=0) - np.mean(np.sqrt(dist_data), axis=0) ** 2
            if kernel_width is not None:
                var_data *= kernel_width ** 2
            res /= var_data
        elif kernel_width is not None:
            res /= kernel_width ** 2

    if reg_time is not None and len(reg_time) == len(dist_y):
        reg_time /= np.max(reg_time)

        dist_t = (t - reg_time[:, None]) ** 2
        mu_res = np.mean(res, axis=1)
        if reg_par is not None:
            mu_res *= reg_par

        res += dist_t * mu_res


    if mode in {"assign_timepoints", "time", "t"}:
        res = t

    elif mode == "likelihood":
        res = 1 / (2 * np.pi * np.sqrt(var_data)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)

    elif mode == "nll":
        res = np.log(2 * np.pi * np.sqrt(var_data)) + 0.5 * res
        if normalized:
            res = normalize(res, min_confidence=min_confidence)

    elif mode == "confidence":
        res = np.array([res[0], res[1]])
        res = 1 / (2 * np.pi * np.sqrt(var_data)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)
        res = np.median(
            np.max(res, axis=0) - (np.sum(res, axis=0) - np.max(res, axis=0)), axis=1
        )

    elif mode == "soft_state":
        res = 1 / (2 * np.pi * np.sqrt(var_data)) * np.exp(-0.5 * res)
        if normalized:
            res = normalize(res, min_confidence=min_confidence)
        res = res[1] - res[0]

    elif mode == "hard_state":
        res = np.argmin(res, axis=0)

    return res


def assign_timepoints(**kwargs):
    return compute_divergence(**kwargs, mode="assign_timepoints")




"""Base Class for Dynamics Recovery"""


class BaseDynamics:
    def __init__(
        self,
        adata,
        gene,
        WX_method,
        max_n_TF=999,
        x=None,
        y=None,
        use_raw=False,
        perc=99,
        max_iter=10,
        fit_time=True,
        fit_scaling=True,
        fit_steady_states=True,
        fit_connected_states=True,
        fit_basal_transcription=None,
        high_pars_resolution=False,
        init_vals=None,
        reg=None,
        WX_thres=20,
        init_weight_method=None,
        fix_theta=0,
        gamma_thres=0,
        n_time_points=1000,
    ):
        self.reg = reg
        self.init_weight_method = init_weight_method
        self.WX_thres = WX_thres
        self.fix_theta = fix_theta
        self.gamma_thres = gamma_thres
        self.max_n_TF = max_n_TF
        self.x, self.target_y, self.use_raw = None, None, None
        self.n_time_points = n_time_points

        _layers = adata[:, gene].layers
        self.gene = gene
        self.use_raw = use_raw or ("M_total" not in _layers.keys())
        layer2use = "total" if self.use_raw else "M_total"
        #print('layer2use:', layer2use)
        gene_id = list(adata.var_names).index(gene)
        self.n_TFs = adata.var['n_TFs'][gene]

        self.n_TFs = min(self.n_TFs, self.max_n_TF)
        if self.n_TFs < 1:
            print(gene, "not recoverable due to n_TFs =", self.n_TFs)
            self.recoverable = False
            return

        self.TFs_correlation = adata.varm['TFs_correlation'][gene_id, :self.n_TFs]
        self.knockTF_Log2FC = adata.varm['knockTF_Log2FC'][gene_id, :self.n_TFs]
        self.only_positive = self.knockTF_Log2FC < 0
        self.only_negative = self.knockTF_Log2FC > 0

        self.TFs_x = adata.layers[layer2use][:, adata.varm['TFs_id'][gene_id][:self.n_TFs]]  
        self.target_y = _layers[layer2use] 

        # Basal transcription
        self.alpha, self.beta, self.omega, self.theta, self.gamma, self.delta = None, None, None, None, None, None
        self.scaling = None
        self.filtered, self.filtered_upper = None, None
        self.t = None
        self.likelihood, self.loss, self.pars = None, None, None

        self.WX_method = WX_method

        self.max_iter = max_iter
        #self.max_iter = np.clip(10+self.n_TFs*2, 10, 50)

        # partition to total of 5 fitting procedures
        # (t_ and alpha, scaling, rates, t_, all together)
        self.simplex_kwargs = {
            "method": "Nelder-Mead",
            "options": {"maxiter": int(self.max_iter)},
        }
        
        self.perc = perc
        self.recoverable = True
        try:
            self.initialize_filter()
        except Exception:
            self.recoverable = False
            logg.warn(f"Model for {self.gene} could not be instantiated.")

        self.refit_time = fit_time

        self.assignment_mode = None

        self.fit_scaling = fit_scaling
        self.fit_steady_states = fit_steady_states
        self.fit_connected_states = fit_connected_states
        self.connectivities = (
            get_connectivities(adata)
            if self.fit_connected_states is True
            else self.fit_connected_states
        )
        self.high_pars_resolution = high_pars_resolution
        self.init_vals = init_vals



    def initialize_filter(self, to_filter=True, to_filter_upper=True): ########## to_filter_upper=True
        nonzero_x = (self.TFs_x > 0).sum(1)>0
        nonzero_y = np.ravel(self.target_y > 0)

        filter_id = np.array(nonzero_x & nonzero_y, dtype=bool)
        self.recoverable = np.sum(filter_id) > 2

        if self.recoverable:
            if to_filter:
                ub_y = np.percentile(self.target_y[filter_id], self.perc)
                if ub_y > 0:
                    filter_id &= np.ravel(self.target_y <= ub_y)

            self.filtered = filter_id

            if to_filter_upper:
                self.filtered_upper = np.array(filter_id)
                if np.any(filter_id):
                    f_upper = (self.target_y > np.max(self.target_y[filter_id]) / 10) 
                    self.filtered_upper &= np.ravel(f_upper)
                self.filtered = self.filtered_upper
            
            TFs_x = self.TFs_x[self.filtered]
            target_y = self.target_y[self.filtered]
            self.std_TFs_x = np.std(TFs_x, 0)
            self.std_target_y = np.std(target_y)

            self.recoverable = np.sum(self.filtered) > 5*self.n_TFs

    def load_pars(self, adata, gene):
        idx = adata.var_names.get_loc(gene) if isinstance(gene, str) else gene
        self.alpha = adata.var["fit_alpha"][idx]
        self.beta = adata.var["fit_beta"][idx] 
        self.omega = adata.var["fit_omega"][idx]
        self.theta = adata.var["fit_theta"][idx]
        self.gamma = adata.var["fit_gamma"][idx]
        self.delta = adata.var["fit_delta"][idx]
        self.scaling = adata.var["fit_scaling"][idx]

        self.pars = [self.alpha, self.beta, self.gamma, self.delta, self.weight, self.t_, self.scaling]
        self.pars = np.array(self.pars)[:, None]

        lt = "latent_time"
        t = adata.obs[lt] if lt in adata.obs.keys() else adata.layers["fit_t"][:, idx]

        if isinstance(self.refit_time, bool):
            self.t, self.omega, self.o = self.get_time_assignment(t=t)
        else:
            tkey = self.refit_time
            self.t = adata.obs[tkey].values if isinstance(tkey, str) else tkey
            self.refit_time = False
            steady_states = t == self.t_
            if np.any(steady_states):
                self.t_ = np.mean(self.t[steady_states])
            self.t, self.omega, self.o = self.get_time_assignment(t=self.t)

        self.loss = [self.get_loss()]

    def get_filter(self, to_filter=None):
        filtered = (
            np.array(self.filtered
            )
            if to_filter
            else np.ones(len(self.x), bool) 
        )
        return filtered

    def get_reads(self, to_filter=None):
        x, y = self.x, self.y
        if to_filter:
            filtered = self.get_filter(to_filter=to_filter)
            x, y = x[filtered], y[filtered]
        return x, y

    def get_vars(
        self,
        alpha=None, 
        beta=None, 
        omega=None,
        theta=None,
        gamma=None,
        delta=None
    ):
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        omega = self.omega if omega is None else omega
        theta = self.theta if theta is None else theta        
        gamma = self.gamma if gamma is None else gamma
        delta = self.delta if delta is None else delta
        return [alpha, beta, omega, theta, gamma, delta]

    def get_divergence(
        self,
        alpha=None, 
        beta=None, 
        omega=None,
        theta=None,
        gamma=None,
        delta=None,
        mode=None,
        **kwargs,
    ):
        alpha, beta, omega, theta, gamma, delta = self.get_vars(
            alpha, beta, omega, theta, gamma, delta
        )
        x, y = self.x, self.y
        kwargs.update(
            dict(
                mode=mode,
                assignment_mode=self.assignment_mode,
                connectivities=self.connectivities,
            )
        )
        res = compute_divergence(x, y, alpha, beta, omega, theta, gamma, delta, self.filtered,**kwargs)
        return res

    def update_WX(self, alpha, beta, omega, theta, gamma, method='NN', thres=20): 
        t = self.get_time_assignment(alpha, beta, omega, theta, gamma, refit_time=False, to_filter=True)
        WX_t, y_t = SplicingDynamics(alpha=alpha, beta=beta, omega=omega, theta=theta, gamma=gamma).get_solution(
            t, stacked=False)
        
        if method == 'constant':
            dim = self.n_TFs
            #tmp = gamma - np.sqrt(gamma*gamma+8*beta)
            #max_WX = (gamma-tmp/2)*alpha*np.exp(-tmp*tmp/(16*beta)) + gamma*theta
            #max_x = self.x_f.sum(1).max()
            #norm_term = max_WX/max_x
            #model = norm_term
            #model = (self.y_f.mean()/self.x_f.mean(0))/dim 
            #model = np.ones(dim)/dim
            model = np.ones(dim)/self.x_f.sum(1).std()
            model = model.reshape(-1,1)

        elif method in ['LS', 'LASSO', 'Ridge', 'LS_constrant']:
            fit_intercept = False
            from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV, LinearRegression, Ridge 
            if method =='LS':
                model = LinearRegression(fit_intercept=fit_intercept)
            elif method =='LS_constrant':
                model = LinearRegression(fit_intercept=False)
            elif method == 'LASSO':
                model = Lasso(alpha=0.01, max_iter=10000, fit_intercept=fit_intercept)  
            elif method == 'Ridge':
                model = Ridge(alpha=0.01, max_iter=10000, fit_intercept=fit_intercept)

            model.fit(self.x_f, WX_t.reshape(-1,1))
            #model.score(self.x_f, WX_t)
            #weight = model.coef_.reshape(-1,1)
            #if fit_intercept:
            #    intercept_ = model.intercept_
            #else:
            #    intercept_ = 0
            #WX_f = np.ravel(model.predict(self.x_f))
            #WX_f_2 = np.ravel(np.dot(self.x_f, new_weight)) + new_intercept_
            if method == 'LS_constrant':
                weight = model.coef_.reshape(-1,1)
                X = self.x_f
                R = np.ones([1, self.n_TFs])
                tmp = np.linalg.pinv(np.dot(X.T, X))
                tmp1 = np.dot(tmp, R.T)
                tmp2_1 = np.dot(R, tmp)
                tmp2 = 1/np.dot(tmp2_1, R.T)
                tmp3 = np.dot(R, weight) - 1
                weight_star = weight - tmp1*tmp2*tmp3
                model.coef_ = weight_star.reshape(1,-1)

        elif method == 'lsq_linear':
            from scipy.optimize import lsq_linear
            # [X,1]
            added_col = np.ones([self.x_f.shape[0], 1])
            input_x = np.concatenate([self.x_f, added_col], axis=1)
            if self.init_weight_method == 'knockTF_Log2FC': 
                lb = np.append(-(1-self.only_positive) * thres, -thres)
                hb = np.append((1-self.only_negative) * thres, thres)
                res = lsq_linear(input_x, WX_t, bounds=(lb, hb), lsmr_tol='auto', verbose=0)
            else:
                res = lsq_linear(input_x, WX_t, bounds=(-thres, thres), lsmr_tol='auto', verbose=0)
            model = res.x.reshape(-1, 1)
            model = [model[:-1], model[-1][0]]
            #print(model)
            #print(res.cost)
            #WX_t_recon = np.ravel(np.dot(self.x_f, model))
            #cost = ((WX_t_recon-WX_t)*(WX_t_recon-WX_t)).sum()/2


        return model

    def get_time_assignment(
        self,
        alpha=None, 
        beta=None, 
        omega=None,
        theta=None,
        gamma=None,
        delta=None,
        refit_time=None,
        rescale_factor=None,
        to_filter=None,
        weights_cluster=None,
    ):

        alpha, beta, omega, theta, gamma, delta = self.get_vars(alpha, beta, omega, theta, gamma, delta)

        if refit_time is None:
            refit_time = self.refit_time

        if refit_time:
            t = assign_t(
                self.x, self.y, alpha, beta, omega, theta, gamma, delta, self.WX_model, self.WX_method,
                self.filtered, norm_std=True ###########
            )

        else:
            t = self.t

        if to_filter:
            filtered = self.get_filter(to_filter=to_filter)
            t = t[filtered]
        return t


    def get_dists(
        self,
        t=None,
        alpha=None, 
        beta=None, 
        omega=None,
        theta=None,
        gamma=None,
        delta=None,
        WX_model=None,
        refit_time=False,
        to_filter=True,
        reg=None,
        norm_std=True
    ):
        if WX_model is None:
            WX_model = self.WX_model

        filter_args = dict(to_filter=to_filter)
        x, y = self.get_reads(**filter_args)

        alpha, beta, omega, theta, gamma, delta = self.get_vars( 
            alpha, beta, omega, theta, gamma, delta
        )

        t = self.get_time_assignment(
            alpha, beta, omega, theta, gamma, delta, refit_time, **filter_args 
        )

        WX_t, y_t = SplicingDynamics(
            alpha=alpha, beta=beta, omega=omega, theta=theta, gamma=gamma
        ).get_solution(t, stacked=False)

        WX = use_model(WX_model, x, method=self.WX_method, bias=delta)
        if norm_std:
            std_WX = get_norm_std(WX_t, WX) # WX, WX_t
            std_y = get_norm_std(y_t, y, par=1.5) # y, y_t
        else:
            std_WX, std_y = 1, 1
        WX_diff = np.array(WX_t - WX) / std_WX
        y_diff = np.array(y_t - np.ravel(y)) / std_y
        if reg is None:
            reg = 0
        return WX_diff, y_diff, reg

    def get_residuals_linear(self, **kwargs):
        udiff, sdiff, reg = self.get_dists(**kwargs)
        return udiff, sdiff

    def get_residuals(self, **kwargs):
        udiff, sdiff, reg = self.get_dists(**kwargs)
        return np.sign(sdiff) * np.sqrt(udiff ** 2 + sdiff ** 2)

    def get_dist(self, noise_model="normal", regularize=True, **kwargs):
        WX_diff, y_diff, reg = self.get_dists(**kwargs)
        if noise_model == "normal":
            dist_data = WX_diff ** 2 + y_diff ** 2
        elif noise_model == "laplace":
            dist_data = np.abs(WX_diff) + np.abs(y_diff)
        if regularize:
            dist_data += reg ** 2
        return dist_data

    def get_se(self, **kwargs):
        return np.sum(self.get_dist(**kwargs))

    def get_mse(self, **kwargs):
        return np.mean(self.get_dist(**kwargs))

    def get_loss(
        self,
        t=None,
        alpha=None, 
        beta=None, 
        omega=None,
        theta=None,
        gamma=None,
        delta=None,
        WX_model=None,
        refit_time=None,
        norm_std=False,
        reg=None,
    ):
        kwargs = dict(t=t, alpha=alpha, beta=beta, omega=omega, theta=theta, gamma=gamma, delta=delta, WX_model=WX_model, reg=reg)
        kwargs.update(dict(refit_time=refit_time, norm_std=norm_std))
        #######return self.get_se(**kwargs)
        return self.get_mse(**kwargs)

    def get_loglikelihood(self, varx=None, noise_model="normal", **kwargs): ##########
        if "to_filter" not in kwargs:
            kwargs.update({"to_filter": True})
        WX_diff, y_diff, reg = self.get_dists(**kwargs)
        dist_data = WX_diff ** 2 + y_diff ** 2 + reg ** 2
        eucl_dist = np.sqrt(dist_data)
        #WX_f = use_model(self.WX_model, self.x_f, method=self.WX_method, bias=self.delta)
        #n = np.clip(len(dist_data) - len(WX_f) * 0.01, 2, None)
        n = len(dist_data)

        # compute variance / equivalent to np.var(np.sign(sdiff) * np.sqrt(distx))
        if varx is None:
            varx = self.get_variance(**kwargs)
            #varx = np.mean(dist_data) - np.mean(np.sign(y_diff) * eucl_dist) ** 2
        varx += varx == 0  # edge case of mRNAs levels to be the same across all cells

        if noise_model == "normal":
            loglik = -1 / 2 / n * np.sum(dist_data) / varx
            loglik -= 1 / 2 * np.log(2 * np.pi * varx)
        elif noise_model == "laplace":
            loglik = -1 / np.sqrt(2) / n * np.sum(eucl_dist) / np.sqrt(varx)
            loglik -= 1 / 2 * np.log(2 * varx)
        else:
            raise ValueError("That noise model is not supported.")
        return loglik

    def get_likelihood(self, **kwargs):
        if "to_filter" not in kwargs:
            kwargs.update({"to_filter": True})
        likelihood = np.exp(self.get_loglikelihood(**kwargs))
        return likelihood


    def get_variance(self, **kwargs):
        if "to_filter" not in kwargs:
            kwargs.update({"to_filter": True})
        WX_diff, y_diff, reg = self.get_dists(**kwargs)
        dist_data = WX_diff ** 2 + y_diff ** 2 
        #return np.mean(dist_data) - np.mean(np.sign(y_diff) * np.sqrt(dist_data)) ** 2
        return dist_data.sum()/(len(dist_data)-1)


    def plot_profile_contour(
        self,
        xkey="gamma",
        ykey="alpha",
        x_sight=0.5,
        y_sight=0.5,
        num=20,
        contour_levels=4,
        fontsize=12,
        refit_time=None,
        ax=None,
        color_map="RdGy",
        figsize=None,
        dpi=None,
        vmin=None,
        vmax=None,
        horizontal_ylabels=True,
        show_path=False,
        show=True,
        return_color_scale=False,
        **kwargs,
    ):
        from scvelo.plotting.utils import update_axes

        x_var = getattr(self, xkey)
        y_var = getattr(self, ykey)

        x = np.linspace(-x_sight, x_sight, num=num) * x_var + x_var
        y = np.linspace(-y_sight, y_sight, num=num) * y_var + y_var

        assignment_mode = self.assignment_mode
        self.assignment_mode = None

        # TODO: Check if list comprehension can be used
        zp = np.zeros((len(x), len(x)))
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                zp[i, j] = self.get_likelihood(
                    **{xkey: xi, ykey: yi}, refit_time=refit_time
                )
        log_zp = np.log1p(zp.T)

        if vmin is None:
            vmin = np.min(log_zp)
        if vmax is None:
            vmax = np.max(log_zp)

        x_label = r"$" + f"\\{xkey}$" if xkey in ["gamma", "alpha", "beta", "weight"] else xkey
        y_label = r"$" + f"\\{ykey}$" if ykey in ["gamma", "alpha", "beta", "weight"] else ykey

        if ax is None:
            figsize = rcParams["figure.figsize"] if figsize is None else figsize
            ax = pl.figure(figsize=(figsize[0], figsize[1]), dpi=dpi).gca()

        ax.contourf(x, y, log_zp, levels=num, cmap=color_map, vmin=vmin, vmax=vmax)
        if contour_levels != 0:
            contours = ax.contour(
                x, y, log_zp, levels=contour_levels, colors="k", linewidths=0.5
            )
            fmt = "%1.1f" if np.isscalar(contour_levels) else "%1.0f"
            ax.clabel(contours, fmt=fmt, inline=True, fontsize=fontsize * 0.75)

        ax.scatter(x=x_var, y=y_var, s=50, c="purple", zorder=3, **kwargs)
        ax.set_xlabel(x_label, fontsize=fontsize)
        rotation = 0 if horizontal_ylabels else 90
        ax.set_ylabel(y_label, fontsize=fontsize, rotation=rotation)
        update_axes(ax, fontsize=fontsize, frameon=True)

        if show_path:
            axis = ax.axis()
            x_hist = self.pars[["alpha", "beta", "gamma", "weight",  "t_", "scaling"].index(xkey)]
            y_hist = self.pars[["alpha", "beta", "gamma", "weight", "t_", "scaling"].index(ykey)]
            ax.plot(x_hist, y_hist)
            ax.axis(axis)

        self.assignment_mode = assignment_mode
        if return_color_scale:
            return np.min(log_zp), np.max(log_zp)
        elif not show:
            return ax

    def plot_profile_hist(
        self,
        xkey="gamma",
        sight=0.5,
        num=20,
        dpi=None,
        fontsize=12,
        ax=None,
        figsize=None,
        color_map="RdGy",
        vmin=None,
        vmax=None,
        show=True,
    ):
        from scvelo.plotting.utils import update_axes

        x_var = getattr(self, xkey)
        x = np.linspace(-sight, sight, num=num) * x_var + x_var

        assignment_mode = self.assignment_mode
        self.assignment_mode = None

        # TODO: Check if list comprehension can be used
        zp = np.zeros((len(x)))
        for i, xi in enumerate(x):
            zp[i] = self.get_likelihood(**{xkey: xi}, refit_time=True)

        log_zp = np.log1p(zp.T)
        if vmin is None:
            vmin = np.min(log_zp)
        if vmax is None:
            vmax = np.max(log_zp)

        x_label = r"$" + f"\\{xkey}$" if xkey in ["gamma", "alpha", "beta", "weight"] else xkey
        figsize = rcParams["figure.figsize"] if figsize is None else figsize
        if ax is None:
            fig = pl.figure(figsize=(figsize[0], figsize[1]), dpi=dpi)
            ax = fig.gca()

        xp = np.linspace(x.min(), x.max(), 1000)
        yp = np.interp(xp, x, log_zp)
        ax.scatter(xp, yp, c=yp, cmap=color_map, edgecolor="none", vmin=vmin, vmax=vmax)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel("likelihood", fontsize=fontsize)
        update_axes(ax, fontsize=fontsize, frameon=True)

        self.assignment_mode = assignment_mode
        if not show:
            return ax

    def plot_profiles(
        self,
        params=None,
        contour_levels=0,
        sight=0.5,
        num=20,
        fontsize=12,
        color_map="RdGy",
        vmin=None,
        vmax=None,
        figsize=None,
        dpi=None,
        **kwargs,
    ):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        if params is None:
            params = ["alpha", "beta", "gamma", "weight"]
        fig = pl.figure(constrained_layout=True, dpi=dpi, figsize=figsize)
        n = len(params)
        gs = gridspec.GridSpec(n, n, figure=fig)

        pkwargs = dict(color_map=color_map, vmin=vmin, vmax=vmax, figsize=figsize)

        for i in range(len(params)):
            for j in range(n - 1, i - 1, -1):
                xkey = params[j]
                ykey = params[i]
                ax = fig.add_subplot(gs[n - 1 - i, n - 1 - j])
                if xkey == ykey:
                    ax = self.plot_profile_hist(
                        xkey,
                        ax=ax,
                        num=num,
                        sight=sight if np.isscalar(sight) else sight[j],
                        fontsize=fontsize,
                        show=False,
                        **pkwargs,
                    )
                    if i == 0 & j == 0:
                        cax = inset_axes(
                            ax,
                            width="7%",
                            height="100%",
                            loc="lower left",
                            bbox_to_anchor=(1.05, 0.0, 1, 1),
                            bbox_transform=ax.transAxes,
                            borderpad=0,
                        )
                        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                        cmap = mpl.cm.get_cmap(color_map)
                        _ = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
                else:
                    vmin_, vmax_ = self.plot_profile_contour(
                        xkey,
                        ykey,
                        ax=ax,
                        contour_levels=contour_levels,
                        x_sight=sight if np.isscalar(sight) else sight[j],
                        y_sight=sight if np.isscalar(sight) else sight[i],
                        num=num,
                        fontsize=fontsize,
                        return_color_scale=True,
                        **pkwargs,
                        **kwargs,
                    )
                    if vmin is None or vmax is None:
                        vmin, vmax = vmin_, vmax_  # scaled to first contour plot

                if i != 0:
                    ax.set_xlabel("")
                    ax.set_xticks([])
                if j - n + 1 != 0:
                    ax.set_ylabel("")
                    ax.set_yticks([])

    def plot_state_likelihoods(
        self,
        num=300,
        dpi=None,
        figsize=None,
        color_map=None,
        color_map_steady=None,
        continuous=True,
        common_color_scale=True,
        var_scale=True,
        kernel_width=None,
        normalized=None,
        transitions=None,
        colorbar=False,
        alpha_=0.5,
        linewidths=3,
        padding_u=0.1,
        padding_s=0.1,
        fontsize=12,
        title=None,
        ax=None,
        **kwargs,
    ):
        from scvelo.plotting.utils import rgb_custom_colormap, update_axes

        if color_map is None:
            color_map = rgb_custom_colormap(
                ["royalblue", "white", "seagreen"], alpha=[1, 0.5, 1]
            )
        if color_map_steady is None:
            color_map_steady = rgb_custom_colormap(
                colors=3 * ["sienna"], alpha=[0, 0.5, 1]
            )

        alpha, beta, gamma, weight, scaling, t_ = self.get_vars()
        u, s = self.u / scaling, self.s
        padding_u *= np.max(u) - np.min(u)
        padding_s *= np.max(s) - np.min(s)
        uu = np.linspace(np.min(u) - padding_u, np.max(u) + padding_u, num=num)
        ss = np.linspace(np.min(s) - padding_s, np.max(s) + padding_s, num=num)

        grid_u, grid_s = np.meshgrid(uu, ss)
        grid_u = grid_u.flatten()
        grid_s = grid_s.flatten()

        if var_scale:
            var_scale = self.get_variance()

        dkwargs = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "weight": weight,
            "scaling": scaling,
            "t_": t_,
            "kernel_width": kernel_width,
            "std_u": self.std_u,
            "std_s": self.std_s,
            "var_scale": var_scale,
            "normalized": normalized,
            "fit_steady_states": True,
            "assignment_mode": "projection",
        }

        likelihoods = compute_divergence(u, s, mode="soft_state", **dkwargs)
        likelihoods_steady = compute_divergence(u, s, mode="steady_state", **dkwargs)

        likelihoods_grid = compute_divergence(
            grid_u, grid_s, mode="soft_state", **dkwargs
        )
        likelihoods_grid_steady = compute_divergence(
            grid_u, grid_s, mode="steady_state", **dkwargs
        )

        figsize = rcParams["figure.figsize"] if figsize is None else figsize
        if ax is None:
            fig = pl.figure(figsize=(figsize[0], figsize[1]), dpi=dpi)
            ax = fig.gca()

        ax.scatter(
            x=s,
            y=u,
            s=50,
            c=likelihoods_steady,
            zorder=3,
            cmap=color_map_steady,
            edgecolors="black",
            **kwargs,
        )
        ax.scatter(
            x=s,
            y=u,
            s=50,
            c=likelihoods,
            zorder=3,
            cmap=color_map,
            edgecolors="black",
            **kwargs,
        )

        l_grid, l_grid_steady = (
            likelihoods_grid.reshape(num, num).T,
            likelihoods_grid_steady.reshape(num, num).T,
        )

        if common_color_scale:
            vmax = vmax_steady = np.max(
                [np.abs(likelihoods_grid), np.abs(likelihoods_grid_steady)]
            )
        else:
            vmax, vmax_steady = np.max(np.abs(likelihoods_grid)), None

        if continuous:
            extent = (min(ss), max(ss), min(uu), max(uu))
            contf_steady = ax.imshow(
                l_grid_steady,
                cmap=color_map_steady,
                alpha=alpha_,
                vmin=0,
                vmax=vmax_steady,
                aspect="auto",
                origin="lower",
                extent=extent,
            )
            contf = ax.imshow(
                l_grid,
                cmap=color_map,
                alpha=alpha_,
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
                origin="lower",
                extent=extent,
            )
        else:
            cmap = color_map_steady
            contf_steady = ax.contourf(
                ss, uu, l_grid_steady, vmin=0, vmax=vmax_steady, levels=30, cmap=cmap
            )
            contf = ax.contourf(
                ss, uu, l_grid, vmin=-vmax, vmax=vmax, levels=30, cmap=color_map
            )

        # Contour lines
        if transitions is not None:
            transitions = np.multiply(
                np.array(transitions),
                [np.min(likelihoods_grid), np.max(likelihoods_grid)],
            )  # trans_width
            ax.contour(
                ss,
                uu,
                likelihoods_grid.reshape(num, num).T,
                transitions,
                linestyles="solid",
                colors="k",
                linewidths=linewidths,
            )

        if colorbar:
            pl.colorbar(contf, ax=ax)
            pl.colorbar(contf_steady, ax=ax)
        ax.set_xlabel("spliced", fontsize=fontsize)
        ax.set_ylabel("unspliced", fontsize=fontsize)
        title = "" if title is None else title
        ax.set_title(title, fontsize=fontsize)
        update_axes(ax, fontsize=fontsize, frameon=True)

        return ax

    # for differential kinetic test
    def initialize_diff_kinetics(self, clusters):
        # after fitting dyn. model
        if self.varx is None:
            self.varx = self.get_variance()
        self.initialize_weights(weighted=False)
        self.steady_state_ratio = None
        self.clusters = clusters
        self.cats = pd.Categorical(clusters).categories
        self.weights_outer = np.array(self.weights) & self.get_divergence(
            mode="outside_of_trajectory"
        )

    def get_orth_fit(self, **kwargs):
        kwargs["weighted"] = True  # include inner vals for orthogonal regression
        u, s = self.get_reads(**kwargs)
        a, b = np.sum(s * u), np.sum(u ** 2 - s ** 2)
        orth_beta = (b + ((b ** 2 + 4 * a ** 2) ** 0.5)) / (2 * a)
        return orth_beta

    def get_orth_distx(self, orth_beta=None, **kwargs):
        if "weighted" not in kwargs:
            kwargs["weighted"] = "outer"
        u, s = self.get_reads(**kwargs)
        if orth_beta is None:
            orth_beta = self.get_orth_fit(**kwargs)
        s_real = np.array((s + (orth_beta * u)) / (1 + orth_beta ** 2))
        sdiff = np.array(s_real - s) / self.std_s
        udiff = np.array(orth_beta * s_real - u) / self.std_u * self.scaling
        return udiff ** 2 + sdiff ** 2

    def get_pval(self, model="dynamical", **kwargs):
        # assuming var-scaled udiff, sdiff follow N(0,1),
        # the sum of errors for the cluster follows chi2(df=2n)
        if "weighted" not in kwargs:
            kwargs["weighted"] = "outer"
        distx = (
            self.get_orth_distx(**kwargs)
            if model == "orthogonal"
            else self.get_distx(**kwargs) / 2
        )
        return chi2.sf(df=2 * len(distx), x=np.sum(distx) / self.varx)

    def get_pval_diff_kinetics(self, orth_beta=None, min_cells=10, **kwargs):
        """
        Calculates the p-value for the likelihood ratio
        using the asymptotic property of the chi^2 distr.

        Derivation:
        - dists_dynamical and dists_orthogonal are squared N(0,1) distributed residuals
        - X1 = sum(dists_dynamical) / variance ~ chi2(df=2n)
        - X2 = sum(dists_orthogonal) / variance ~ chi2(df=2n)
        - Y1 = (X1 - df) / sqrt(2*df) ~ N(0,1) for large df
        - Y2 = (X2 - df) / sqrt(2*df) ~ N(0,1) for large df
        - since Y1~N(0,1) and Y2~N(0,1), Y1 - Y2 ~ N(0,2) or (Y1 -Y2) / sqrt(2) ~ N(0,1)
        - thus Z = (X1 - X2) / sqrt(4*df) ~ N(0,1) for large df

        Parameters
        ----------
        indices: "bool" array
            bool array for cluster of interest
        orth_beta: "float"
            orthogonal line fit beta

        Returns
        -------
        p-value
        """

        if (
            "weights_cluster" in kwargs
            and np.sum(kwargs["weights_cluster"]) < min_cells
        ):
            return 1
        if "weighted" not in kwargs:
            kwargs["weighted"] = "outer"
        distx = self.get_distx(**kwargs) / 2  # due to convolved assignments (tbd)
        orth_distx = self.get_orth_distx(orth_beta=orth_beta, **kwargs)
        denom = self.varx * np.sqrt(4 * 2 * len(distx))
        pval = norm.sf(
            (np.sum(distx) - np.sum(orth_distx)) / denom
        )  # see derivation above
        return pval

    def get_cluster_mse(self, clusters=None, min_cells=10, weighted="outer"):
        if self.clusters is None or clusters is not None:
            self.initialize_diff_kinetics(clusters)
        mse = np.array(
            [
                self.get_mse(weights_cluster=self.clusters == c, weighted=weighted)
                for c in self.cats
            ]
        )
        if min_cells is not None:
            w = (
                self.weights_outer
                if weighted == "outer"
                else self.weights_upper
                if weighted == "upper"
                else self.weights
            )
            mse[
                np.array([np.sum(w & (self.clusters == c)) for c in self.cats])
                < min_cells
            ] = 0
        return mse

    def get_cluster_pvals(self, clusters=None, model=None, orth_beta=None, **kwargs):
        if self.clusters is None or clusters is not None:
            self.initialize_diff_kinetics(clusters)
        pvals = np.array(
            [
                self.get_pval_diff_kinetics(
                    weights_cluster=self.clusters == c, orth_beta=orth_beta, **kwargs
                )
                if model is None
                else self.get_pval(
                    model=model, weights_cluster=self.clusters == c, **kwargs
                )
                for c in self.cats
            ]
        )
        return pvals



def get_reads(adata, key="fit", scaled=True, use_raw=False):
    WX = adata.layers["WX"]
    y = adata.layers["y"]
    return WX, y


def get_vars(adata, scaled=True, key="fit"):
    alpha = (
        adata.var[f"{key}_alpha"].values if f"{key}_alpha" in adata.var.keys() else 1
    )
    beta = adata.var[f"{key}_beta"].values if f"{key}_beta" in adata.var.keys() else 1
    omega = adata.var[f"{key}_omega"].values
    theta = adata.var[f"{key}_theta"].values
    gamma = adata.var[f"{key}_gamma"].values
    delta = adata.var[f"{key}_delta"].values
    return alpha, beta, omega, theta, gamma, delta



def get_divergence(
    adata, mode="soft", use_pseudo_time=None, use_connectivities=None, **kwargs
):
    vdata = adata[:, ~np.isnan(adata.var["fit_alpha"].values)].copy()
    alpha, beta, omega, theta, gamma, delta = get_vars(vdata)

    kwargs_ = {
        "kernel_width": None,
        "normalized": True,
        "var_scale": True,
        "reg_par": None,
        "min_confidence": 1e-2,
        "fit_steady_states": True,
        "fit_basal_transcription": None,
    }
    kwargs_.update(adata.uns["recover_dynamics"])
    kwargs_.update(**kwargs)

    reg_time = None
    if use_pseudo_time is True:
        use_pseudo_time = "velocity_pseudotime"
    if isinstance(use_pseudo_time, str) and use_pseudo_time in adata.obs.keys():
        reg_time = adata.obs[use_pseudo_time].values
    WX, y = get_reads(vdata, use_raw=kwargs_["use_raw"])

    kwargs_.update(dict(reg_time=reg_time, mode=mode))
    conn = get_connectivities(adata) if use_connectivities else None
    res = compute_divergence(
        y=y, alpha=alpha, beta=beta, omega=omega, theta=theta, gamma=gamma, 
        delta=delta, WX=WX, connectivities=conn, **kwargs_
    )
    return res
