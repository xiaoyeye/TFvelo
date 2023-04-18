import os
#from this import d

import numpy as np
import pandas as pd
from scipy.optimize import minimize, basinhopping

import matplotlib.pyplot as pl
from matplotlib import rcParams

from .. import logging as logg
from .. import settings
from ..core import get_n_jobs, parallelize, SplicingDynamics
from ..preprocessing.moments import get_connectivities
from .dynamical_model_utils import BaseDynamics, convolve, linreg, get_norm_std, use_model
from .utils import make_unique_list, test_bimodality

MAX_LOSS = 1e6

class DynamicsRecovery(BaseDynamics):
    def __init__(self, adata, gene, load_pars=None, **kwargs):
        super().__init__(adata, gene, **kwargs)
        if load_pars and "fit_alpha" in adata.var.keys():
            try:
                self.load_pars(adata, gene)
            except Exception:
                self.recoverable = False
                logg.warn(f"Model for {self.gene} could not be instantiated in load_pars.")

        elif self.recoverable:
            #try: 
                self.initialize()
            #except Exception:
            #    self.recoverable = False
            #    logg.warn(f"Model for {self.gene} could not be instantiated in recovering.")

    def initialize_WX(self, method):
        dim = self.n_TFs
        #tmp = self.gamma - np.sqrt(self.gamma*self.gamma+8*self.beta)
        #max_WX = (self.gamma-tmp/2)*self.alpha*np.exp(-tmp*tmp/(16*self.beta)) + self.gamma*self.rho
        #max_x = self.x_f.sum(1).max()
        #norm_term = max_WX/max_x
        if method == 'rand':
            weight = np.random.rand(dim)*2/dim  #* (2*norm_term)
        elif method == 'ones':
            #weight = np.ones(dim) * norm_term
            #weight = np.ones(dim)/dim            
            #weight = (self.y_f.mean()/self.x_f.mean(0))/dim
            weight = np.ones(dim)/self.x_f.sum(1).std()
        elif method == 'correlation':
            weight = self.TFs_correlation / np.dot(self.x_f, self.TFs_correlation.reshape(-1,1)).std()
        elif method == 'PCA':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pca.fit_transform(self.x_f)
            weight = pca.components_[0]
        weight = weight.reshape(-1,1)
        WX = np.ravel(np.dot(self.x, weight))
        return WX, weight


    def init_par(self, alpha=None, beta=None, omega=None, theta=None, gamma=None, init_weight_method='correlation'):
        self.alpha = float(self.max_yf.mean()-self.min_yf.mean())/2 if alpha is None else alpha  
        self.beta = float(self.max_yf.mean()+self.min_yf.mean())/2 if beta is None else beta 
        self.omega = 2*np.pi if omega is None else omega 
        self.theta = 0 if theta is None else theta 
        self.gamma = 1 if gamma is None else gamma 

        if not init_weight_method is None:
            self.WX, self.weights_init = self.initialize_WX(method=init_weight_method) 
        self.t = self.assign_time(self.WX, self.y, self.alpha, self.beta, self.omega, self.theta, 
            self.gamma, self.filtered, norm_std=True) 

        # update object with initialized vars
        self.likelihood, self.varx = 0, 0

        # initialize time point assignment
        if not init_weight_method is None:
            self.WX_model, self.delta = self.update_WX(self.alpha, self.beta, self.omega, self.theta, 
                self.gamma, method=self.WX_method) 
        self.WX = use_model(self.WX_model, self.x, method=self.WX_method, bias=self.delta)
        max_yf_WX = use_model(self.WX_model, self.max_yf_x, method=self.WX_method, bias=self.delta)
        self.gamma = float(np.mean(max_yf_WX/self.max_yf))
        self.t = self.assign_time(self.WX, self.y, self.alpha, self.beta, self.omega, self.theta, 
            self.gamma, self.filtered, norm_std=True)
        self.pars = np.array([self.alpha, self.beta, self.omega, self.theta, self.gamma, self.delta,
            self.likelihood, self.varx, self.scaling_y] + list(self.scaling))[:, None]

        self.loss = [self.get_loss(refit_time=False, norm_std=True)]
        return

    def initialize(self):
        TFs_x, target_y, f = self.TFs_x, self.target_y, self.filtered
        TFs_x_f = TFs_x[f]
        target_y_f = target_y[f]

        # initialize scaling
        #for x_i in self.TFs_expression
        self.std_TFs_x, self.std_target_y = np.std(TFs_x_f, 0), np.std(target_y_f)
        #self.std_TFs_x[self.std_TFs_x==0] = 1e6 #TFs_x_f[:,self.std_TFs_x==0].mean(0)
        if self.std_target_y == 0:
            self.std_TFs_x = self.std_target_y = 1
        _scaling = self.fit_scaling
        if isinstance(_scaling, bool):
            if _scaling:
                scaling = 1/self.std_TFs_x
                scaling = np.nan_to_num(scaling, nan=0, posinf=0, neginf=0)
                scaling_y = 1/self.std_target_y
            else:
                scaling = np.ones_like(self.std_TFs_x)
                scaling_y = 1
        else:
            scaling = _scaling

        x, x_f = TFs_x * scaling, TFs_x_f * scaling
        y, y_f = target_y * scaling_y, target_y_f * scaling_y
        self.x, self.x_f, self.y, self.y_f = x, x_f, y, y_f
        self.scaling, self.scaling_y = scaling, scaling_y
        
        self.max_yf_idx = np.ravel(y_f >= np.percentile(y_f, 99, axis=0))
        self.max_yf_x = x_f[self.max_yf_idx]
        self.max_yf = np.ravel(y_f[self.max_yf_idx])
        self.min_yf_idx = np.ravel(y_f <= np.percentile(y_f, 1, axis=0))
        self.min_yf = np.ravel(y_f[self.min_yf_idx])

        self.init_par()
        return


    def fit(self):
        self.search_and_fit(to_update_weight=True)
        weight_check = np.ravel(self.weights_init * self.WX_model)
        mean_TFs = self.x_f.mean(0)
        #pos = (np.ravel((self.weights_init - self.WX_model)**2)*mean_TFs).sum()
        #neg = (np.ravel((self.weights_init + self.WX_model)**2)*mean_TFs).sum()
        if (weight_check*mean_TFs).sum() < 0:
            self.refit_flag = True
            #self.search_and_fit(to_update_weight=False) ###########
        else:
            self.refit_flag = False
        self.WX = use_model(self.WX_model, self.x, method=self.WX_method, bias=self.delta)
        self.t = self.assign_time(self.WX, self.y, self.alpha, self.beta, self.omega, self.theta,
            self.gamma, self.filtered, norm_std=True)
        self.varx = self.get_variance(to_filter=True, refit_time=False, norm_std=True)
        self.likelihood = self.get_likelihood(to_filter=True, varx=self.varx, refit_time=False, norm_std=True)
        self.get_raw_velocity()


    def save_par(self):
        self.best_loss = self.loss[-1]
        self.alpha_best, self.beta_best, self.omega_best, self.theta_best, self.gamma_best, self.delta_best = \
            self.alpha, self.beta, self.omega, self.theta, self.gamma, self.delta 
        self.loss_best = self.loss
        self.t_best = self.t
        self.WX_model_best = self.WX_model
        self.WX_best = self.WX
        return
    def load_par(self):
        self.alpha, self.beta, self.omega, self.theta, self.gamma, self.delta =\
            self.alpha_best, self.beta_best, self.omega_best, self.theta_best, self.gamma_best, self.delta_best
        self.loss = self.loss_best
        self.t = self.t_best 
        self.WX_model = self.WX_model_best
        self.WX = self.WX_best
        return
    def search_and_fit(self, to_update_weight, **kwargs):
        if not to_update_weight:
            self.WX_model = - self.WX_model
            self.delta = - self.delta
        omega_vales = [2*np.pi] #np.linspace(1/4*np.pi, 2*np.pi, num=8) 
        self.save_par()
        self.best_loss = MAX_LOSS
        for omega in omega_vales:
            theta_vals = np.linspace(-0.5, 0.5, num=1+int(omega/(np.pi/4))) * omega
            for theta in theta_vals:
                if to_update_weight == True:
                    self.init_par(omega=omega, theta=theta)
                    self.fit_paras_(**kwargs)
                else:
                    self.init_par(omega=omega, theta=theta, init_weight_method=None)
                    self.refit_paras_(**kwargs)
                if (self.best_loss<MAX_LOSS) and ((self.WX.max() - self.WX.min()) < 1/3 * (2*self.alpha*np.sqrt(4*np.pi**2 + self.gamma**2))):
                    continue
                if self.loss[-1] < self.best_loss:
                    self.save_par()
        self.load_par()
        self.tmp_best_loss = self.best_loss

        gamma_vals = np.linspace(2.5, 7.5, num=3) 
        for omega in omega_vales:
            theta_vals = np.linspace(-0.5, 0.5, num=1+int(omega/(np.pi/4))) * omega
            for theta in theta_vals:
                for gamma in gamma_vals:
                    if to_update_weight == True:
                        self.init_par(omega=omega, theta=theta, gamma=gamma)
                        self.fit_paras_(**kwargs)
                    else:
                        self.init_par(omega=omega, theta=theta, init_weight_method=None)
                        self.refit_paras_(**kwargs)
                    if (self.best_loss<MAX_LOSS) and ((self.WX.max() - self.WX.min()) < 1/3 * (2*self.alpha*np.sqrt(4*np.pi**2 + self.gamma**2))):
                        continue
                    if (self.loss[-1] < self.best_loss) and (self.loss[-1] < 0.9*self.tmp_best_loss):
                        self.save_par()
        self.load_par()
        return 
    def fit_paras_(self, **kwargs):
        def mse(data):
            #return self.get_mse(alpha=data[0], beta=data[1], omega=data[2], theta=data[3], gamma=data[4], **kwargs)
            return self.get_mse(alpha=data[0], beta=data[1], theta=data[2], gamma=data[3], **kwargs)
        #data_0, cb = np.array([self.alpha, self.beta, self.omega, self.theta, self.gamma]), self.cb_fit_paras_
        data_0, cb = np.array([self.alpha, self.beta, self.theta, self.gamma]), self.cb_fit_paras_
        #bounds=((0, None), (None, None), (0, 2*np.pi), (-np.pi, np.pi), (0, None))
        bounds=((0, None), (None, None), (-np.pi, np.pi), (0, None))
        res = minimize(mse, data_0, callback=cb, bounds=bounds, **self.simplex_kwargs)
        self.cb_fit_paras_(res.x)
    def cb_fit_paras_(self, data):
        #self.update(alpha=data[0], beta=data[1], omega=data[2], theta=data[3], gamma=data[4], refit_time=True, update_weight=True)
        self.update(alpha=data[0], beta=data[1], theta=data[2], gamma=data[3], refit_time=True, update_weight=True)

    def refit_paras_(self, **kwargs):
        def mse(data):
            return self.get_mse(alpha=data[0], beta=data[1], theta=data[2], gamma=data[3], delta=data[4], **kwargs)
        data_0, cb = np.array([self.alpha, self.beta, self.theta, self.gamma, self.delta]), self.cb_refit_paras_
        bounds=((0, None), (None, None), (-np.pi, np.pi), (0, None), (-10, 10))
        res = minimize(mse, data_0, callback=cb, bounds=bounds, **self.simplex_kwargs)
        self.cb_refit_paras_(res.x)
    def cb_refit_paras_(self, data):
        self.update(alpha=data[0], beta=data[1], theta=data[2], gamma=data[3], delta=data[4], refit_time=True, update_weight=False)
                     

    def update(
        self,
        t=None,
        alpha=None, 
        beta=None, 
        omega=None,
        theta=None,
        gamma=None,
        delta=None,
        refit_time=False,
        update_weight=True,
    ):
        loss_prev = self.loss[-1] if len(self.loss) > 0 else MAX_LOSS

        alpha, beta, omega, theta, gamma, delta = self.get_vars(alpha, beta, omega, theta, gamma, delta)
        
        if update_weight:
            WX_model, delta = self.update_WX(alpha, beta, omega, theta, gamma, method=self.WX_method)
        else:
            WX_model = self.WX_model
        
        WX = use_model(WX_model, self.x, method=self.WX_method, bias=delta)        
        if refit_time:
            t = self.assign_time(WX, self.y, alpha, beta, omega, theta, gamma, self.filtered, norm_std=True) 
        else:
            t = self.t

        loss = self.get_loss(t, alpha, beta, omega, theta, gamma, delta, WX_model, refit_time=False, norm_std=True)
        perform_update = loss < loss_prev

        if perform_update:
            self.alpha, self.beta, self.omega, self.theta, self.gamma, self.delta = alpha, beta, omega, theta, gamma, delta
            self.t = t
            self.WX = WX
            self.WX_model = WX_model 
            new_pars = np.array([alpha, beta, omega, theta, gamma, delta, self.likelihood, self.varx, self.scaling_y] + self.scaling.reshape(-1).tolist())[:, None]
            self.pars = np.c_[self.pars, new_pars]
            self.loss.append(loss)
            #print('loss:',loss)

        return perform_update


    def assign_time(
        self, WX, y, alpha, beta, omega, theta, gamma, filtered, norm_std=False
    ):
        num = np.clip(len(y), 1000, 2000)
        tpoints = np.linspace(0, 1, num=num)
        data_model = SplicingDynamics(alpha=alpha, beta=beta, omega=omega, theta=theta, 
            gamma=gamma).get_solution(tpoints)

        if norm_std:
            std_WX = get_norm_std(data_model[:,0], WX[filtered]) # WX[filtered], data_model[:,0]
            std_y = get_norm_std(data_model[:,1], y[filtered], par=1.5) # y[filtered], data_model[:,1]
        else:
            std_WX, std_y = 1, 1
        data_model = data_model/np.array([std_WX, std_y])
        data_obs = np.vstack([WX/std_WX, np.ravel(y)/std_y]).T
        t = tpoints[
            ((data_model[None, :, :] - data_obs[:, None, :]) ** 2).sum(axis=2).argmin(axis=1)
        ]
        return t


    def get_raw_velocity(self):
        self.velo_hat = use_model(self.WX_model, self.x, method=self.WX_method, bias=self.delta) - np.ravel(self.gamma * self.y)
        tmp1 = self.omega * self.t + self.theta
        self.y_t = self.alpha * np.sin(tmp1) + self.beta
        self.velo_t = self.alpha * self.omega * np.cos(tmp1) 
        self.velo_normed = (self.velo_hat-self.velo_hat.min()) / (self.velo_hat.max()-self.velo_hat.min()+1e-6) 
        return



default_pars_names = ["alpha", "beta", "omega", "theta", "gamma", "delta", "likelihood", "varx", "scaling_y", "refit_flags"]
default_pars_names_vec = ["scaling", "weights", "weights_init"]
default_pars_names += default_pars_names_vec


def read_pars(adata, pars_names=None, key="fit"):
    pars = []
    for name in default_pars_names if pars_names is None else pars_names:
        pkey = f"{key}_{name}"
        if name in default_pars_names_vec:
            par = np.zeros([adata.n_vars, adata.var['n_TFs'].max()])
        else:
            par = np.zeros(adata.n_vars) * np.nan
        if pkey in adata.var.keys():
            par = adata.var[pkey].values
        if pkey in adata.varm.keys():
            par = adata.varm[pkey]
        pars.append(par)

    return pars


default_results_names = ["velo_hat", "velo_t", "y_t", "velo_normed", "filtered", "WX", "y"]
def write_result(adata, results):
    for i, name in enumerate(default_results_names):
        adata.layers[name] = results[i]


def write_pars(adata, pars, pars_names=None, add_key="fit"):
    for i, name in enumerate(default_pars_names if pars_names is None else pars_names):
        if name in default_pars_names_vec:
            adata.varm[f"{add_key}_{name}"] = pars[i]
        else:
            adata.var[f"{add_key}_{name}"] = pars[i]


def recover_dynamics(
    data,
    var_names="all",#"velocity_genes",
    n_top_genes=None,
    max_iter=10,
    WX_method='Gaussian_Kernel',
    max_n_TF=999,
    t_max=None,
    fit_time=True,
    fit_scaling=True,
    fit_steady_states=True,
    fit_connected_states=None,
    fit_basal_transcription=None,
    use_raw=False,
    load_pars=None,
    return_model=None,
    plot_results=False,
    steady_state_prior=None,
    add_key="fit",
    copy=False,
    n_jobs=None,
    backend="loky",
    **kwargs,
):
    """Recovers the full splicing kinetics of specified genes.

    The model infers transcription rates, splicing rates, degradation rates,
    as well as cell-specific latent time and transcriptional states,
    estimated iteratively by expectation-maximization.

    .. image:: https://user-images.githubusercontent.com/31883718/69636459-ef862800-1056-11ea-8803-0a787ede5ce9.png

    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    var_names: `str`,  list of `str` (default: `'velocity_genes'`)
        Names of variables/genes to use for the fitting. If `var_names='velocity_genes'`
        but there is no column `'velocity_genes'` in `adata.var`, velocity genes are
        estimated using the steady state model.
    n_top_genes: `int` or `None` (default: `None`)
        Number of top velocity genes to use for the dynamical model.
    max_iter:`int` (default: `10`)
        Maximal iterations in the EM-Algorithm.
    t_max: `float`, `False` or `None` (default: `None`)
        Total range for time assignments.
    fit_scaling: `bool` or `float` or `None` (default: `True`)
        Whether to fit scaling between unspliced and spliced.
    fit_time: `bool` or `float` or `None` (default: `True`)
        Whether to fit time or keep initially given time fixed.
    fit_steady_states: `bool` or `None` (default: `True`)
        Whether to explicitly model and fit steady states (next to induction/repression)
    fit_connected_states: `bool` or `None` (default: `None`)
        Restricts fitting to neighbors given by connectivities.
    fit_basal_transcription: `bool` or `None` (default: `None`)
        Enables model to incorporate basal transcriptions.
    use_raw: `bool` or `None` (default: `None`)
        if True, use .layers['sliced'], else use moments from .layers['Ms']
    load_pars: `bool` or `None` (default: `None`)
        Load parameters from past fits.
    return_model: `bool` or `None` (default: `None`)
        Whether to return the model as :DynamicsRecovery: object.
    plot_results: `bool` or `None` (default: `False`)
        Plot results after parameter inference.
    steady_state_prior: list of `bool` or `None` (default: `None`)
        Mask for indices used for steady state regression.
    add_key: `str` (default: `'fit'`)
        Key to add to parameter names, e.g. 'fit_t' for fitted time.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.
    n_jobs: `int` or `None` (default: `None`)
        Number of parallel jobs.
    backend: `str` (default: "loky")
        Backend used for multiprocessing. See :class:`joblib.Parallel` for valid
        options.

    Returns
    -------
    fit_alpha: `.var`
        inferred transcription rates
    fit_beta: `.var`
        inferred splicing rates
    fit_gamma: `.var`
        inferred degradation rates
    fit_weight: `.var`
        inferred weight of a TF
    fit_t_: `.var`
        inferred switching time points
    fit_scaling: `.var`
        internal variance scaling factor for un/spliced counts
    fit_likelihood: `.var`
        likelihood of model fit
    fit_alignment_scaling: `.var`
        scaling factor to align gene-wise latent times to a universal latent time
    """  # noqa E501

    adata = data.copy() if copy else data

    n_jobs = get_n_jobs(n_jobs=n_jobs)
    logg.info(f"recovering dynamics (using {n_jobs}/{os.cpu_count()} cores)", r=True)

    if len(set(adata.var_names)) != len(adata.var_names):
        logg.warn("Duplicate var_names found. Making them unique.")
        adata.var_names_make_unique()

    if "M_total" not in adata.layers.keys():
        use_raw = True
    if fit_connected_states is None:
        fit_connected_states = not use_raw

    adata.uns["recover_dynamics"] = {
        "fit_connected_states": fit_connected_states,
        "fit_basal_transcription": fit_basal_transcription,
        "use_raw": use_raw,
    }

    if isinstance(var_names, str) and var_names not in adata.var_names:
        if var_names in adata.var.keys():
            if adata.var[var_names][0] in ['True', 'False']:
                adata.var[var_names] = adata.var[var_names].map({'True': True, 'False': False})
            var_names = adata.var_names[adata.var[var_names].values]
        elif use_raw or var_names == "all":
            var_names = adata.var_names
        elif "_genes" in var_names:
            from .velocity import Velocity

            velo = Velocity(adata, use_raw=use_raw)
            #velo.compute_deterministic(perc=[5, 95]) 
            #var_names = adata.var_names[velo._velocity_genes]
            #adata.var["fit_r2"] = velo._r2
        else:
            # raise ValueError("Variable name not found in var keys.")
            return False
    if not isinstance(var_names, str):
        var_names = list(np.ravel(var_names))

    var_names = make_unique_list(var_names, allow_array=True)
    var_names = np.array([name for name in var_names if name in adata.var_names])
    if len(var_names) == 0:
        # raise ValueError("Variable name not found in var keys.")
        return False
    if n_top_genes is not None and len(var_names) > n_top_genes:
        X = adata[:, var_names].layers[("total" if use_raw else "M_total")]
        var_names = var_names[np.argsort(np.sum(X, 0))[::-1][:n_top_genes]]
    if return_model is None:
        return_model = len(var_names) < 5

    pars = read_pars(adata)
    alpha, beta, omega, theta, gamma, delta, likelihood, varx, scaling_y, refit_flags = pars[:-3]
    scaling, weights, weights_init = pars[-3], pars[-2], pars[-1]
    # likelihood[np.isnan(likelihood)] = 0
    idx, L = [], []
    velo_hat, velo_t, velo_normed = np.zeros(adata.shape) * np.nan, np.zeros(adata.shape) * np.nan, np.zeros(adata.shape) * np.nan
    y_t, y, filtered = np.zeros(adata.shape) * np.nan, np.zeros(adata.shape) * np.nan, np.zeros(adata.shape) * np.nan
    WX = np.zeros(adata.shape) * np.nan
 
    T = np.zeros(adata.shape) * np.nan
    if "fit_t" in adata.layers.keys():
        T = adata.layers["fit_t"]

    conn = get_connectivities(adata) if fit_connected_states else None

    res = parallelize(
        _fit_recovery,
        var_names,
        n_jobs,
        unit="gene",
        as_array=False,
        backend=backend,
        show_progress_bar=len(var_names) > 9,
    )(
        adata=adata,
        use_raw=use_raw,
        load_pars=load_pars,
        max_iter=max_iter,
        fit_time=fit_time,
        fit_steady_states=fit_steady_states,
        fit_scaling=fit_scaling,
        fit_basal_transcription=fit_basal_transcription,
        conn=conn,
        WX_method=WX_method,
        max_n_TF=max_n_TF,
        **kwargs,
    )
    idx, dms = map(_flatten, zip(*res)) # gene_id_list, model_list

    for ix, dm in zip(idx, dms): 
        T[:, ix] = dm.t
        if WX_method in ['lsq_linear']:
            weight = np.ravel(dm.WX_model)
            weights[ix, :len(weight)] = weight
            weights_init[ix, :len(weight)] = np.ravel(dm.weights_init)
        alpha[ix], beta[ix], omega[ix], theta[ix], gamma[ix], delta[ix] = dm.alpha, dm.beta, dm.omega, dm.theta, dm.gamma, dm.delta #dm.pars[:5, -1]
        scaling[ix][:dm.n_TFs], scaling_y[ix] = dm.scaling, dm.scaling_y # dm.pars[10, -1],  dm.pars[7, -1]
        likelihood[ix], varx[ix] = dm.likelihood, dm.varx # dm.pars[5, -1],  dm.pars[6, -1]
        velo_hat[:,ix], velo_t[:,ix], velo_normed[:,ix] = dm.velo_hat.reshape(-1), dm.velo_t.reshape(-1), dm.velo_normed.reshape(-1)
        y_t[:,ix], y[:,ix], filtered[:,ix] =  dm.y_t.reshape(-1), dm.y.reshape(-1), dm.filtered.reshape(-1)
        WX[:,ix] = dm.WX.reshape(-1)
        L.append(dm.loss)
        refit_flags[ix] = dm.refit_flag

    _pars = [
        alpha, beta, omega, theta, gamma, delta, likelihood, varx, scaling_y, refit_flags, \
        scaling, weights, weights_init
    ]
    write_pars(adata, _pars)
    write_result(adata, [velo_hat, velo_t, y_t, velo_normed, filtered, WX, y]) 

    adata.layers["fit_t_raw"] = T.copy() 
    #if "fit_t" in adata.layers.keys():
    #    adata.layers["fit_t"][:, idx] = (
    #        T[:, idx] if conn is None else conn.dot(T[:, idx])
    #    )
    #else:
    adata.layers["fit_t"] = T if conn is None else conn.dot(T)

    if L:  # is False if only one invalid / irrecoverable gene was given in var_names
        cur_len = adata.varm["loss"].shape[1] if "loss" in adata.varm.keys() else 2
        max_len = max(np.max([len(loss) for loss in L]), cur_len) if L else cur_len
        loss = np.ones((adata.n_vars, max_len)) * np.nan

        if "loss" in adata.varm.keys():
            loss[:, :cur_len] = adata.varm["loss"]

        loss[idx] = np.vstack(
            [
                np.concatenate([loss, np.ones(max_len - len(loss)) * np.nan])
                for loss in L
            ]
        )
        adata.varm["loss"] = loss


    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        f"    '{add_key}_pars', "
        f"fitted parameters for splicing dynamics (adata.var)"
    )

    '''
    P = []
    if plot_results:  # Plot Parameter Stats
        n_rows, n_cols = len(var_names[:4]), 6
        figsize = [2 * n_cols, 1.5 * n_rows]  # rcParams['figure.figsize']
        fontsize = rcParams["font.size"]
        fig, axes = pl.subplots(nrows=n_rows, ncols=6, figsize=figsize)
        pl.subplots_adjust(wspace=0.7, hspace=0.5)
        for i, gene in enumerate(var_names[:4]):
            #if t_max is not False:
            #    mi = dm.m[i]
            #    P[i] *= np.array([1 / mi, 1 / mi, 1 / mi, mi, 1])[:, None]
            ax = axes[i] if n_rows > 1 else axes
            for j, pij in enumerate(P[i]):
                ax[j].plot(pij)
            ax[len(P[i])].plot(L[i])
            if i == 0:
                pars_names = ["alpha", "beta", "gamma", "weight", "t_", "scaling", "loss"]
                for j, name in enumerate(pars_names):
                    ax[j].set_title(name, fontsize=fontsize)
    '''
    if return_model:
        logg.info("\noutputs model fit of gene:", dm.gene)

    #return dm if return_model else adata if copy else None
    return True



def latent_time(
    data,
    vkey="velocity",
    min_likelihood=0.1,
    min_confidence=0.75,
    min_corr_diffusion=None,
    weight_diffusion=None,
    root_key=None,
    end_key=None,
    t_max=None,
    copy=False,
):
    """Computes a gene-shared latent time.

    Gene-specific latent timepoints obtained from the dynamical model are coupled to a
    universal gene-shared latent time, which represents the cell’s internal clock and
    is based only on its transcriptional dynamics.

    .. image:: https://user-images.githubusercontent.com/31883718/69636500-03318e80-1057-11ea-9e14-ae9f907711cc.png

    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    min_likelihood: `float` between `0` and `1` or `None` (default: `.1`)
        Minimal likelihood fitness for genes to be included to the weighting.
    min_confidence: `float` between `0` and `1` (default: `.75`)
        Parameter for local coherence selection.
    min_corr_diffusion: `float` between `0` and `1` or `None` (default: `None`)
        Only select genes that correlate with velocity pseudotime obtained
        from diffusion random walk on velocity graph.
    weight_diffusion: `float` or `None` (default: `None`)
        Weight applied to couple latent time with diffusion-based velocity pseudotime.
    root_key: `str` or `None` (default: `'root_cells'`)
        Key (.uns, .obs) of root cell to be used.
        If not set, it obtains root cells from velocity-inferred transition matrix.
    end_key: `str` or `None` (default: `None`)
        Key (.obs) of end points to be used.
    t_max: `float` or `None` (default: `None`)
        Overall duration of differentiation process.
        If not set, a overall transcriptional timescale of 20 hours is used as prior.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.

    Returns
    -------
    latent_time: `.obs`
        latent time from learned dynamics for each cell
    """  # noqa E501

    adata = data.copy() if copy else data

    from .dynamical_model_utils import compute_shared_time, root_time
    from .terminal_states import terminal_states
    from .utils import scale, vcorrcoef
    from .velocity_graph import velocity_graph
    from .velocity_pseudotime import velocity_pseudotime

    if "fit_t" not in adata.layers.keys():
        raise ValueError("you need to run `tl.recover_dynamics` first.")

    if f"{vkey}_graph" not in adata.uns.keys():
        velocity_graph(adata, approx=True)

    if root_key is None:
        terminal_keys = ["root_cells", "starting_cells", "root_states_probs"]
        keys = [key for key in terminal_keys if key in adata.obs.keys()]
        if len(keys) > 0:
            root_key = keys[0]
    if root_key not in adata.uns.keys() and root_key not in adata.obs.keys():
        root_key = "root_cells"
    if root_key not in adata.obs.keys():
        terminal_states(adata, vkey=vkey)

    t = np.array(adata.layers["fit_t"])
    idx_valid = ~np.isnan(t.sum(0))
    if min_likelihood is not None:
        likelihood = adata.var["fit_likelihood"].values
        idx_valid &= np.array(likelihood >= min_likelihood, dtype=bool)
    t = t[:, idx_valid]
    t_sum = np.sum(t, 1)
    conn = get_connectivities(adata)

    if root_key not in adata.uns.keys():
        roots = np.argsort(t_sum)
        idx_roots = np.array(adata.obs[root_key][roots])
        idx_roots[pd.isnull(idx_roots)] = 0
        if np.any([isinstance(ix, str) for ix in idx_roots]):
            idx_roots = np.array([isinstance(ix, str) for ix in idx_roots], dtype=int)
        idx_roots = idx_roots.astype(float) > 1 - 1e-3
        if np.sum(idx_roots) > 0:
            roots = roots[idx_roots]
        else:
            logg.warn(
                "No root cells detected. Consider specifying "
                "root cells to improve latent time prediction."
            )
    else:
        roots = [adata.uns[root_key]]
        root_key = f"root cell {adata.uns[root_key]}"

    if end_key in adata.obs.keys():
        fates = np.argsort(t_sum)[::-1]
        idx_fates = np.array(adata.obs[end_key][fates])
        idx_fates[pd.isnull(idx_fates)] = 0
        if np.any([isinstance(ix, str) for ix in idx_fates]):
            idx_fates = np.array([isinstance(ix, str) for ix in idx_fates], dtype=int)
        idx_fates = idx_fates.astype(float) > 1 - 1e-3
        if np.sum(idx_fates) > 0:
            fates = fates[idx_fates]
    else:
        fates = [None]

    logg.info(
        f"computing latent time using {root_key}"
        f"{', ' + end_key if end_key in adata.obs.keys() else ''} as prior",
        r=True,
    )

    VPT = velocity_pseudotime(
        adata, vkey, root_key=roots[0], end_key=fates[0], return_model=True
    )
    vpt = VPT.pseudotime

    if min_corr_diffusion is not None:
        corr = vcorrcoef(t.T, vpt)
        t = t[:, np.array(corr >= min_corr_diffusion, dtype=bool)]

    if root_key in adata.uns.keys():
        root = adata.uns[root_key]
        t, t_ = root_time(t, root=root)
        latent_time = compute_shared_time(t)
    else:
        roots = roots[:4]
        latent_time = np.ones(shape=(len(roots), adata.n_obs))
        for i, root in enumerate(roots):
            t, t_ = root_time(t, root=root)
            latent_time[i] = compute_shared_time(t)
        latent_time = scale(np.mean(latent_time, axis=0))

    if fates[0] is not None:
        fates = fates[:4]
        latent_time_ = np.ones(shape=(len(fates), adata.n_obs))
        for i, fate in enumerate(fates):
            t, t_ = root_time(t, root=fate)
            latent_time_[i] = 1 - compute_shared_time(t)
        latent_time = scale(latent_time + 0.2 * scale(np.mean(latent_time_, axis=0)))

    tl = latent_time
    tc = conn.dot(latent_time)

    z = tl.dot(tc) / tc.dot(tc)
    tl_conf = (1 - np.abs(tl / np.max(tl) - tc * z / np.max(tl))) ** 2
    idx_low_confidence = tl_conf < min_confidence

    if weight_diffusion is not None:
        w = weight_diffusion
        latent_time = (1 - w) * latent_time + w * vpt
        latent_time[idx_low_confidence] = vpt[idx_low_confidence]
    else:
        conn_new = conn.copy()
        conn_new[:, idx_low_confidence] = 0
        conn_new.eliminate_zeros()
        latent_time = conn_new.dot(latent_time)

    latent_time = scale(latent_time)
    if t_max is not None:
        latent_time *= t_max

    adata.obs["latent_time"] = latent_time

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added \n" "    'latent_time', shared time (adata.obs)")
    return adata if copy else None


recover_latent_time = latent_time


def differential_kinetic_test(
    data,
    var_names="velocity_genes",
    groupby=None,
    use_raw=None,
    return_model=None,
    add_key="fit",
    copy=None,
    **kwargs,
):
    """Test to detect cell types / lineages with different kinetics.

    Likelihood ratio test for differential kinetics to detect clusters/lineages that
    display kinetic behavior that cannot be sufficiently explained by a single model
    for the overall dynamics. Each cell type is tested whether an independent fit yields
    a significantly improved likelihood.

    .. image:: https://user-images.githubusercontent.com/31883718/78930730-dc737200-7aa4-11ea-92f6-269b7609c3a5.png

    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    var_names: `str`,  list of `str` (default: `'velocity_genes`)
        Names of variables/genes to use for the fitting.
    groupby: `str` (default: `None`)
        Key of observations grouping to consider, e.g. `'clusters'`.
    use_raw: `bool` (default: `False`)
        Whether to use raw data for estimation.
    add_key: `str` (default: `'fit'`)
        Key to add to parameter names, e.g. 'fit_t' for fitted time.
    copy: `bool` (default: `None`)
        Return a copy instead of writing to `adata`.

    Returns
    -------
    fit_pvals_kinetics: `.varm`
        P-values of competing kinetic for each group and gene
    fit_diff_kinetics: `.var`
        Groups that have differential kinetics for each gene.
    """  # noqa E501

    adata = data.copy() if copy else data

    if "Ms" not in adata.layers.keys() or "Mu" not in adata.layers.keys():
        use_raw = True
    if isinstance(var_names, str) and var_names not in adata.var_names:
        if var_names in adata.var.keys():
            var_names = adata.var_names[adata.var[var_names].values]
        elif use_raw or var_names == "all":
            var_names = adata.var_names
        elif "_genes" in var_names:
            from .velocity import Velocity

            velo = Velocity(adata, use_raw=use_raw)
            velo.compute_deterministic(perc=[5, 95])
            var_names = adata.var_names[velo._velocity_genes]
            adata.var["fit_r2"] = velo._r2
        else:
            # raise ValueError("Variable name not found in var keys.")
            return False
    if not isinstance(var_names, str):
        var_names = list(np.ravel(var_names))

    var_names = make_unique_list(var_names, allow_array=True)
    var_names = [name for name in var_names if name in adata.var_names]
    if len(var_names) == 0:
        # raise ValueError("Variable name not found in var keys.")
        return False
    if return_model is None:
        return_model = len(var_names) < 5

    # fit dynamical model first, if not done yet.
    var_names_for_fit = (
        adata.var_names[np.isnan(adata.var["fit_alpha"])].intersection(var_names)
        if "fit_alpha" in adata.var.keys()
        else var_names
    )
    if len(var_names_for_fit) > 0:
        recover_dynamics(adata, var_names_for_fit)

    logg.info("testing for differential kinetics", r=True)

    if groupby is None:
        groupby = (
            "clusters"
            if "clusters" in adata.obs.keys()
            else "louvain"
            if "louvain" in adata.obs.keys()
            else None
        )
    clusters = adata.obs[groupby] if isinstance(groupby, str) else groupby
    groups = clusters.cat.categories
    pars_names = ["diff_kinetics", "pval_kinetics"]
    diff_kinetics, pval_kinetics = read_pars(adata, pars_names=pars_names)

    pvals = None
    if "fit_pvals_kinetics" in adata.varm.keys():
        pvals = pd.DataFrame(adata.varm["fit_pvals_kinetics"]).to_numpy()
    if pvals is None or pvals.shape[1] != len(groups):
        pvals = np.zeros((adata.n_vars, len(groups))) * np.nan
    if "fit_diff_kinetics" in adata.var.keys():
        diff_kinetics = np.array(adata.var["fit_diff_kinetics"])
    else:
        diff_kinetics = np.empty(adata.n_vars, dtype="object")
    idx = []

    progress = logg.ProgressReporter(len(var_names))
    for i, gene in enumerate(var_names):
        dm = DynamicsRecovery(adata, gene, use_raw=use_raw, load_pars=True, max_iter=0)
        if dm.recoverable:
            dm.differential_kinetic_test(clusters, **kwargs)

            ix = adata.var_names.get_loc(gene)
            idx.append(ix)
            diff_kinetics[ix] = dm.diff_kinetics
            pval_kinetics[ix] = dm.pval_kinetics
            pvals[ix] = np.array(dm.pvals_kinetics)

            progress.update()
        else:
            logg.warn(dm.gene, "not recoverable due to insufficient samples.")
            dm = None
    progress.finish()

    pars_names = ["diff_kinetics", "pval_kinetics"]
    write_pars(adata, [diff_kinetics, pval_kinetics], pars_names=pars_names)
    adata.varm[f"{add_key}_pvals_kinetics"] = np.rec.fromarrays(
        pvals.T, dtype=[(f"{rn}", "float32") for rn in groups]
    ).T
    adata.uns["recover_dynamics"]["fit_diff_kinetics"] = groupby

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        f"    '{add_key}_diff_kinetics', "
        f"clusters displaying differential kinetics (adata.var)\n"
        f"    '{add_key}_pvals_kinetics', "
        f"p-values of differential kinetics (adata.var)"
    )

    if return_model:
        logg.info("\noutputs model fit of gene:", dm.gene)

    return dm if return_model else adata if copy else None


def rank_dynamical_genes(data, n_genes=100, groupby=None, copy=False):
    """Rank genes by likelihoods per cluster/regime.

    This ranks genes by their likelihood obtained from the
    dynamical model grouped by clusters specified in groupby.

    Arguments
    ----------
    data : :class:`~anndata.AnnData`
        Annotated data matrix.
    n_genes : `int`, optional (default: 100)
        The number of genes that appear in the returned tables.
    groupby: `str`, `list` or `np.ndarray` (default: `None`)
        Key of observations grouping to consider.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to data.

    Returns
    -------
    rank_dynamical_genes : `.uns`
        Structured array to be indexed by group id storing the gene
        names. Ordered according to scores.
    """

    from .dynamical_model_utils import get_divergence

    adata = data.copy() if copy else data

    logg.info("ranking genes by cluster-specific likelihoods", r=True)

    groupby = (
        groupby
        if isinstance(groupby, str) and groupby in adata.obs.keys()
        else "clusters"
        if "clusters" in adata.obs.keys()
        else "louvain"
        if "louvain" in adata.obs.keys()
        else "velocity_clusters"
        if "velocity_clusters" in adata.obs.keys()
        else None
    )

    vdata = adata[:, ~np.isnan(adata.var["fit_alpha"])]
    groups = vdata.obs[groupby].cat.categories

    ll = get_divergence(
        vdata,
        mode="gene_likelihood",
        use_connectivities=True,
        clusters=adata.obs[groupby],
    )
    idx_sorted = np.argsort(np.nan_to_num(ll), 1)[:, ::-1][:, :n_genes]
    rankings_gene_names = vdata.var_names[idx_sorted]
    rankings_gene_scores = np.sort(np.nan_to_num(ll), 1)[:, ::-1][:, :n_genes]

    key = "rank_dynamical_genes"
    if key not in adata.uns.keys():
        adata.uns[key] = {}

    adata.uns[key] = {
        "names": np.rec.fromarrays(
            [n for n in rankings_gene_names],
            dtype=[(f"{rn}", "U50") for rn in groups],
        ),
        "scores": np.rec.fromarrays(
            [n.round(2) for n in rankings_gene_scores],
            dtype=[(f"{rn}", "float32") for rn in groups],
        ),
    }

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint("added \n" f"    '{key}', sorted scores by group ids (adata.uns)")

    return adata if copy else None


def _fit_recovery(
    var_names,
    adata,
    use_raw,
    load_pars,
    max_iter,
    fit_time,
    fit_steady_states,
    conn,
    fit_scaling,
    fit_basal_transcription,
    max_n_TF,
    queue,
    WX_method,
    **kwargs,
):

    idx, dms = [], []
    for gene in var_names:#################
        dm = DynamicsRecovery(
            adata,
            gene,
            use_raw=use_raw,
            load_pars=load_pars,
            max_iter=max_iter,
            fit_time=fit_time,
            fit_steady_states=fit_steady_states,
            fit_connected_states=conn,
            fit_scaling=fit_scaling,
            fit_basal_transcription=fit_basal_transcription,
            WX_method=WX_method,
            max_n_TF=max_n_TF,
            **kwargs,
        )
        tmp = str(list(var_names).index(gene)) + '/' + str(len(var_names))
        if dm.recoverable:
            #try:
            print('Processing', tmp, dm.gene)
            dm.fit()

            ix = np.where(adata.var_names == gene)[0][0]
            idx.append(ix)
            dms.append(dm)
            print(tmp, dm.gene, 'FINISHED with n_TFs:', dm.n_TFs)
            #except:
            #    dm.recoverable = False
            #    logg.warn(dm.gene, "not recoverable during optimization.")

        else:
            logg.warn(tmp, dm.gene, "not recoverable due to insufficient samples.")

        if queue is not None:
            queue.put(1)


    if queue is not None:
        queue.put(None)

    return idx, dms # gene_id_list, model_list


def _flatten(iterable):
    return [i for it in iterable for i in it]
