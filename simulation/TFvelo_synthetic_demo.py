from optparse import NO_DEFAULT
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from matplotlib.pyplot import MultipleLocator
from cmath import isnan
import scipy
from scipy.optimize import minimize, lsq_linear
import seaborn as sns
import os, sys
from TFvelo.core import SplicingDynamics
from TFvelo.tools.dynamical_model_utils import get_norm_std, use_model 

MAX_LOSS = 1e6

class DynamicsRecovery():
    def __init__(self, x, y, max_iter, WX_thres):
        self.TFs_x, self.target_y, self.filtered = x, y, np.ones(x.shape[0], dtype=bool)
        self.n_TFs = x.shape[1]
        self.simplex_kwargs = {
            "method": "Nelder-Mead",
            "options": {"maxiter": int(max_iter)},
        }
        self.WX_method = 'lsq_linear'
        self.WX_thres = WX_thres
        self.fit_scaling = False
        self.initialize()

    def get_correlation(self):
        dim = self.n_TFs
        target_expression = np.ravel(self.y_f)
        correlations = []
        for i in range(dim):
            TF_expression = self.x_f[:, i]
            flag = (TF_expression>0) & (target_expression>0)
            correlation, _ = scipy.stats.spearmanr(target_expression[flag], TF_expression[flag])
            correlations.append(correlation)
        return np.array(correlations)

    def initialize_WX(self, method):
        dim = self.n_TFs
        if method == 'rand':
            weight = np.random.normal(0.45, 1, dim)
        elif method == 'ones':
            weight = np.ones(dim)/self.x_f.sum(1).std()
        elif method == 'correlation':
            weight = self.get_correlation()
        elif method == 'PCA':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pca.fit_transform(self.x_f)
            weight = pca.components_[0]
        weight = weight.reshape(-1,1)
        WX = np.ravel(np.dot(self.x, weight))
        return WX


    def init_par(self, alpha=None, beta=None, omega=None, theta=None, gamma=None, init_weight_method='correlation'):
        self.alpha = float(self.max_yf.mean()-self.min_yf.mean())/2 if alpha is None else alpha  
        self.beta = float(self.max_yf.mean()+self.min_yf.mean())/2 if beta is None else beta 
        self.omega = 2 * np.pi if omega is None else omega 
        self.theta = 0 if theta is None else theta 
        self.gamma = 1 if gamma is None else gamma 

        self.WX = self.initialize_WX(method=init_weight_method) 
        self.t = self.assign_time(self.WX, self.y, self.alpha, self.beta, self.omega, self.theta, 
            self.gamma, self.filtered, norm_std=True) 

        # update object with initialized vars
        self.likelihood, self.varx = 0, 0

        # initialize time point assignment
        self.WX_model, self.delta = self.update_WX(self.alpha, self.beta, self.omega, self.theta, 
            self.gamma, method=self.WX_method, thres=self.WX_thres) 
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
        
        self.max_yf_idx = np.ravel(y_f >= np.percentile(y_f, 98, axis=0))
        self.max_yf_x = x_f[self.max_yf_idx]
        self.max_yf = np.ravel(y_f[self.max_yf_idx])
        self.min_yf_idx = np.ravel(y_f <= np.percentile(y_f, 2, axis=0))
        self.min_yf = np.ravel(y_f[self.min_yf_idx])

        self.init_par()
        return

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
    ):
        kwargs = dict(t=t, alpha=alpha, beta=beta, omega=omega, theta=theta, gamma=gamma, delta=delta, WX_model=WX_model)
        kwargs.update(dict(refit_time=refit_time, norm_std=norm_std))
        #######return self.get_se(**kwargs)
        return self.get_mse(**kwargs)


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


    def update_WX(self, alpha, beta, omega, theta, gamma, method, thres=1): 
        t = self.t
        WX_t, y_t = SplicingDynamics(alpha=alpha, beta=beta, omega=omega, theta=theta, gamma=gamma).get_solution(
            t, stacked=False)
        
        if method == 'constant':
            dim = self.n_TFs
            #tmp = gamma - np.sqrt(gamma*gamma+8*beta)
            #max_WX = (gamma-tmp/2)*alpha*np.exp(-tmp*tmp/(16*beta)) + gamma*rho
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
            if 0: # [X,1]
                added_col = np.ones([self.x_f.shape[0], 1])
                input_x = np.concatenate([self.x_f, added_col], axis=1)
                #lb = np.append(- thres * np.ones(self.n_TFs), -1)
                #hb = np.append(thres * np.ones(self.n_TFs), 1)
                #res = lsq_linear(input_x, WX_t, bounds=(lb, hb), lsmr_tol='auto', verbose=0)
                res = lsq_linear(input_x, WX_t, bounds=(-thres, thres), lsmr_tol='auto', verbose=0)
                model = res.x.reshape(-1, 1)
                model = [model[:-1], model[-1][0]]
            else: # X
                res = lsq_linear(self.x_f, WX_t, bounds=(-thres, thres), lsmr_tol='auto', verbose=0)
                model = res.x.reshape(-1, 1)
                model = [model, 0]

        return model


    def get_filter(self, to_filter=None):
        filtered = (
            np.array(self.filtered
            )
            if to_filter
            else np.ones(len(self.weight), bool)
        )
        return filtered

    def get_reads(self, to_filter=None):
        x, y = self.x, self.y
        if to_filter:
            filtered = self.get_filter(to_filter=to_filter)
            x, y = x[filtered], y[filtered]
        return x, y


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

        t = self.t

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

    def get_dist(self, noise_model="normal", regularize=True, **kwargs):
        WX_diff, y_diff, reg = self.get_dists(**kwargs)
        if noise_model == "normal":
            dist_data = WX_diff ** 2 + y_diff ** 2
        elif noise_model == "laplace":
            dist_data = np.abs(WX_diff) + np.abs(y_diff)
        if regularize:
            dist_data += reg ** 2
        return dist_data

    def get_mse(self, **kwargs):
        return np.mean(self.get_dist(**kwargs))


    def fit(self):
        self.search_and_fit()
      
        self.WX = use_model(self.WX_model, self.x, method=self.WX_method, bias=self.delta)
        self.t = self.assign_time(self.WX, self.y, self.alpha, self.beta, self.omega, self.theta,
            self.gamma, self.filtered, norm_std=True)
        self.get_raw_velocity()
        self.weight = self.WX_model


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
    def search_and_fit(self, **kwargs):
        omega_vales = [2*np.pi] #np.linspace(1/4*np.pi, 2*np.pi, num=8) 
        self.save_par()
        self.best_loss = MAX_LOSS
        for omega in omega_vales:
            theta_vals = np.linspace(-0.5, 0.5, num=1+int(omega/(np.pi/4))) * omega
            for theta in theta_vals:
                self.init_par(omega=omega, theta=theta)
                self.fit_paras_(**kwargs)
                if (self.best_loss<MAX_LOSS) and ((self.WX.max() - self.WX.min()) < 0.2 * (2*self.alpha*np.sqrt(4*np.pi**2 + self.gamma**2))):
                    continue
                if self.loss[-1] < self.best_loss:
                    self.save_par()
        self.load_par()
        self.tmp_best_loss = self.best_loss
        if 1:   
            gamma_vals = np.linspace(2.5, 7.5, num=3) 
            for omega in omega_vales:
                theta_vals = np.linspace(-0.5, 0.5, num=1+int(omega/(np.pi/4))) * omega
                for theta in theta_vals:
                    for gamma in gamma_vals:
                        self.init_par(omega=omega, theta=theta, gamma=gamma)
                        self.fit_paras_(**kwargs)
                        if (self.best_loss<MAX_LOSS) and ((self.WX.max() - self.WX.min()) < 0.2 * (2*self.alpha*np.sqrt(4*np.pi**2 + self.gamma**2))):
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
        self.update(alpha=data[0], beta=data[1], theta=data[2], gamma=data[3], refit_time=True, update_weight=True)
             

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
            WX_model, delta = self.update_WX(alpha, beta, omega, theta, gamma, method=self.WX_method, thres=self.WX_thres)
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




def compute_dynamics(n_cells, alpha, beta, omega, theta, gamma, tpoints=None):
    num = np.clip(int(n_cells/5), 1000, 2000)
    if tpoints is None:
        tpoints = np.linspace(0, 1, num=num) #########################################
    WX_t, y_t = SplicingDynamics(alpha=alpha, beta=beta, omega=omega, theta=theta, 
        gamma=gamma).get_solution(tpoints, stacked=False)
    v_t = WX_t - gamma*y_t
    return WX_t, y_t, tpoints, v_t


def my_curve(result_path, alpha, beta, omega, theta, gamma, delta, 
    to_draw=1, n_cells=1000, tpoints=None, save_name='', ref_points=False):    

    WX_t, y_t, tpoints, v_t = compute_dynamics(n_cells, alpha, beta, omega, theta, gamma, tpoints)
    
    if to_draw:
        linewidth = 1
        label = "learned dynamics"
        key = 'fit'
        color = "red"
        fig = plt.figure()
        ax = axisartist.Subplot(fig, 111)        
        fig.add_axes(ax)
        ax.axis["bottom"].set_axisline_style("->", size = 1.5)
        ax.axis["left"].set_axisline_style("->", size = 1.5)
        ax.axis["top"].set_visible(False)
        ax.axis["right"].set_visible(False)        
        plt.plot(y_t, WX_t, color=color, linewidth=linewidth, label=label)
        if ref_points:
            idx = np.array(np.arange(10)*len(WX_t)/10, dtype=int)
            plt.plot(y_t[idx], WX_t[idx], "ro")
        text = str(alpha)[0:5] + ', ' + str(beta)[0:5] + ', ' + str(omega)[0:5] +\
            ', ' + str(theta)[0:5]+ ', ' + str(gamma)[0:5] + ', ' + str(delta)[0:5]    
        plt.text(0.05, 0.95, text, fontdict={'size':'16','color':'Red'},  transform = plt.gca().transAxes)
        plt.xlabel("y")
        plt.ylabel("WX")
        #plt.title(save_name)

        plt.savefig(result_path + save_name+'.png', bbox_inches='tight', pad_inches=0.5)
        plt.close()
        plt.clf()
    return WX_t, y_t, tpoints, v_t

def my_plot(result_path, t, x, y, v_x, v_y, alpha, beta, omega, theta, gamma, delta, target, text=None,
             ref_line=False, velo_arrow=False, save_name='', ref_points=False, show_full_dynamic=False, draw_axis=True):
    x=x
    y=y  
    fig = plt.figure() 
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis["bottom"].set_axisline_style("->", size = 3)
    ax.axis["left"].set_axisline_style("->", size = 3)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)    
    #ax.set_xlabel('y', fontsize=20) 
    #ax.set_ylabel('WX', fontsize=20) 

    sc = plt.scatter(x, y, c=v_x, s=20, alpha=0.9, cmap='rainbow')
    plt.colorbar(sc)
    #plt.scatter(x[np.isnan(v_x)], y[np.isnan(v_x)], s=s, color='black')
    if text is not None:
        plt.text(0.05, 0.95, text, fontdict={'size':'20','color':'Black'},  transform = plt.gca().transAxes)

    if ref_line:
        min_v = max(min(x), min(y))
        max_v = min(max(x), max(y))
        plt.plot([min_v,max_v],[min_v,max_v],c='grey',ls='--')
    #plt.axis('scaled') # x,y 比例尺相同

    WX_t, y_t, tpoints, v_t = compute_dynamics(len(y), alpha, beta, omega, theta, gamma)
    if not show_full_dynamic:
        data = t
        if not 0 in data:
            data = np.insert(data, 0, 0)
        if not 1 in data:
            data = np.insert(data, len(data), 1)
        sorted_data = np.sort(data)
        intervals = np.diff(sorted_data)
        blank_start_id = np.argmax(intervals)
        if (blank_start_id==0) or (blank_start_id==len(intervals)-1):
            nonblank_idx = (tpoints>sorted_data[1]) & (tpoints<sorted_data[len(intervals)-1])
            mid_idx = int((nonblank_idx).sum()/2)
        else:
            nonblank_idx = (tpoints<sorted_data[blank_start_id]) | (tpoints>sorted_data[blank_start_id+1])
            mid_idx = int((nonblank_idx).sum()/2) - (tpoints>sorted_data[blank_start_id+1]).sum()
        WX_t = WX_t[nonblank_idx]
        y_t = y_t[nonblank_idx]
        mid_dt_idx = min(mid_idx+20, len(WX_t)-1)
        ax.quiver(y_t[mid_idx], WX_t[mid_idx], y_t[mid_dt_idx]-y_t[mid_idx], WX_t[mid_dt_idx]-WX_t[mid_idx],
                    scale=0.2, width=0.05, color='purple', angles='xy', scale_units='xy')
    plt.scatter(y_t, WX_t, color="Purple", linewidth=2, label="learned dynamics")
    
    if ref_points:
        idx = np.array(np.arange(10)*len(WX_t)/10, dtype=int)
        plt.plot(y_t[idx], WX_t[idx], "ro")

    dt = 0.1
    if velo_arrow:
        for i in range(len(x)): 
            if isnan(v_x[i]) or isnan(v_y[i]) or ((v_x[i]==0) and (v_y[i]==0)):
                continue
            else:
                plt.arrow(x[i],y[i], v_x[i]*dt,v_y[i]*dt, head_width=(y.max()-y.min())/100, color='grey')

    if not draw_axis:
        plt.xticks([])
        plt.yticks([])
        #plt.axis('off')
    #plt.title(target)
    plt.savefig(result_path +target+ save_name+'.png', bbox_inches='tight', pad_inches=0.5)
    plt.close()
    plt.clf()

def my_plot_t(result_path, t, WX_hat, y_hat, v_target, alpha, beta, omega, theta, gamma, delta,
    target, save_name='', text=None, ref_points=False, draw_axis=True, show_full_dynamic=False):
    WX_t, y_t, tpoints, v_t = compute_dynamics(len(y_hat), alpha, beta, omega, theta, gamma)

    for tmp in range(2):
        if tmp==0:
            value = WX_hat
            value_t = WX_t
            value_name = 'WX'
        else:
            value = y_hat
            value_t = y_t
            value_name = 'y'

        fig = plt.figure()
        ax = axisartist.Subplot(fig, 111)        
        fig.add_axes(ax)
        ax.axis["bottom"].set_axisline_style("->", size = 3)
        ax.axis["left"].set_axisline_style("->", size = 3)
        ax.axis["top"].set_visible(False)
        ax.axis["right"].set_visible(False)  
        #plt.xlabel('t', fontdict={'family':'Times New Roman', 'size':20})
        #plt.ylabel(value_name, fontdict={'family':'Times New Roman', 'size':20})
        sc = plt.scatter(t, value, c=v_target, s=20, alpha=0.9, cmap='rainbow')
        #plt.colorbar(sc)
        if ref_points:
            idx = np.arange(10)
            plt.scatter(t[idx], value[idx], c=idx, cmap='rainbow')
        if text is not None:
            plt.text(0.05, 0.88, text, fontdict={'size':'20','color':'Black'},  transform = plt.gca().transAxes)
        nonblank_idx = (tpoints>=t.min()) & (tpoints<=t.max())
        if show_full_dynamic:
            plt.plot(tpoints, value_t, color="Purple", linewidth=5, label="learned dynamics")
        else:
            tpoints_ = tpoints[nonblank_idx]
            value_t_ = value_t[nonblank_idx]
            plt.plot(tpoints_, value_t_, color="Purple", linewidth=5, label="learned dynamics")

        if not draw_axis:
            plt.xticks([])
            plt.yticks([])
            #plt.axis('off')
        #plt.title(target)
        plt.savefig(result_path +target+ '_' + value_name+ save_name+'.png', bbox_inches='tight', pad_inches=0.5)
        plt.close()
        plt.clf()
    
    return

def draw_imgs(e_TF, e_target, v_TF, v_target, t, alpha, beta, omega, theta, gamma, delta, result_path, loss=None, 
        target='synthetic', save_name='', draw_axis=True, show_full_dynamic=False):
    #text = str(alpha)[0:5] + ', ' + str(beta)[0:5] + ', ' + str(omega)[0:5] +\
    #            ', ' + str(theta)[0:5]+ ', ' + str(gamma)[0:5] + ', ' + str(delta)[0:5]
    #if loss is not None:
    #    text += '\nloss: '+str(loss)[0:8]
    text = None
    if loss is not None:
        text = 'Loss: '+str(loss)[0:5]
    my_plot(result_path, t, e_target, e_TF, v_target, v_TF, alpha, beta, omega, theta, gamma, delta,
        text=text, target=target, save_name=save_name, draw_axis=draw_axis, show_full_dynamic=show_full_dynamic)
    my_plot_t(result_path, t, e_TF, e_target, v_target, alpha, beta, omega, theta, gamma, delta, 
        text=None, target=target, save_name=save_name, draw_axis=draw_axis, show_full_dynamic=show_full_dynamic)
    return


def add_noise(arr, gaussian=0.1, drop_prob=0.0):
    arr = arr + gaussian*arr.mean() * np.random.randn(arr.shape[0], arr.shape[1])
    mask = np.random.rand(arr.shape[0], arr.shape[1]) > drop_prob
    arr = arr * mask
    arr = np.clip(arr, 0.0, 1e6)
    return arr


def pad(x, n):
    if len(x.shape) == 1:
        x_out = np.append(x, np.zeros(n - len(x)))
    elif len(x.shape) == 2:
        x_out = np.concatenate((x, np.zeros([x.shape[0], n - x.shape[1]])), axis=1)
    return x_out

def main(args):
    if not args.result_path[-1] =='/':
        args.result_path = args.result_path + '/'
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    figure_path = args.result_path + 'figures/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)  
    weights_all, weights_final_all, velocity_all, data_all = {}, {}, {}, {}
    weights_all['gt'] = np.zeros([args.n_genes, args.n_TFs])
    weights_final_all['gt'] = np.zeros([args.n_genes, args.n_TFs])
    velocity_all['gt'] = np.zeros([args.n_genes, args.n_cells])
    data_all['gt'] = []

    for max_iter in args.max_iter_list:
        key = 'rc_'+str(max_iter)
        weights_all[key] = np.zeros([args.n_genes, args.n_TFs])
        weights_final_all[key] = np.zeros([args.n_genes, args.n_TFs])
        velocity_all[key] = np.zeros([args.n_genes, args.n_cells])
        data_all[key] = []

    for gene_i in range(args.n_genes):   
        print('--------------------------',gene_i,'----------------------------------')
        alpha = np.random.rand()*5 + 1
        beta = alpha + np.random.rand()*5
        omega = 2*np.pi #np.random.rand() * 2*np.pi
        theta = (np.random.rand() - 0.5) * omega
        gamma = np.random.rand()*5 + 3
        delta = 0 #np.random.rand()*5 -2.5
        t_length = 1 - np.random.rand() * 0.5
        t_shift = np.random.rand() * (1-t_length)
        tpoints = np.random.rand(args.n_cells) * t_length + t_shift
        posi_weight_count = np.random.randint(low=int(0.7*args.n_TFs), high=args.n_TFs)
        posi_nega_weight = np.append(np.ones(posi_weight_count), -np.ones(args.n_TFs-posi_weight_count))
        clip_weight = 0.2
        weight_init = posi_nega_weight * ((1-clip_weight) * np.random.rand(args.n_TFs) + clip_weight)
        weight = weight_init.copy()

        WX_t, y_t, tpoints, v_t = my_curve(figure_path, alpha, beta, omega, theta, gamma, delta, 
            to_draw=0, n_cells=args.n_cells, tpoints=tpoints, save_name=str(gene_i))
        #x_t = np.dot(WX_t.reshape(-1,1), np.linalg.pinv(weight.reshape(-1,1)))
        from sklearn.linear_model import Lasso, Ridge
        model = Ridge(alpha=0.1, max_iter=10000, fit_intercept=False)  # positive=True
        model.fit(weight.reshape(1,-1), WX_t.reshape(1,-1))
        x_t = model.coef_

        # delete zero TFs and corresponding weight
        nonzero_TF_id = x_t.mean(0) != 0
        x_t = x_t[:, nonzero_TF_id]
        weight = weight[nonzero_TF_id]

        # deal with negative TFs
        x_mins = x_t.min(0)
        x_t_new = x_t.copy()
        bias = 0
        for i, x_min in enumerate(x_mins):
            if x_min < 0:
                x_t_new[:, i] = x_t_new[:, i] - x_min
                bias = bias + x_min * weight[i]
        tmp = bias/weight[weight>0].sum()
        for i, weight_i in enumerate(weight):
            if weight_i > 0:
                x_t_new[:, i] = x_t_new[:, i] + tmp
                bias = bias - tmp * weight_i

        x_t = x_t_new
        delta = bias

        # add noise
        x = add_noise(x_t)
        y = np.ravel(add_noise(y_t.reshape(-1,1)))

        # norm std 
        norm_term = x.std(0)
        x = x/norm_term
        weight = weight * norm_term
        norm_term_2 = abs(weight).max()
        weight = weight/norm_term_2
        delta = delta/norm_term_2
        alpha = alpha/norm_term_2
        beta = beta/norm_term_2
        y = y/norm_term_2
        weight_final = weight * x.mean(0)
        weight_final = weight_final/abs(weight_final).max()
        WX = np.ravel(np.dot(x, weight.reshape(-1,1))) + delta

        velo_hat = WX - gamma * y  
        target = 'synthetic_gene_'+str(gene_i)
        draw_imgs(e_TF = WX,
            e_target = y,
            v_TF = np.zeros_like(velo_hat), #adata.layers['velo_WX_hat'].copy()
            v_target = velo_hat,
            t = tpoints,
            alpha = alpha,
            beta = beta,
            omega=omega, 
            theta=theta, 
            gamma=gamma, 
            delta=delta, 
            result_path=figure_path,
            target=target,
            save_name = '_gt',
            draw_axis=True,
        )
        print('paras:', alpha, beta, omega, theta, gamma, delta)
        print('weight:', weight)
        print('weight_final:', weight_final)

        if args.to_recover:  
            losses_all = np.zeros([args.n_genes, len(args.max_iter_list)])
            for ii, max_iter in enumerate(args.max_iter_list):
                dm = DynamicsRecovery(x, y.reshape(-1,1), max_iter=max_iter, WX_thres=1) #WX_thres=abs(weight).max()
                dm.fit()
                dm.y = np.ravel(dm.y)
                loss = dm.loss[-1]
                losses_all[gene_i, ii] = loss
                draw_imgs(e_TF = dm.WX,
                    e_target = dm.y,
                    v_TF = np.zeros_like(dm.velo_hat), #adata.layers['velo_WX_hat'].copy()
                    v_target = dm.velo_hat,
                    t = dm.t,
                    alpha = dm.alpha,
                    beta = dm.beta,
                    omega=dm.omega, 
                    theta=dm.theta, 
                    gamma=dm.gamma, 
                    delta=dm.delta,
                    loss=loss,
                    result_path=figure_path,
                    target=target,
                    save_name = '_rc'+str(max_iter),
                    draw_axis=True
                )
                print(dm.alpha, dm.beta, dm.omega, dm.theta, dm.gamma, dm.delta)
                weight_rc = np.ravel(dm.weight)
                velocity_rc = np.ravel(dm.velo_hat)
                print(weight_rc)
                weight_final_rc = weight_rc * x.mean(0)
                weight_final_rc = weight_final_rc/abs(weight_final_rc).max()
                print(weight_final_rc)
                weight_rc = pad(weight_rc, args.n_TFs)
                weight_final_rc = pad(weight_final_rc, args.n_TFs)
                dm.x = pad(dm.x, args.n_TFs)
                weights_all['rc_'+str(max_iter)][gene_i] = np.array(weight_rc) 
                weights_final_all['rc_'+str(max_iter)][gene_i] = np.array(weight_final_rc) 
                velocity_all['rc_'+str(max_iter)][gene_i] = np.array(velocity_rc) 
                data_all['rc_'+str(max_iter)].append([dm.alpha, dm.beta, dm.omega, dm.theta, dm.gamma, dm.delta, dm.t, dm.weight, dm.x, dm.y, dm.WX])
        
        weight = pad(weight, args.n_TFs)
        weight_final = pad(weight_final, args.n_TFs)
        x = pad(x, args.n_TFs)
        weights_all['gt'][gene_i] = np.array(weight)
        weights_final_all['gt'][gene_i] = np.array(weight_final)
        velocity_all['gt'][gene_i] = velo_hat
        data_all['gt'].append([alpha, beta, omega, theta, gamma, delta, tpoints, weight, x, y, WX])
        #print(losses_all[gene_i, :])

    np.save(args.result_path+"weights.npy", weights_all)
    np.save(args.result_path+"weights_final.npy", weights_final_all)
    np.save(args.result_path+"velocity.npy", velocity_all)
    np.save(args.result_path+"data.npy", data_all)
    np.save(args.result_path+"losses_all.npy", losses_all)
    # weights_all = np.load(args.result_path+"weights.npy", allow_pickle=True)[()]   



def evaluate(args): 
    from sklearn.metrics import roc_curve, auc
    # data_all = np.load(args.result_path+"data.npy", allow_pickle=True)[()]  
    losses_all = np.load(args.result_path+"losses_all.npy", allow_pickle=True)
    velocity_all = np.load(args.result_path+"velocity.npy", allow_pickle=True)[()]   
    velocity_gt = velocity_all['gt']
    weights_all = np.load(args.result_path+"weights_final.npy", allow_pickle=True)[()]  # weights_final, weights
    weights_gt = weights_all['gt']
    weights_gt_binary = np.array(weights_gt>0, dtype=int)
    weights_gt_bool = np.array(weights_gt_binary, dtype=bool)
    weights_gt_bool_inv = np.array(1-weights_gt_binary, dtype=bool)
    TPs, TNs, FPs, FNs = np.zeros(len(args.max_iter_list)), np.zeros(len(args.max_iter_list)), \
        np.zeros(len(args.max_iter_list)), np.zeros(len(args.max_iter_list))
    colors = ['red', 'darkorange', 'gold', 'green', 'blue', 'purple'][:len(args.max_iter_list)]
    
    lw = 2
    ## roc of weight: Posi or Nega
    plt.figure(figsize=(10,10))
    for ii, max_iter in enumerate(args.max_iter_list):
        print('---------------------')
        print('n_iter:', max_iter)
        #print('mean loss:', losses_all[:,ii].mean())
        weight_rc = weights_all['rc_'+str(max_iter)]
        TPs[ii] = (weight_rc[weights_gt_bool] > 0).sum()
        FPs[ii] = (weight_rc[weights_gt_bool_inv] > 0).sum()
        TNs[ii] = (weight_rc[weights_gt_bool_inv] < 0).sum()
        FNs[ii] = (weight_rc[weights_gt_bool] < 0).sum()

        TP, FP, TN, FN = TPs[ii], FPs[ii], TNs[ii], FNs[ii]
        print("TP, FP, TN, FN")
        print(TP, FP, TN, FN)
        p = TP/(TP+FP)
        r = TP/(TP+FN)
        F1 = 2*(p*r)/(p+r)
        print('F1:', F1)

        fpr, tpr, thresholds = roc_curve(np.ravel(weights_gt_binary), np.ravel(weight_rc))  
        roc_auc = auc(fpr, tpr)
        print('roc_auc', roc_auc)
        parameters = {'legend.fontsize': 30}
        plt.rcParams.update(parameters)
        plt.plot(fpr, tpr, color=colors[ii],
                lw=lw, label='iter = %d (area = %0.3f)' % (max_iter, roc_auc)) 

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(size=30)
    plt.yticks(size=30)
    plt.xlabel('False Positive Rate', fontdict={'family':'Times New Roman', 'size':35})
    plt.ylabel('True Positive Rate', fontdict={'family':'Times New Roman', 'size':35})
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(args.result_path + 'roc.jpg', bbox_inches='tight', dpi=300) #pad_inches=0.0
    plt.close()

    ## Correlation of weights
    corr_weight_list, corr_velocity_list = [], []
    for ii, max_iter in enumerate(args.max_iter_list):
        plt.figure(figsize=(10,10))
        print('---------------------')
        print('n_iter:', max_iter)
        weight_rc = weights_all['rc_'+str(max_iter)]
        corr = np.corrcoef(weight_rc.reshape(1,-1), weights_gt.reshape(1,-1))[0,1]
        #corr, _ = scipy.stats.spearmanr(weight_rc.reshape(-1), weights_gt.reshape(-1))
        print('Weight corr:', corr)
        _, ax = plt.subplots(figsize=(9, 9)) ###
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        #plt.scatter(weights_gt, weight_rc, color='navy', alpha=0.2, lw=lw, linestyle='--')
        min_value = np.min([weight_rc.min(), weights_gt.min()])
        max_value = np.max([weight_rc.max(), weights_gt.max()])
        value = np.arange(min_value, max_value, 0.1)
        plt.plot(value, value, color='red', linestyle='--', label='y = x (Red Dashed Line)')

        data = np.hstack([weights_gt.reshape(-1,1), weight_rc.reshape(-1,1)])
        df = pd.DataFrame(data, columns=['Ground Truth Velocity', 'Infered Velocity'])
        sns.kdeplot(data=df, x='Ground Truth Velocity', y='Infered Velocity', fill=True, cmap='Blues', levels=5)
        
        plt.xticks(size=30)
        plt.yticks(size=30)
        plt.text(x=0.6, y=0.05, s='Corr: '+str(round(corr, 2)), size=35, transform=ax.transAxes) ###
        plt.xlabel('Ground Truth Weights', fontdict={'family':'Times New Roman', 'size':35})
        plt.ylabel('Infered Weights', fontdict={'family':'Times New Roman', 'size':35})
        plt.savefig(args.result_path + 'Weights_'+str(max_iter)+'.jpg', bbox_inches='tight', dpi=300) #pad_inches=0.0
        plt.close()
        corr_weight_list.append(corr)

    ## Correlation of velocities
    for ii, max_iter in enumerate(args.max_iter_list):
        print('---------------------')
        print('n_iter:', max_iter)
        velocity_rc = velocity_all['rc_'+str(max_iter)]
        corr = np.corrcoef(velocity_rc.reshape(1,-1), velocity_gt.reshape(1,-1))[0,1]
        #corr, _ = scipy.stats.spearmanr(velocity_rc.reshape(-1), velocity_gt.reshape(-1))
        print('Velocity corr:', corr)
        plt.figure(figsize=(10,10))
        _, ax = plt.subplots(figsize=(9, 9))
        #plt.scatter(velocity_gt, velocity_rc, color='navy', alpha=0.03, lw=lw, linestyle='--')
        min_value = np.min([velocity_gt.min(), velocity_rc.min()])
        max_value = np.max([velocity_gt.max(), velocity_rc.max()])
        value = np.arange(min_value, max_value, 0.1)
        plt.plot(value, value, color='red', linestyle='--', label='y = x (Red Dashed Line)')

        data = np.hstack([velocity_gt.reshape(-1,1), velocity_rc.reshape(-1,1)])
        df = pd.DataFrame(data, columns=['Ground Truth Velocity', 'Infered Velocity'])
        sns.kdeplot(data=df, x='Ground Truth Velocity', y='Infered Velocity', fill=True, cmap='Blues', levels=5)
        
        plt.xticks(size=30)
        plt.yticks(size=30)
        plt.axis('scaled')
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        plt.xlim((min_value, max_value))
        plt.ylim((min_value, max_value))
        plt.text(x=0.6, y=0.05, s='Corr: '+str(round(corr, 2)), size=35, transform=ax.transAxes)
        plt.xlabel('Ground Truth Velocity', fontdict={'family':'Times New Roman', 'size':35})
        plt.ylabel('Infered Velocity', fontdict={'family':'Times New Roman', 'size':35})
        plt.savefig(args.result_path + 'Velocity_'+str(max_iter)+'.jpg', bbox_inches='tight', dpi=300) #pad_inches=0.0
        plt.close()
        corr_velocity_list.append(corr)

    ## Correlation of weights under different iterations
    plt.figure()
    x_data = range(len(args.max_iter_list))
    y_data = np.array(corr_weight_list)
    plt.bar(x_data, y_data, color=colors, tick_label=args.max_iter_list)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.ylim((0, 1))
    plt.xlabel('Iterations', fontdict={'family':'Times New Roman', 'size':25})
    plt.ylabel('Correlation of Weights', fontdict={'family':'Times New Roman', 'size':25})
    for x,y in zip(x_data, y_data):
        plt.text(x, y+0.01, round(y, 2) ,fontsize=20, horizontalalignment='center')
    plt.savefig(args.result_path + 'Correlation of Weights.jpg', bbox_inches='tight', dpi=300) #, pad_inches=0.0
    plt.close()

    ## Correlation of Velocity under different iterations
    plt.figure()
    x_data = range(len(args.max_iter_list))
    y_data = np.array(corr_velocity_list)
    plt.bar(x_data, y_data, color=colors, tick_label=args.max_iter_list)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.ylim((0, 1))
    plt.xlabel('Iterations', fontdict={'family':'Times New Roman', 'size':25})
    plt.ylabel('Correlation of Velocity', fontdict={'family':'Times New Roman', 'size':25})
    for x,y in zip(x_data, y_data):
        plt.text(x, y+0.01, round(y, 2) ,fontsize=20, horizontalalignment='center')
    plt.savefig(args.result_path + 'Correlation of Velocity.jpg', bbox_inches='tight', dpi=300) #, pad_inches=0.0
    plt.close()
    return 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--result_path', type=str, default='synthetic_demo/', help='synthetic/') 
    parser.add_argument( '--n_genes', type=int, default=200, help='200')
    parser.add_argument( '--n_TFs', type=int, default=10, help='10')
    parser.add_argument( '--n_cells', type=int, default=1000, help='500')
    parser.add_argument( '--to_recover', type=int, default=1, help='1 or 0')
    parser.add_argument( '--max_iter_list', type=int, default=[5, 10, 20, 30], help='[5, 10, 20, 30]')

    args = parser.parse_args() 
    print('********************************************************************************************************')
    print(args)
    main(args) 
    evaluate(args)