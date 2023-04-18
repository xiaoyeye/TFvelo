from . import velocity
from .utils import make_unique_list
from .. import logging as logg
from .dynamical_model_utils import get_divergence
from .dynamical_model import DynamicsRecovery, write_pars, read_pars, recover_dynamics
import numpy as np
from scanpy.tools import louvain
from scipy.stats.distributions import laplace
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def get_orth_dists(S, U, beta, std_s, std_u, scaling):
    S_real = np.array((S + (beta * U)) / (1 + beta ** 2))
    orth_dists = np.array(S - S_real) / std_s
    orth_distu = np.array(U - (beta * S_real)) / std_u * scaling
    orth_distx = orth_distu ** 2 + orth_dists ** 2
    return orth_distx


def get_orth_fit(U, S, weights):
    U, S = U * weights.astype(int), S * weights.astype(int)
    a, b = np.sum(S * U, axis=0), np.sum(U ** 2 - S ** 2, axis=0)
    orth_beta = (b + ((b ** 2 + 4 * a ** 2) ** .5)) / (2 * a)
    return orth_beta


def appendInt(num):
    if num > 9:
        secondToLastDigit = str(num)[-2]
        if secondToLastDigit == '1':
            return 'th'
    lastDigit = num % 10
    if lastDigit == 1:
        return 'st'
    elif lastDigit == 2:
        return 'nd'
    elif lastDigit == 3:
        return 'rd'
    else:
        return 'th'


def expfun(x, a, b):
    return a * (-np.exp(-b * x) + 1)


def f(a, b):
    def fun(x):
        return a * (-np.exp(-b * x) + 1)

    return fun


def inverse(x, a, b):
    return -np.log((-x / a) + 1) / b


def get_curve_fit(X, Y, sigma):
    kwargs = dict(bounds=([0, 0], [np.inf, 1.5]), sigma=sigma, method='trf', maxfev=4000)
    fun, fup = None, None
    try:
        pars, cov = curve_fit(f=expfun, xdata=X, ydata=Y, **kwargs)
        up_dist = np.sum((Y - expfun(X, *pars)) ** 2)
        fun, fup = f(*pars), True
        x = np.linspace(0, np.max(X), 100)
        y = expfun(x, *pars)
    except RuntimeError:
        up_dist = np.inf
    try:
        down_pars, cov = curve_fit(f=expfun, xdata=Y, ydata=X, **kwargs)
        down_dist = np.sum((X - expfun(Y, *down_pars)) ** 2)
        if down_dist < up_dist:
            fun, fup = f(*down_pars), False
            y = np.linspace(0, np.max(Y), 100)
            x = expfun(y, *down_pars)
    except RuntimeError:
        down_dist = np.inf
    return fun, fup#, x, y  # np.min([down_dist, up_dist])*2#, y, x


def plot(S, U, fit_weights, weights, i, x2, y2, sigma=None):
    fig, ax = plt.subplots()
    ax.scatter(S[:, i], U[:, i], c="black", s=10)
    U, S = U * fit_weights.astype(int), S.copy() * fit_weights.astype(int)
    ax.scatter(S[:, i], U[:, i], c="orange", s=2)
    U, S = U * weights.astype(int), S * weights.astype(int)
    ax.scatter(S[:, i], U[:, i], c="yellow", s=4)
    ax.plot(x2, y2, "r")
    #if sigma != None:
    #    print(sigma[:,i])
    #ax.scatter(S[:, i], U[:, i], c=sigma[:,i], s=5)
    plt.show()


def differential_kinetic_test(adata, var_names=None, cluster_name="clusters", key="fit", fit_type="line",
                              test_with_refit=False,
                              min_cluster_cells=15, use_raw=False, refit_main_dynamics=True, refit_at_end=True,
                              **kwargs):
    """
    Parameters
    ----------
    adata
    var_names: list of "str"
        Names of variables/genes to use for the fitting.
        If none given, will use all genes having dynamical model fit.
    cluster_name: "str" (default: "clusters")
        Name of the adata.obs column by which to cluster the cells.
    key: "str" (default "fit")
        key under which the dynamical model fit parameters are saved
    fit_type: str "line" or "curve"
        line or curve fit to test if additional clusters also differential (in (2))
    test_with_refit: bool (default False)
        whether to fit the dyn-model without the cluster before testing for differential kinetic of that
        cluster (in (1))
    min_cluster_cells: int (default 15)
        Minimal number of cells needed in cluster.
    use_raw: bool (default False)
        Whether to use raw unspliced and spliced counts or first order moments.
    refit_main_dynamics: bool (default: True)
        Whether the kinetic fit should be recomputed to exclude the competing kinetic.
        with **kwargs optional additional parameters for the recover_dynamics function
    refit_at_end: bool (default: False)
        Whether the main kinetic fit should be recomputed at all iteration or only at the end when all competing
         kinetics have been identified. Only relevant if nk>1.
    Returns
    -------
    """
    # check if params are ok or set if needed
    if var_names is not None:
        var_names = np.array(
            [name for name in make_unique_list(var_names, allow_array=True) if name in adata.var_names])
        if len(var_names) == 0:
            raise ValueError('Variable name not found in var keys.')
    else:
        var_names = adata.var_names
    var_names = adata[:, var_names].var_names[~np.isnan(adata[:, var_names].var[key + "_alpha"])]
    if len(var_names) == 0:
        logg.error(
            "No dynamical model parameters found for key \"" + key + "\" in adata. Please call scv.tl.recover_dynamics "
                                                                     "before testing for differential kinetics.")
    if cluster_name not in adata.obs.keys():
        logg.error("Cluster name: " + cluster_name + " not in adata.obs.keys() \nUsing louvain clusters instead.")
        if "louvain" not in adata.obs.keys():
            louvain(adata)
        cluster_name = "louvain"
    if min_cluster_cells < 15:
        logg.warn("Normal approximation for the statistical test works only if there are enough points in the tested "
                  "cluster. Setting \"min_cluster_cells\" parameter to " + min_cluster_cells + "might cause inaccurate "
                                                                                               "results.")
    #########
    # fetch #
    #########
    idx = np.where([i in var_names for i in adata.var_names])[0]
    var_names = adata.var_names[idx]  # sort var_names according to adata order
    spliced, unspliced = "spliced" if use_raw else "Ms", "unspliced" if use_raw else "Mu"
    S, U = adata.layers[spliced][:, idx], adata.layers[unspliced][:, idx]
    scaling = np.array(adata.var.loc[var_names, :][key + "_scaling"])
    U = U / scaling[None, :]
    std_s, std_u, varx = np.array(adata.var.loc[var_names, :][key + "_std_s"]), np.array(
        adata.var.loc[var_names, :][key + "_std_u"]), np.array(adata.var.loc[var_names, :][key + "_variance"])
    weights = ((S > 0) & (U > 0))
    weights &= ((S > np.nanmax(S, axis=0) / 5) | (U > np.nanmax(U, axis=0) / 5))
    # sub_idx because get_divergence returns obs * n_vars with fit_alpha, thus only subset amongst the vars with dynfit
    sub_idx = np.where([i in var_names for i in adata[:, ~np.isnan(adata.var[key + "_alpha"].values)].var_names])[0]
    dyn_distx = get_divergence(adata, mode='distx', use_connectivities=False, use_raw=use_raw)[:, sub_idx]
    clusters = adata.obs[cluster_name]
    clus = list(clusters.cat.categories)
    ###########
    # testing #
    ###########
    main_kinetic = np.ones(weights.shape).astype(bool)  # stores whether the cells are part of a competing kinetic
    all_diff = np.array(np.repeat("", adata.shape[1]), dtype=object)  # all competing kinetics

    LR, i, where, n = [], 0, [True] * len(var_names), len(var_names)
    while np.sum(where) > 0:
        # (1) get cluster that fits worst to main kinetic
        best_clus, best_beta, best_pval = np.array(np.repeat("", n), dtype=object), np.zeros(n), np.ones(n)
        for c in clus:
            if test_with_refit:
                bdata = adata.copy()
                bdata.var["fit_diff_kinetics"] = (cluster_name + ":" + c) if i == 0 else [cluster_name + ":" + i[1:] + str("," + c) for i in all_diff]
                #print(bdata.var["fit_diff_kinetics"])
                recover_dynamics(bdata, var_names[where],
                                 use_raw=use_raw, add_key=key, load_pars=False, main=True, **kwargs)
                # var_names[where]
                dyn_distx = get_divergence(bdata, mode='distx', use_connectivities=False, use_raw=use_raw)[:, sub_idx]
            weights_cluster = np.array([(clusters == c).tolist()] * n).T & weights & main_kinetic
            orth_beta = get_orth_fit(U, S, weights=weights_cluster)  # include inner vals
            pval, lr = orth_diff_kinetics(dyn_distx, weights_cluster, S, U, orth_beta, varx, std_s, std_u, scaling)
            if fit_type == "line":
                LR.extend(lr.tolist())
            test = (pval < best_pval) & (np.sum(weights_cluster, axis=0) > min_cluster_cells)
            best_pval[test], best_clus[test], best_beta[test] = pval[test], c, orth_beta[test]
        where = (best_pval < 0.05)
        logg.info(str(np.sum(where)) + " genes with " + str(i + 1) + " competing kinetic" + ("s." if i > 0 else "."))
        if np.sum(where) == 0:
            break
        main_kinetic[:, ~where], best_clus[~where] = False, ""  # no need to look again at genes with only main kinetic
        if test_with_refit:
            bdata.var["fit_diff_kinetics"] = [cluster_name + ":" + i if i != "" else "" for i in best_clus]
            recover_dynamics(bdata, var_names[where],
                             use_raw=use_raw, add_key=key, load_pars=False, main=True, **kwargs)
            dyn_distx = get_divergence(bdata, mode='distx', use_connectivities=False, use_raw=use_raw)[:, sub_idx]
        prev = (np.array([(clusters == j) for j in best_clus]).T & weights)
        # to save test parameters
        mn = main_kinetic[:, where]
        # (2) check if other cluster fit better to competing kinetic than to main one
        if fit_type == "line":
            diff_kinetics = best_clus[where]
            beta, best_pval = np.zeros(n), np.zeros(n)
            beta[:], best_pval[:] = np.nan, np.nan
            for c in clus:
                weights_cluster = (np.array(
                    [(clusters == c).tolist()] * len(var_names)).T & ~prev & weights & main_kinetic)[:,
                                  where]
                pval, lr = orth_diff_kinetics(dyn_distx[:, where], weights_cluster, S[:, where], U[:, where],
                                              best_beta[where], varx[where], std_s[where], std_u[where], scaling[where])
                diff_kinetics[pval < 0.05] += "," + c
            # get final pval
            best_clus[where] = diff_kinetics
            final_weights = np.array([[i in diff.split(",") for i in clusters] for diff in best_clus[where]]).T \
                            & (weights & main_kinetic)[:, where]
            mn[final_weights] = False
            pval_kinetics, lr = orth_diff_kinetics(dyn_distx[:, where], final_weights, S[:, where], U[:, where],
                                                   best_beta[where],
                                                   varx[where], std_s[where], std_u[where], scaling[where])
        elif fit_type == "curve":
            fit, m = where.copy(), 1
            while np.sum(fit) > 0:
                best_lr, b_pval, diff_kinetics = np.ones(np.sum(fit)) * -np.inf, np.ones(np.sum(fit)), best_clus[fit]
                for c in clus:
                    weights_cluster = ((np.array(
                        [(clusters == c).tolist()] * len(var_names)).T) & ~prev & weights & main_kinetic)[:,
                                      fit]
                    sigma = (weights_cluster.astype(int) * np.sum(weights_cluster, axis=0)) * m + (
                            prev[:, fit].astype(int) * np.sum(prev[:, fit], axis=0))  # m scales number of prev kinetics
                    pval, lr = curve_diff_kinetics(dyn_distx[:, fit], weights_cluster, (prev[:, fit] | weights_cluster),
                                                   sigma, (S / std_s)[:, fit], (U / std_u * scaling)[:, fit], varx[fit])
                    LR.extend(lr)
                    diff_kinetics[pval < b_pval] = c  # ,
                    b_pval[pval < b_pval] = pval[pval < b_pval]
                c = best_clus[fit]
                c[b_pval < 0.05] += "," + diff_kinetics[b_pval < 0.05]
                best_clus[fit], fit[fit] = c, b_pval < 0.05
                prev = (np.array([[i in j for i in clusters] for j in best_clus]).T & weights)
                m += 1
                # logg.info(str(np.sum(fit)) + " genes with " + str(m) + " clusters in the competing kinetic.")
            prev = (np.array([[i in j for i in clusters] for j in best_clus]).T & weights)
            pval_kinetics, lr = curve_diff_kinetics(dyn_distx[:, where], prev[:, where], prev[:, where],
                                                    prev[:, where], (S / std_s)[:, where],
                                                    (U / std_u * scaling)[:, where], varx[where])
            mn[prev[:, where]] = False
        main_kinetic[:, where] = mn
        nidx = idx[where]  # index of multi-kinetic genes
        best_pval = np.ones(adata.shape[1]) * np.nan
        best_pval[nidx], all_diff[nidx] = pval_kinetics, all_diff[nidx] + "," + best_clus[nidx]
        local_diff = np.array([cluster_name + ":" + best_clus[i] if i in nidx else "" for i in range(adata.shape[1])])

        # save diff kinetics in var
        main_diff = [cluster_name + ":" + d[1:] if d != "" else "" for d in all_diff]
        write_pars(adata, [main_diff], pars_names=["diff_kinetics"], add_key=key)
        write_pars(adata, [best_pval, local_diff],
                   pars_names=["pval_kinetics", "kinetics"], add_key=key + str(i + 1))
        if refit_main_dynamics:
            logg.warn("Refitting main dynamic for genes with competing kinetics.")
            # refit multi-kinetic genes where competing kinetic has been found at this iteration
            recover_dynamics(adata, adata[:, adata.var[key + str(i + 1) + "_kinetics"] != ""].var_names,
                             use_raw=use_raw, add_key=key, load_pars=False, main=True, **kwargs)
        i += 1
    if (not refit_main_dynamics) & refit_at_end:
        logg.warn("Refitting main dynamic for genes with competing kinetics.")
        recover_dynamics(adata, adata[:, adata.var[key + "_diff_kinetics"] != ""].var_names, use_raw=use_raw,
                         add_key=key, load_pars=False, main=True, **kwargs)
    return LR


def orth_diff_kinetics(distx, weights, S, U, orth_beta, varx, std_s, std_u, scaling):
    U, S = U * weights.astype(int), S * weights.astype(int)
    orth_distx = get_orth_dists(S, U, orth_beta, std_s, std_u, scaling)
    denom = varx * np.sqrt(8 * np.sum(weights, axis=0))  # sum of weights is all cells taken into account
    distx = distx * 0.5  # todo double check!
    lr = (np.sum(distx * weights.astype(int), axis=0) - np.sum(orth_distx * weights.astype(int), axis=0)) / denom
    return np.array([laplace.sf(x) for x in lr]), lr


def curve_diff_kinetics(distx, testing_weights, fitting_weights, sigma, S, U, varx):
    Ufit, Sfit = U.copy() * fitting_weights.astype(int), S.copy() * fitting_weights.astype(int)
    Ut, St = U.copy() * testing_weights.astype(int), S.copy() * testing_weights.astype(int)
    curve_dist = np.ones((U.shape[1])) * np.inf
    distx = np.sum(distx * testing_weights.astype(int), axis=0) * 0.5
    denom = varx * np.sqrt(8 * np.sum(testing_weights, axis=0))
    # for each gene, get curve fit
    for i in np.where(np.sum(testing_weights, axis=0) > 4)[0]:
        # get curve fit
        sig = (sigma[:, i] / np.sum(sigma[:, i])) * np.sum(fitting_weights[:, i])

        fun, fup = get_curve_fit(Sfit[:, i][sig > 0], Ufit[:, i][sig > 0], sigma=sig[sig > 0])
        if fup is None:  # curve fit did not converge
            curve_dist[i] = np.inf
        elif fup:
            curve_dist[i] = np.sum((Ut[:, i] - fun(St[:, i])) ** 2)
        else:
            curve_dist[i] = np.sum((St[:, i] - fun(Ut[:, i])) ** 2)
        # lr = (distx[i] - curve_dist[i]) / denom[i]
        # if laplace.sf(lr) < 0.05:
        #    print(lr)
        #    plt.hist(sig[sig > 0], bins=100)
        #    plot(S, U, fitting_weights, testing_weights, i, x, y)
    lr = (distx - curve_dist) / denom
    return np.array([laplace.sf(x) for x in lr]), lr


def recover_second_kinetic_velocity(adata, use_raw=False):  # just for rename not to break everything right away
    recover_competing_kinetic_velocity(adata, use_raw)


def recover_competing_kinetic_velocity(adata, use_raw=False, nk=None):
    logg.info("Recovering main velocity.")
    velocity(adata, mode='dynamical', diff_kinetics=True, use_raw=True)  # velocity=0 for clusters in fit_diff_kinetics
    nk = np.max([(int(i.split("_")[0][3:]) if len(i.split("_")[0][3:]) else 0) if i.startswith("fit") else 1
                 for i in adata.var.columns.tolist()]) if nk is None else nk  # get highest x for fitx_ in var

    for i in range(nk):
        logg.info("Recovering velocity of " + str(i + 1) + appendInt(i + 1) + " competing kinetic.")

        subdata = adata.copy()
        competing_kinetic = 'fit' + str(i + 1) + "_kinetics"
        subdata.var['fit_diff_kinetics'] = ""
        var_names = subdata[:, adata.var[competing_kinetic] != ""].var_names

        alpha, beta, gamma, t_, scaling, std_u, std_s, likelihood, u0, s0, pval, steady_u, steady_s, varx = read_pars(
            subdata)
        idx, L, P = [], [], []
        T = subdata.layers['fit_t'] if 'fit_t' in subdata.layers.keys() else np.zeros(subdata.shape) * np.nan
        Tau = subdata.layers['fit_tau'] if 'fit_tau' in subdata.layers.keys() else np.zeros(subdata.shape) * np.nan
        Tau_ = subdata.layers['fit_tau_'] if 'fit_tau_' in subdata.layers.keys() else np.zeros(subdata.shape) * np.nan
        diff_kinetics = ["" for x in range(subdata.n_vars)]

        progress = logg.ProgressReporter(len(var_names))
        for gene in var_names:
            progress.update()
            ix = np.where(subdata.var_names == gene)[0][0]
            diff_k = adata.var[competing_kinetic][ix].split(":")
            groupby = diff_k[0]
            clusters = subdata.obs[groupby if isinstance(groupby, str) else "clusters"]
            groups = clusters.cat.categories
            v = diff_k[1].split(",")
            m = [i in v for i in clusters]
            # check if cluster in steady-state
            s, u = adata.layers["spliced" if use_raw else "Ms"][m, ix], \
                   subdata.layers["unspliced" if use_raw else "Mu"][
                       m, ix]
            if spearmanr(s, u).pvalue < 0.01:  # 0.01:
                dm = DynamicsRecovery(subdata, gene, use_raw=use_raw, load_pars=False, model=m)
                if dm.recoverable:
                    dm.fit()
                    idx.append(ix)
                    T[:, ix], Tau[:, ix], Tau_[:, ix] = dm.t, dm.tau, dm.tau_
                    alpha[ix], beta[ix], gamma[ix], t_[ix], scaling[ix] = dm.pars[:, -1]
                    u0[ix], s0[ix], pval[ix], steady_u[ix], steady_s[
                        ix] = dm.u0, dm.s0, dm.pval_steady, dm.steady_u, dm.steady_s
                    beta[ix] /= scaling[ix]
                    steady_u[ix] *= scaling[ix]
                    std_u[ix], std_s[ix], likelihood[ix], varx[ix] = dm.std_u, dm.std_s, dm.likelihood, dm.varx
                    L.append(dm.loss)
                diff_kinetics[ix] = groupby + ":" + ",".join(groups[[i not in diff_k for i in groups]])
        progress.finish()

        write_pars(subdata,
                   [alpha, beta, gamma, t_, scaling, std_u, std_s, likelihood, u0, s0, pval, steady_u, steady_s, varx])
        write_pars(subdata, [diff_kinetics], pars_names=["diff_kinetics"])

        # write_pars(adata,
        #           [alpha, beta, gamma, t_, scaling, std_u, std_s, likelihood, u0, s0, pval, steady_u, steady_s, varx])

        subdata.layers['fit_t'], subdata.layers['fit_tau'], subdata.layers['fit_tau_'] = T, Tau, Tau_
        # now recover velocities
        velocity(subdata, mode='dynamical', diff_kinetics=True, use_raw=use_raw)
        # merge velocities
        where = subdata.var['fit_diff_kinetics'] != ""
        velo, velo_u = adata.layers["velocity"], adata.layers["velocity_u"]
        velo[:, where], velo_u[:, where] = subdata[:, where].layers["velocity"] + velo[:, where], \
                                           subdata[:, where].layers[
                                               "velocity_u"] + velo_u[:, where]
        adata.layers["velocity"], adata.layers["velocity_u"] = velo, velo_u