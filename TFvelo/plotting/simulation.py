import numpy as np

import matplotlib.pyplot as pl
from matplotlib import rcParams

from ..core import SplicingDynamics
from ..tools.dynamical_model_utils import get_vars
from .utils import make_dense


def get_dynamics(adata, key="fit", extrapolate=False, sorted=False, t=None):
    alpha, beta, gamma, weight, scaling, t_ = get_vars(adata, key=key)
    if extrapolate:
        u0_ = unspliced(t_, 0, alpha, beta)
        tmax = t_ + omega_inv(u0_ * 1e-4, u0=u0_, alpha=0, beta=beta)
        t = np.concatenate(
            [np.linspace(0, t_, num=500), t_ + np.linspace(0, tmax, num=500)]
        )
    elif t is None or t is True:
        t = adata.obs[f"{key}_t"].values if key == "true" else adata.layers[f"{key}_t"]

    omega, alpha, u0, s0 = vectorize(np.sort(t) if sorted else t, t_, alpha, beta, gamma, weight)
    ut, st = SplicingDynamics(
        alpha=alpha, beta=beta, gamma=gamma, weight=weight, initial_state=[u0, s0]
    ).get_solution(omega)
    return alpha, ut, st


def compute_dynamics(
    adata, basis, key="true"
):
    idx = adata.var_names.get_loc(basis) if isinstance(basis, str) else basis
    key = "fit" if f"{key}_gamma" not in adata.var_keys() else key
    alpha, beta, omega, theta, gamma, delta = get_vars(adata[:, basis], key=key)

    #omega, alpha, u0, s0 = vectorize(np.sort(t) if sort else t, t_, alpha, beta, gamma, weight)
    num = np.clip(int(adata.X.shape[0] / 5), 1000, 2000)
    tpoints = np.linspace(0, 1, num=num)
    WX_t, y_t = SplicingDynamics(alpha=alpha, beta=beta, omega=omega, theta=theta, 
        gamma=gamma).get_solution(tpoints, stacked=False)
    return alpha, WX_t, y_t, tpoints


def show_full_dynamics(
    adata,
    basis,
    key="true",
    use_raw=False,
    linewidth=1,
    linecolor=None,
    show_assignments=None,
    ax=None,
):
    if ax is None:
        ax = pl.gca()
    color = linecolor if linecolor else "grey" if key == "true" else "purple"
    linewidth = 0.5 * linewidth if key == "true" else linewidth
    label = "learned dynamics" if key == "fit" else "true dynamics"
    line = None

    if key != "true":
        _, WX_t, y_t, _ = compute_dynamics(
            adata, basis, key
        )
        if not isinstance(show_assignments, str) or show_assignments != "only":
            ax.scatter(y_t, WX_t, color=color, s=1)
        if show_assignments is not None and show_assignments is not False:
            WX_key, y_key = (
                ("WX", "y")
            )
            WX, y = (
                make_dense(adata[:, basis].layers[WX_key]).flatten(),
                make_dense(adata[:, basis].layers[y_key]).flatten(),
            )
            ax.plot(
                np.array([y, y_t]),
                np.array([WX, WX_t]),
                color="grey",
                linewidth=0.1 * linewidth,
            )

    if not isinstance(show_assignments, str) or show_assignments != "only":
        _, WX_t, y_t, _ = compute_dynamics(
            adata, basis, key
        )
        (line,) = ax.plot(y_t, WX_t, color=color, linewidth=linewidth, label=label)

        idx = adata.var_names.get_loc(basis)
        gamma = adata.var[f"{key}_gamma"][idx]
        #xnew = np.linspace(np.min(y_t), np.max(y_t))
        #ynew = np.linspace(np.min(WX_t), np.max(WX_t))#gamma / weight * (xnew - np.min(xnew)) + np.min(ut)
        #ax.plot(xnew, ynew, color=color, linestyle="--", linewidth=linewidth)
    return line, label


def simulation(
    adata,
    var_names="all",
    legend_loc="upper right",
    legend_fontsize=20,
    linewidth=None,
    dpi=None,
    xkey="true_t",
    ykey=None,
    colors=None,
    **kwargs,
):
    from ..tools.utils import make_dense
    from .scatter import scatter

    if ykey is None:
        ykey = ["unspliced", "spliced", "alpha"]
    if colors is None:
        colors = ["darkblue", "darkgreen", "grey"]
    var_names = (
        adata.var_names
        if isinstance(var_names, str) and var_names == "all"
        else [name for name in var_names if name in adata.var_names]
    )

    figsize = rcParams["figure.figsize"]
    ncols = len(var_names)
    for i, gs in enumerate(
        pl.GridSpec(
            1, ncols, pl.figure(None, (figsize[0] * ncols, figsize[1]), dpi=dpi)
        )
    ):
        idx = adata.var_names.get_loc(var_names[i])
        alpha, ut, st = compute_dynamics(adata, idx)
        t = (
            adata.obs[xkey]
            if xkey in adata.obs.keys()
            else make_dense(adata.layers["fit_t"][:, idx])
        )
        idx_sorted = np.argsort(t)
        t = t[idx_sorted]

        ax = pl.subplot(gs)
        _kwargs = {"alpha": 0.3, "title": "", "xlabel": "time", "ylabel": "counts"}
        _kwargs.update(kwargs)
        linewidth = 1 if linewidth is None else linewidth

        ykey = [ykey] if isinstance(ykey, str) else ykey
        for j, key in enumerate(ykey):
            if key in adata.layers:
                y = make_dense(adata.layers[key][:, idx])[idx_sorted]
                ax = scatter(x=t, y=y, color=colors[j], ax=ax, show=False, **_kwargs)

            if key == "unspliced":
                ax.plot(t, ut, label="unspliced", color=colors[j], linewidth=linewidth)
            elif key == "spliced":
                ax.plot(t, st, label="spliced", color=colors[j], linewidth=linewidth)
            elif key == "alpha":
                largs = dict(linewidth=linewidth, linestyle="--")
                ax.plot(t, alpha, label="alpha", color=colors[j], **largs)

        pl.xlim(0)
        pl.ylim(0)
        if legend_loc != "none" and i == ncols - 1:
            pl.legend(loc=legend_loc, fontsize=legend_fontsize)
