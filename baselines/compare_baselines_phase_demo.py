# use the dynamo env
import pandas as pd
import unitvelo as utv
import scvelo as scv
import scanpy as sc
import anndata as ad
import numpy as np
import scipy
import TFvelo as TFv
import dynamo as dyn
import os
import matplotlib.pyplot as plt
import seaborn as sns
'''
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlepad'] = 20
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
'''
utv_config = utv.config.Configuration()
utv_config.R2_ADJUST = True
utv_config.IROOT = None
utv_config.FIT_OPTION = '1'
utv_config.AGENES_R2 = 1

def upper_var(adata):
    var_names = list(adata.var_names)
    GENES = []
    for g in var_names:
        GENES.append(g.upper())
    adata.var_names = GENES
    return adata


def get_adatas(args, baselines):
    adata_TFv = ad.read_h5ad('TFvelo_'+args.dataset_name+'/rc.h5ad')
    adata_TFv_velocity = ad.read_h5ad('TFvelo_'+args.dataset_name+'/TFvelo.h5ad')
    adata_TFv.obs['velocity_pseudotime'] = adata_TFv_velocity.obs['velocity_pseudotime']
    losses = adata_TFv.varm['loss'].copy()
    losses[np.isnan(losses)] = 1e6
    adata_TFv.var['min_loss'] = losses.min(1)
    thres_loss = np.percentile(adata_TFv.var['min_loss'], 50) 
    adata_TFv_copy = adata_TFv[:, adata_TFv.var['min_loss'] <= thres_loss]
    adata_TFv_copy.uns['clusters_colors'] = adata_TFv.uns['clusters_colors']
    adata_TFv = adata_TFv_copy
    n_cells = adata_TFv.shape[0]
    expanded_scaling_y = np.expand_dims(np.array(adata_TFv.var['fit_scaling_y']),0).repeat(n_cells, axis=0)
    adata_TFv.layers['velocity'] = adata_TFv.layers['velo_hat'] / expanded_scaling_y  

    n_colors = len(adata_TFv.obs['clusters'].cat.categories)
    if 'clusters_colors' in adata_TFv.uns:
        adata_TFv.uns['clusters_colors'] = adata_TFv.uns['clusters_colors'][:n_colors]
    print(adata_TFv)
    adata_baselines = {}

    for b in baselines:
        print(b)
        adata_b = ad.read_h5ad('TFvelo_'+args.dataset_name+'/' + b + '.h5ad') 
        adata_b.uns['clusters_colors'] = adata_TFv.uns['clusters_colors'].copy()
        if b == 'scvelo':
            if args.baseline_velo_gene:
                subvar = adata_b.var.loc[adata_b.var['velocity_genes'] == True]
                adata_b = adata_b[:, subvar.index]
        if b == 'unitvelo':
            adata_b.uns['temp'] = 'figures/unitvelo' 
            if args.baseline_velo_gene:
                subvar = adata_b.var.loc[adata_b.var['velocity_genes'] == True]
                adata_b = adata_b[:, subvar.index]
            if not os.path.exists(adata_b.uns['temp']):
                os.makedirs(adata_b.uns['temp'])
        elif b == 'dynamo':
            subvar = adata_b.var.loc[adata_b.var['gamma'] != 'None']
            adata_b = adata_b[:, subvar.index]
            adata_b.var['gamma'] = np.array(adata_b.var['gamma'], dtype=float)
            adata_b.var['gamma_b'] = np.array(adata_b.var['gamma_b'], dtype=float)
            adata_b.uns['clusters_colors'] = adata_TFv.uns['clusters_colors'][adata_TFv.obs['clusters'].cat.categories.argsort()]
        
        adata_b = upper_var(adata_b)
        adata_baselines[b] = adata_b
        print(adata_baselines[b])
    return adata_TFv, adata_baselines


def get_gene(args, adata_TFv, adata_baselines):
    genes = []
    for g in adata_TFv.var_names:
        flag = True
        for b in adata_baselines.keys():
            if (not g in adata_baselines[b].var_names):
                flag = False
                break
        if flag:
            genes.append(g)
    print(len(genes))
    print(genes)
    return genes

def get_gene_only4TFvelo(args, adata_TFv, adata_baselines):
    genes = []
    for g in adata_TFv.var_names:
        flag = False
        for b in adata_baselines.keys():
            adata_b = adata_baselines[b]
            if b in ['scvelo', 'unitvelo']:
                velo_genes = adata_b.var_names[adata_b.var['velocity_genes'] == True]
                if (not g in velo_genes) and (g in adata_b.var_names) :
                    flag = True
                    break
        if flag:
            genes.append(g)
    print(len(genes))
    print(genes)
    return genes


def get_phase_cluster_g(args, adata, g, TFvelo=True):
    labels = []
    n_labels = len(adata.obs['clusters'].cat.categories)
    for c in adata.obs['clusters']:
        labels.append(list(adata.obs['clusters'].cat.categories).index(c))
    labels = np.array(labels)

    if TFvelo:
        data1 = adata[:, g].layers['WX']
        data2 = adata[:, g].layers['M_total']
    else:
        data1 = adata[:, g].layers['Mu']
        data2 = adata[:, g].layers['Ms']
    data1 = data1/data1.std()
    data2 = data2/data2.std()
    data = np.hstack([data1, data2])

    cluster_centers = []
    for cluster_id in range(n_labels):
        points_in_cluster = data[labels == cluster_id]
        cluster_center = np.mean(points_in_cluster, axis=0)
        cluster_centers.append(cluster_center)

    intra_class_distance = []
    for cluster_id in range(n_labels):
        points_in_cluster = data[labels == cluster_id]
        for point in points_in_cluster:
            value = np.linalg.norm(point - cluster_centers[cluster_id])
            intra_class_distance.append(value)
    intra_class_distance = np.array(intra_class_distance).mean()
    #print("mean of intra_class_distance:", intra_class_distance) 

    inter_class_distance = []
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            value = np.linalg.norm(cluster_centers[i] - cluster_centers[j])          
            inter_class_distance.append(value)
    inter_class_distance = np.array(inter_class_distance).mean()
    #print("mean of inter_class_distance:", inter_class_distance) 
    return intra_class_distance, inter_class_distance

def get_phase_err_TFv(args, adata, g):
    if args.to_show:
        TFv.pl.velocity(adata, g, ncols=4, add_outline=True, layers='na', dpi=300, fontsize=15, save='phase_TFvelo_'+g+'.png') #layers='all'
    alpha = adata[:, g].var['fit_alpha'][0]
    beta = adata[:, g].var['fit_beta'][0] 
    omega = adata[:, g].var['fit_omega'][0] 
    theta = adata[:, g].var['fit_theta'][0] 
    gamma = adata[:, g].var['fit_gamma'][0] 
    delta = adata[:, g].var['fit_delta'][0] 
    t = np.array(adata[:, g].layers['fit_t_raw'][:, 0])
    WX = np.array(adata[:, g].layers['WX'][:, 0])
    y = np.array(adata[:, g].layers['y'][:, 0])
    y_t = np.array(adata[:, g].layers['y_t'][:, 0])
    phi = np.arctan(omega/gamma)
    WX_t = alpha*np.sqrt(4*np.pi*np.pi+gamma*gamma) * np.sin(omega*t+theta+phi) + beta*gamma 
    mse = (((WX_t-WX)/WX.std())**2 + ((y_t-y)/y.std())**2).mean()
    return mse


def get_phase_err_scv(args, adata, g):
    if args.to_show:
        scv.pl.velocity(adata, g, ncols=4, add_outline=True, layers='na', dpi=300, fontsize=15, save='phase_scvelo_'+g+'.png') #layers='all'
    s = np.array(adata[:, g].layers['Ms'][:, 0])
    u = np.array(adata[:, g].layers['Mu'][:, 0])
    from scvelo.plotting.simulation import compute_dynamics
    _, ut, st = compute_dynamics(
                adata, g, key="fit", extrapolate=False, sort=False
            )
    ut, st = np.array(ut), np.array(st)
    mse = (((ut-u)/u.std())**2 + ((st-s)/s.std())**2).mean()
    return mse

def get_phase_err_utv(args, adata, g):
    if args.to_show:
        label = adata.uns['label']
        utv.pl.plot_range(g, adata, utv_config, show_legend=False, show_ax=True, palette=list(adata.uns[label+'_colors']), save_fig=True)
    gdata = adata[:, g]
    s = np.array(gdata.layers['Ms'][:, 0])
    u = np.array(gdata.layers['Mu'][:, 0])
    t = np.array(gdata.layers['fit_t'][:, 0])
    from unitvelo.pl import rbf, rbf_u
    st = np.squeeze(rbf(t, gdata.var.fit_h.values, gdata.var.fit_a.values, gdata.var.fit_t.values, gdata.var.fit_offset.values))
    ut = np.squeeze(rbf_u(t, gdata.var.fit_h.values, gdata.var.fit_a.values, gdata.var.fit_t.values, gdata.var.fit_offset.values, gdata.var.fit_beta.values, gdata.var.fit_gamma.values, gdata.var.fit_intercept.values))
    mse = (((ut-u)/u.std())**2 + ((st-s)/s.std())**2).mean()
    return mse



def show_phase_dynamo(args, adata, g):
    if g in adata.var_names:
        save_kwargs = {
                        "path": 'figures/',
                        "prefix": 'dyn_phase_'+g,
                        "dpi": 300,
                        "ext": 'png',
                        "transparent": True,
                        "close": True,
                        "verbose": True
                    }

        dyn.pl.phase_portraits(adata, genes=g, color='clusters',  
                            discrete_continous_div_color_key=[list(adata.uns['clusters_colors']), list(adata.uns['clusters_colors']), list(adata.uns['clusters_colors'])], 
                            ncols=6,  pointsize=5, save_show_or_return='save', save_kwargs=save_kwargs,
                            show_quiver=True, quiver_size=5) # no_vel_u=False
    return

def get_phase_err(args, adata_TFv, adata_baselines, genes):
    losses_all = {}
    losses_all['TFvelo'] = []
    for b in adata_baselines.keys():
        if b in ['scvelo', 'unitvelo']:
            losses_all[b] = []

    for g in genes:
        err_TFv = get_phase_err_TFv(args, adata_TFv, g)
        losses_all['TFvelo'].append(err_TFv)

        for b in adata_baselines.keys():
            adata_b = adata_baselines[b]
            if b == 'scvelo':
                err_scv = get_phase_err_scv(args, adata_b, g)
                losses_all[b].append(err_scv)
            elif b == 'unitvelo':
                err_utv = get_phase_err_utv(args, adata_b, g)
                losses_all[b].append(err_utv)
            elif b == 'dynamo':
                if args.to_show:
                    show_phase_dynamo(args, adata_b, g)

    losses_all_df = pd.DataFrame(losses_all)
    losses_all_df.to_csv('figures/Losses_'+args.dataset_name+'.txt', sep='\t', index=False)
    return


def get_phase_cluster(args, adata_TFv, adata_baselines, genes):
    intra_class_distance_all, inter_class_distance_all = {}, {}
    intra_class_distance_all['TFvelo'] = []
    intra_class_distance_all['Un/spliced'] = []
    inter_class_distance_all['TFvelo'] = []
    inter_class_distance_all['Un/spliced'] = []

    for g in genes:
        intra_class_distance_TFv, inter_class_distance_TFv = get_phase_cluster_g(args, adata_TFv, g, TFvelo=True)
        intra_class_distance_scv, inter_class_distance_scv = get_phase_cluster_g(args, adata_baselines['scvelo'], g, TFvelo=False)
        intra_class_distance_all['TFvelo'].append(intra_class_distance_TFv)
        intra_class_distance_all['Un/spliced'].append(intra_class_distance_scv)
        inter_class_distance_all['TFvelo'].append(inter_class_distance_TFv)
        inter_class_distance_all['Un/spliced'].append(inter_class_distance_scv)

    intra_class_distance_all_df = pd.DataFrame(intra_class_distance_all)
    intra_class_distance_all_df.to_csv('figures/intra_class_distance_'+args.dataset_name+'.txt', sep='\t', index=False)
    inter_class_distance_all_df = pd.DataFrame(inter_class_distance_all)
    inter_class_distance_all_df.to_csv('figures/inter_class_distance_'+args.dataset_name+'.txt', sep='\t', index=False)
    return

def draw_plot(df, save_name, fig_type='box_plot', y_min=None, y_max=None, t_test=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    my_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    sns.set_palette(my_palette[:df.shape[-1]])
    if fig_type=='boxplot':
        ax = sns.boxplot(data=df, showfliers = False)
    elif fig_type=='barplot':
        ax = sns.barplot(x=list(df.keys()), y=list(df.values))
        for index, value in enumerate(list(df.values)):
            ax.text(index, value, f"{value:.2f}", ha='center', va='bottom', fontsize=12)
    if t_test:
        from scipy.stats import ttest_rel
        t_statistic, p_value = ttest_rel(df["TFvelo"], df[df.keys()[1]])
        t_statistic_sci = "{:.2e}".format(t_statistic)
        p_value_sci = "{:.2e}".format(p_value)
        if df["TFvelo"].mean() < df[df.keys()[1]].mean():
            yy = 0.85
        else:
            yy = 0.05
        plt.text(0.45, yy, f"T-stat: {t_statistic_sci}\nP-val: {p_value_sci}",
         fontsize=15, ha='right', va='bottom', transform = plt.gca().transAxes)

    plt.title(save_name.replace('_', ' '), fontsize=16)
    plt.xlabel('Methods', fontsize=16)
    plt.ylabel('Values', fontsize=16)
    if (y_min is not None) and (y_max is not None):
        plt.ylim(y_min, y_max)
    #ax = plt.gca()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.savefig('figures/'+save_name+'_'+fig_type+'.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return


def check_gene(adata, genes):
    genes_checked = []
    for g in genes:
        if g in adata.var_names:
            genes_checked.append(g)
    return genes_checked


def show_phase(args, adata_TFv, adata_baselines, genes):    
    genes_checked = check_gene(adata_TFv, genes)
    n_fig = 20
    for ii in range(int((len(genes_checked)-1)/n_fig)+1):
        TFv.pl.velocity(adata_TFv, genes_checked[ii*n_fig: (ii+1)*n_fig], ncols=4, add_outline=True, layers='na', dpi=300, fontsize=15, save='phase_TFvelo_'+str(ii)+'.png') #layers='all'

    for b in baselines:
        print(b)
        adata_b = adata_baselines[b]

        if b == 'scvelo':
            genes_checked = check_gene(adata_b, genes)
            n_fig = 20
            for ii in range(int((len(genes_checked)-1)/n_fig)+1):
                scv.pl.velocity(adata_b, genes_checked[ii*n_fig: (ii+1)*n_fig], ncols=4, add_outline=True, layers='na', dpi=300, fontsize=15, save='phase_scvelo_'+str(ii)+'.png') #layers='all'
        elif b == 'unitvelo':
            genes_checked = check_gene(adata_b, genes)
            label = adata_b.uns['label']
            for g in genes_checked:
                print(g)
                utv.pl.plot_range(g, adata_b, utv_config, show_legend=False, show_ax=True, palette=list(adata_b.uns[label+'_colors']), save_fig=True)

        elif b == 'dynamo':
            genes_checked = check_gene(adata_b, genes)
            n_fig = 5
            for ii in range(int((len(genes_checked)-1)/n_fig)+1):
                save_kwargs = {
                                "path": 'figures/',
                                "prefix": 'phase_dynamo_'+str(ii),
                                "dpi": 300,
                                "ext": 'pdf',
                                "transparent": True,
                                "close": True,
                                "verbose": True
                            }

                dyn.pl.phase_portraits(adata_b, genes=genes_checked[ii*n_fig: (ii+1)*n_fig], color='clusters',  
                                    discrete_continous_div_color_key=[list(adata_b.uns['clusters_colors']), list(adata_b.uns['clusters_colors']), list(adata_b.uns['clusters_colors'])], 
                                    ncols=3,  pointsize=5, save_show_or_return='save', save_kwargs=save_kwargs,
                                    show_quiver=True, quiver_size=5) # no_vel_u=False
    return 


def draw_y_WX_t(args, adata, g):
    TFs = adata[:, g].layers['WX'].reshape(-1)
    target = adata[:, g].layers['y'].reshape(-1)
    t = adata.obs['velocity_pseudotime']
    
    sns.set(font_scale=2, style="white")
    plt.grid(False)
    fig, ax1 = plt.subplots()
    color1 = '#1f77b4'
    sns.scatterplot(x=t, y=TFs, color=color1, label='TFs', ax=ax1, legend=False)
    ax1.set_xlabel('t')
    ax1.set_ylabel('TFs', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    sns.scatterplot(x=t, y=target, color=color2, label='Target', ax=ax2, legend=False)
    ax2.set_ylabel('Target', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title(g)
    plt.tight_layout(pad=0.2) 
    plt.savefig('figures/TFvelo_phase_TFvelo_'+g+'_y_WX_t.png', dpi=300)
    plt.close()
    
    weights_final = np.array(adata[:, g].varm['fit_weights_final'][0])
    #TF_id = np.argmax(abs(weights_final))
    for TF_id, TF_weight in enumerate(weights_final):
        if abs(TF_weight)<0.5:
            continue
        TF_name = adata[:, g].varm['TFs'][0, TF_id]
        if not TF_name in adata.var_names:
            continue
        TF = adata[:, TF_name].layers['M_total'].reshape(-1)
        sns.set(font_scale=2, style="white")
        fig, ax1 = plt.subplots()
        color1 = '#1f77b4'
        sns.scatterplot(x=t, y=TF, color=color1, label='TF('+TF_name+')', ax=ax1, legend=False)
        ax1.set_xlabel('t')
        ax1.set_ylabel('TF('+TF_name+')', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax2 = ax1.twinx()
        color2 = '#ff7f0e'
        sns.scatterplot(x=t, y=target, color=color2, label='Target', ax=ax2, legend=False)
        ax2.set_ylabel('Target', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        plt.title(g)
        plt.tight_layout(pad=0.2) 
        plt.savefig('figures/TFvelo_phase_TFvelo_'+g+'_y_'+TF_name+'_t_' + str(round(TF_weight,2))+'.png', dpi=300)
        plt.close()
        sns.set(style="white")
        plt.grid(False)
    return



def draw_u_s_t(adata, g, t_key):
    u = adata[:, g].layers['Mu'].reshape(-1)
    s = adata[:, g].layers['Ms'].reshape(-1)
    if t_key=='fit_t':
        t = adata[:, g].layers['fit_t'].reshape(-1)
    elif t_key=='pseudotime':
        t = adata.obs['velocity_pseudotime']

    sns.set(font_scale=2, style="white")
    plt.grid(False)
    fig, ax1 = plt.subplots()
    color1 = '#1f77b4'
    sns.scatterplot(x=t, y=u, color=color1, label='Unspliced', ax=ax1, legend=False)
    ax1.set_xlabel('t')
    ax1.set_ylabel('Unspliced', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    sns.scatterplot(x=t, y=s, color=color2, label='Spliced', ax=ax2, legend=False)
    ax2.set_ylabel('Spliced', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title(g)
    plt.tight_layout(pad=0.2) 
    plt.savefig('figures/scvelo_phase_scvelo_'+g+'_u_s_'+t_key+'.png', dpi=300)
    plt.close()
    
    return

def show_one_phase(args, adata_TFv, adata_baselines, genes):
    for g in genes:
        if not g in adata_TFv.var_names:
            print(adata_TFv.shape, 'does not include gene', g)
            continue
        TFv.pl.velocity(adata_TFv, g, ncols=4, add_outline=True, layers=['y'], dpi=300, color_map='gnuplot_r', fontsize=15, save='phase_TFvelo_'+g+'_y.png') 
        #TFv.pl.velocity(adata_TFv, g, ncols=4, add_outline=True, layers=['M_total'], dpi=300, fontsize=15, save='phase_TFvelo_'+g+'_Mtotal.png') 
        #TFv.pl.velocity(adata_TFv, g, ncols=4, add_outline=True, layers='all', dpi=300, fontsize=15, save='phase_TFvelo_'+g+'_all.png')
        TFv.pl.velocity(adata_TFv, g, ncols=4, add_outline=True, layers=['velocity', 'y'], color_map='gnuplot_r', dpi=300, fontsize=15, save='phase_TFvelo_'+g+'_all.png')
        TFv.pl.velocity(adata_TFv, g, ncols=4, add_outline=True, layers='na', dpi=300, fontsize=15, save='phase_TFvelo_'+g+'.png') 
        if 'fit_t_shift' in adata_TFv.layers:
            TFv.pl.scatter(adata_TFv, x='fit_t_shift', y=g, ncols=4, frameon=False, fontsize=20, xlabel='fit_t', ylabel='expression', save='phase_TFvelo_'+g+'_yFitT.png')
        TFv.pl.scatter(adata_TFv, x='velocity_pseudotime', y=g, ncols=4, frameon=True, fontsize=20, xlabel='t', ylabel='expression', save='phase_TFvelo_'+g+'_yPseudoT.png')
        draw_y_WX_t(args, adata_TFv, g)
        for b in adata_baselines.keys():
            adata_b = adata_baselines[b]
            if (b == 'scvelo') and (g in adata_b.var_names):
                scv.pl.velocity(adata_b, g, ncols=4, add_outline=True, layers='na', dpi=300, fontsize=15, save='phase_scvelo_'+g+'.png') #layers='all'
                scv.pl.scatter(adata_b, x='velocity_pseudotime', y=g, ncols=4, frameon=True, fontsize=20, xlabel='t', ylabel='expression', save='phase_scvelo_'+g+'_t.png')
                if adata_b[:,g].var['velocity_genes'][0]:
                    scv.pl.scatter(adata_b, x='fit_t', y=g, ncols=4, frameon=False, fontsize=20, xlabel='fit_t', ylabel='expression', save='phase_scvelo_'+g+'_fit_t.png')
                    draw_u_s_t(adata_b, g, t_key='fit_t')
                    draw_u_s_t(adata_b, g, t_key='pseudotime')
            elif (b == 'unitvelo') and (g in adata_b.var_names):
                label = adata_b.uns['label']
                utv.pl.plot_range(g, adata_b, utv_config, show_legend=False, show_ax=True, palette=list(adata_b.uns[label+'_colors']), save_fig=True)
            elif (b == 'dynamo') and (g in adata_b.var_names):
                show_phase_dynamo(args, adata_b, g)
    return 



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="pancreas", help='pancreas, gastrulation_erythroid, hesc1, merfish, pons, 10x_mouse_brain') 
    parser.add_argument( '--save_name', type=str, default='_demo', help='save_name')
    parser.add_argument( '--to_show', type=int, default=0, help='to show during getting error')

    args = parser.parse_args() 
    if args.dataset_name in ['pancreas', 'gastrulation_erythroid', 'pons']:
        baselines = ['scvelo', 'unitvelo', 'dynamo']
    else:
        baselines = []

    args.result_path = 'TFvelo_'+ args.dataset_name + args.save_name+ '/'
    print('------------------------------------------------------------')   
    print(args)   

    args.baseline_velo_gene = 1
    adata_TFv, adata_baselines = get_adatas(args, baselines)
    genes = get_gene(args, adata_TFv, adata_baselines) # common genes
    
    get_phase_cluster(args, adata_TFv, adata_baselines, genes)
    intra_class_distance_all_df = pd.read_csv('figures/intra_class_distance_'+args.dataset_name+'.txt', sep='\t')
    draw_plot(intra_class_distance_all_df, save_name='Intra_Class_Distance_on_Phase_Portrait',
            fig_type='boxplot', t_test=True)#, y_min=0.2, y_max=1)
    inter_class_distance_all_df = pd.read_csv('figures/inter_class_distance_'+args.dataset_name+'.txt', sep='\t')
    draw_plot(inter_class_distance_all_df, save_name='Inter_Class_Distance_on_Phase_Portrait',
            fig_type='boxplot', t_test=True)#, y_min=0.5, y_max=2)

    get_phase_err(args, adata_TFv, adata_baselines, genes)
    losses_all_df = pd.read_csv('figures/Losses_'+args.dataset_name+'.txt', sep='\t')
    draw_plot(losses_all_df, save_name='Fitting_Loss_on_Phase_Portrait',
            fig_type='boxplot')#, y_min=0, y_max=3)
    