import unitvelo as utv
import scvelo as scv
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import TFvelo as TFv
import os

velo = utv.config.Configuration()
velo.R2_ADJUST = True 
velo.IROOT = None
velo.FIT_OPTION = '1'
velo.GPU = 1



def summary_scores(all_scores):
    sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s}
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg

def utv_eva(args, method='unitvelo'):
    print('-------------------', method, '--------------------')
    adata = ad.read_h5ad(args.result_path + method + '.h5ad')
    if method == 'scvelo':
        adata = adata[:, adata.var['velocity_genes'] == True]
    elif method == 'unitvelo':
        adata = adata[:, adata.var['velocity_genes'] == True]
    elif method == 'dynamo':
        subvar = adata.var.loc[adata.var['gamma'] != 'None']
        adata = adata[:, subvar.index]
        adata.obsm['velocity_umap'] = np.nan_to_num(adata.obsm['velocity_umap']) 
        adata.layers['velocity'] = adata.layers['velocity_S'] 
        scv.pp.neighbors(adata)

    if args.dataset_name == 'pancreas':
        cluster_edges = [('Ductal', 'Ngn3 low EP'), ('Ngn3 low EP', 'Ngn3 high EP'), ('Ngn3 high EP', 'Pre-endocrine'), 
            ('Pre-endocrine', 'Beta'), ('Pre-endocrine', 'Alpha'), ('Pre-endocrine', 'Delta'), ('Pre-endocrine', 'Epsilon')]

    elif args.dataset_name == 'gastrulation_erythroid':
        cluster_edges = [('Blood progenitors 1', 'Blood progenitors 2'), ('Blood progenitors 2', 'Erythroid1'), 
                         ('Erythroid1', 'Erythroid2'), ('Erythroid2', 'Erythroid3')]

    elif args.dataset_name == 'pons':
        cluster_edges = [('OPCs', 'COPs'), ('COPs', 'NFOLs'), ('NFOLs', 'MFOLs')]

    metrics = utv.evaluate(adata, cluster_edges, k_cluster='clusters', k_velocity='velocity')
    Cross_Boundary_Direction_Correctness = list(summary_scores(metrics['Cross-Boundary Direction Correctness (A->B)'])[0].values())
    In_Cluster_Coherence = list(summary_scores(metrics['In-cluster Coherence'])[0].values())
    return Cross_Boundary_Direction_Correctness, In_Cluster_Coherence


def draw_plot(df, save_name, fig_type='boxplot'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    my_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    color_mapping = {method: color for method, color in zip(df.keys(), my_palette)}

    plt.figure()
    sns.set_palette(my_palette[:df.shape[-1]])
    df_mean = df.mean()
    df_mean_sorted = df_mean.sort_values(ascending=False)
    colors_sorted = [color_mapping[method] for method in df_mean_sorted.index]
    if fig_type=='boxplot':
        #ax = sns.boxplot(data=df)
        #plt.ylim(np.percentile(df['unitvelo'], 10), 1)
        ax = sns.boxplot(data=df, order=df_mean_sorted.index.tolist(), palette=colors_sorted)
    elif fig_type=='violinplot':
        ax = sns.violinplot(data=df)
    elif fig_type=='barplot':
        ax = sns.barplot(x=df_mean_sorted.index, y=df_mean_sorted.values, palette=colors_sorted)
        for index, value in enumerate(list(df_mean_sorted.values)):
            ax.text(index, value, f"{value:.2f}", ha='center', va='bottom', fontsize=12)
    else:
        return
    if len(df.keys())>4:
        plt.xticks(rotation=30) 
    plt.title(save_name.replace('_', ' '), fontsize=16) 
    plt.xlabel('Methods', fontsize=16)
    plt.ylabel('Values', fontsize=16)
    #ax = plt.gca()
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    plt.savefig('figures/'+save_name+'_'+fig_type+'.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return

def run_utv_metric(args, methods):
    Cross_Boundary_Direction_Correctness_all, In_Cluster_Coherence_all = {}, {}
    for method in methods:
        Cross_Boundary_Direction_Correctness, In_Cluster_Coherence = utv_eva(args, method)
        Cross_Boundary_Direction_Correctness_all[method] = Cross_Boundary_Direction_Correctness
        In_Cluster_Coherence_all[method] = In_Cluster_Coherence

    Cross_Boundary_Direction_Correctness_df = pd.DataFrame(Cross_Boundary_Direction_Correctness_all)
    Cross_Boundary_Direction_Correctness_df.to_csv('figures/Cross_Boundary_Direction_Correctness_'+args.dataset_name+'.txt', sep='\t', index=False)
    In_Cluster_Coherence_all_df = pd.DataFrame(In_Cluster_Coherence_all)
    In_Cluster_Coherence_all_df.to_csv('figures/In_Cluster_Coherence_all_'+args.dataset_name+'.txt', sep='\t', index=False)
    return

def draw_utv_metric(args):
    Cross_Boundary_Direction_Correctness_df = pd.read_csv('figures/Cross_Boundary_Direction_Correctness_'+args.dataset_name+'.txt', sep='\t')
    In_Cluster_Coherence_all_df = pd.read_csv('figures/In_Cluster_Coherence_all_'+args.dataset_name+'.txt', sep='\t')

    draw_plot(Cross_Boundary_Direction_Correctness_df, save_name='Cross_Boundary_Direction_Correctness',
              fig_type='boxplot')
    draw_plot(Cross_Boundary_Direction_Correctness_df, save_name='Cross_Boundary_Direction_Correctness',
              fig_type='barplot')
    draw_plot(In_Cluster_Coherence_all_df, save_name='In_Cluster_Coherence', 
              fig_type='boxplot')
    draw_plot(In_Cluster_Coherence_all_df, save_name='In_Cluster_Coherence',
              fig_type='barplot')
    return

def get_velocity_consistency(args, methods):
    confidence_all = {}
    for method in methods:           
        print('-------------------', method, '--------------------')
        adata = ad.read_h5ad(args.result_path + method + '.h5ad')
        if method == 'dynamo':
            subvar = adata.var.loc[adata.var['gamma'] != 'None']
            adata = adata[:, subvar.index]
            adata.obsm['velocity_umap'] = np.nan_to_num(adata.obsm['velocity_umap']) 
            adata.layers['velocity'] = adata.layers['velocity_S'].todense()
            adata.layers['Ms'] = adata.layers['M_s'].todense()
            scv.pp.neighbors(adata)
            scv.tl.velocity_graph(adata)

        if method == 'TFvelo':
            TFv.tl.velocity_confidence(adata)
        else:
            scv.tl.velocity_confidence(adata)
        adata.obs['velocity_consistency'] = adata.obs['velocity_confidence']
        del adata.obs['velocity_confidence']
        scv.pl.scatter(adata, c='velocity_consistency', cmap='coolwarm', fontsize=20, save='velocity_consistency_'+method+'.png')
        confidence_all[method] = adata.obs['velocity_consistency']
        print(adata.obs['velocity_consistency'].mean())
    confidence_all_df = pd.DataFrame(confidence_all)
    confidence_all_df.to_csv('figures/velocity_consistency_'+args.dataset_name+'.txt', sep='\t', index=False)
    return

def draw_confidence(args):
    confidence_all_df = pd.read_csv('figures/velocity_consistency_'+args.dataset_name+'.txt', sep='\t')
    draw_plot(confidence_all_df, save_name='velocity_consistency', 
              fig_type='boxplot')
    draw_plot(confidence_all_df, save_name='velocity_consistency', 
              fig_type='barplot')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="pancreas", help='pancreas, gastrulation_erythroid, pons') 
    parser.add_argument( '--save_name', type=str, default='_demo', help='save_name')
    args = parser.parse_args() 

    args.result_path = 'TFvelo_'+ args.dataset_name + args.save_name+ '/'
    print('------------------------------------------------------------')   
    print(args)   
    methods = ['TFvelo', 'scvelo', 'unitvelo', 'dynamo', 'cellDancer']

    if not os.path.exists('figures'):
        os.makedirs('figures')

    run_utv_metric(args, methods)
    draw_utv_metric(args)

    get_velocity_consistency(args, methods)
    draw_confidence(args)


