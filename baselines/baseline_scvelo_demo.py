import pandas as pd    
import scvelo as scv
import anndata as ad
import numpy as np
import scanpy as sc

import matplotlib
matplotlib.use('AGG')
import umap   

import os, sys


def main(args):
    scv.set_figure_params()
    if args.dataset_name == 'pancreas':
        adata = scv.datasets.pancreas()
    elif args.dataset_name == 'gastrulation_erythroid':
        adata = scv.datasets.gastrulation_erythroid()   
        adata.uns['clusters_colors'] = adata.uns['celltype_colors'].copy()
        adata.obs['clusters'] = adata.obs['celltype'].copy()

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(adata, n_jobs=args.n_jobs)
    scv.tl.velocity(adata, mode='dynamical')
    scv.tl.velocity_graph(adata)
    adata.write(args.result_path + 'scvelo.h5ad')
    return


def analysis(args):
    adata_TFv = ad.read_h5ad(args.result_path+'TFvelo.h5ad') 
    n_colors = len(adata_TFv.obs['clusters'].cat.categories)
    adata_TFv.uns['clusters_colors'] = adata_TFv.uns['clusters_colors'][:n_colors]

    method = 'scvelo'
    adata = ad.read_h5ad(args.result_path+method+'.h5ad')
    adata.uns['clusters_colors'] = adata_TFv.uns['clusters_colors'].copy()
    if args.dataset_name == 'gastrulation_erythroid':
        adata.obs['clusters'] = adata.obs['celltype'].copy()


    scv.tl.latent_time(adata)
    print(adata)
    scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot', fontsize=20, save=args.dataset_name+'_'+method+'_pseudotime.png')
    scv.pl.velocity_embedding_stream(adata, color='clusters', dpi=300, title='', save= args.dataset_name+'_'+method+'_embedding_stream.png')
    scv.pl.velocity_embedding_grid(adata, color='clusters', arrow_size=10, dpi=300, title='', save= args.dataset_name+'_'+method+'_embedding_grid.png')

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="pancreas", help='pancreas, gastrulation_erythroid') 
    parser.add_argument( '--n_jobs', type=int, default=28, help='n_jobs')
    parser.add_argument( '--save_name', type=str, default='_demo', help='save_name')

    args = parser.parse_args() 
    args.result_path = 'TFvelo_'+ args.dataset_name + args.save_name+ '/'
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    print('********************************************************************************************************')
    print('********************************************************************************************************')
    print(args)

    main(args) 
    analysis(args)
  
