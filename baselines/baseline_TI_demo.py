import scvelo as scv
import anndata as ad
import numpy as np
import scanpy as sc

import matplotlib
matplotlib.use('AGG')

import os


def run_paga(args, adata, iroot_tyre):
    # paga trajectory inference
    sc.tl.paga(adata, groups='clusters')
    #sc.pl.paga(adata, color=['clusters'], save='')

    # dpt_pseudotime  inference
    adata.uns['iroot'] = np.flatnonzero(adata.obs['clusters']  == iroot_tyre)[0]
    sc.tl.dpt(adata)
    #sc.pl.umap(adata, color=['dpt_pseudotime'], legend_loc='on data', save='_'+args.dataset_name+'_dpt_pseudotime.png')
    scv.pl.scatter(adata, color='dpt_pseudotime', color_map='gnuplot', size=20, save=args.dataset_name+'_dpt_pseudotime.png')
    return adata

def run_palantir(args, adata, iroot_tyre):
    sc.external.tl.palantir(adata, n_components=5, knn=30)

    iroot = np.flatnonzero(adata.obs['clusters']  == iroot_tyre)[0]
    start_cell = adata.obs_names[iroot]

    pr_res = sc.external.tl.palantir_results(
        adata,
        early_cell=start_cell,
        ms_data='X_palantir_multiscale',
        num_waypoints=500,
    )
    adata.obs['pr_pseudotime'] = pr_res.pseudotime
    adata.obs['pr_entropy'] = pr_res.entropy
    #adata.obs['pr_branch_probs'] = pr_res.branch_probs
    #adata.uns['pr_waypoints'] = pr_res.waypoints

    #sc.pl.umap(adata, color=['pr_pseudotime'], legend_loc='on data', save='_'+args.dataset_name+'_pr_pseudotime.png')
    scv.pl.scatter(adata, color='pr_pseudotime', color_map='gnuplot', size=20, save=args.dataset_name+'_pr_pseudotime.png')
    return adata


def main(args):
    adata = ad.read_h5ad(args.result_path + 'pp.h5ad')
    #adata.X = adata.layers['M_total']

    if args.dataset_name == 'pancreas':
        iroot_tyre = 'Ductal'
    elif args.dataset_name == 'gastrulation_erythroid':
        iroot_tyre = 'Blood progenitors 1'
    elif args.dataset_name == 'hesc1':
        iroot_tyre = 'E3'
    elif args.dataset_name == '10x_mouse_brain':
        iroot_tyre = 'RG, Astro, OPC'  

    if 'X_pca' not in adata.obsm.keys():
        print('PCA ing')
        sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
        # sc.pl.pca(adata, color=['clusters'], show=False, save='_clusters.png')
    if ('X_umap' not in adata.obsm.keys()):
        print('Umap ing')
        if args.dataset_name == 'hesc1':
            sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=30, n_pcs=5)
        sc.tl.umap(adata)

    adata = run_paga(args, adata, iroot_tyre)
    adata = run_palantir(args, adata, iroot_tyre)

    adata.write(args.result_path + 'TI.h5ad')

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="pancreas", help='pancreas, gastrulation_erythroid') 
    parser.add_argument( '--n_jobs', type=int, default=16, help='n_jobs')
    parser.add_argument( '--save_name', type=str, default='', help='save_name')
    args = parser.parse_args() 
    
    for args.dataset_name in ['pancreas', 'gastrulation_erythroid', '10x_mouse_brain', 'hesc1']:
        args.result_path = 'TFvelo_'+ args.dataset_name + args.save_name+ '/'
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)
        print('********************************************************************************************************')
        print('********************************************************************************************************')
        print(args)
        main(args) 
  
