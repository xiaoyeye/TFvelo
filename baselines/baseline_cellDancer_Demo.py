# import packages
import os
import sys
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
import celldancer as cd
import celldancer.cdplt as cdplt
from celldancer.cdplt import colormap, build_colormap
import celldancer.utilities as cdutil
import scvelo as scv
import anndata as ad
import scanpy as sc
import numpy as np

args_dataset = 'pancreas' #gastrulation_erythroid, pancreas
args_pp = 0
args_train = 0
args_postp = 0
n_jobs = 8


color_map = None
if args_dataset == 'pancreas':
    adata = scv.datasets.pancreas() 
    color_map = colormap.colormap_pancreas
elif args_dataset == 'gastrulation_erythroid':
    adata = scv.datasets.gastrulation_erythroid()   
    color_map = colormap.colormap_erythroid
    adata.obs['clusters'] = adata.obs['celltype'].copy()
elif args_dataset == 'bonemarrow':
    adata = scv.datasets.bonemarrow()
elif args_dataset == 'dentategyrus':
    adata = scv.datasets.dentategyrus()
elif args_dataset == 'larry':
    adata = ad.read_h5ad("data/larry/larry.h5ad")  
    adata.obs['clusters'] = adata.obs['state_info']
    adata.obsm['X_umap'] = np.stack([np.array(adata.obs['SPRING-x']), np.array(adata.obs['SPRING-y'])]).T
elif args_dataset == 'pons':
    adata = ad.read_h5ad("data/pons/oligo_lite.h5ad")  
    adata.obs['clusters'] = adata.obs['celltype']
      
print(adata)

if color_map is None:
    cluster_list = list(adata.obs['clusters'].cat.categories)
    color_map = build_colormap(cluster_list)

try:
    adata_TFv = ad.read_h5ad("../TFvelo_master/TFvelo_"+args_dataset+"/TFvelo.h5ad")  
    color_map = {}
    for i, c in enumerate(adata_TFv.obs['clusters'].cat.categories):
        color_map[c] = adata_TFv.uns['clusters_colors'][i]
except:
    print('No TFvelo colors')

save_folder = 'cellDancer_' + args_dataset +'/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
csv_path = save_folder + 'cell_type_u_s.csv'


if args_pp:
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30) # cell amount will influence the setti
    if 'X_umap' not in adata.obsm:
        if 'X_tsne' in adata.obsm:
            print('Copying tsne to umap')
            adata.obsm['X_umap'] = adata.obsm['X_tsne']
        else:
            sc.pp.neighbors(adata, n_neighbors=30)
            sc.tl.umap(adata)
        
    print(adata)  
    adata.write(save_folder + 'pp.h5ad')


    cell_type_u_s = cdutil.adata_to_df_with_embed(adata,
                                us_para=['Mu','Ms'],
                                cell_type_para='clusters',
                                embed_para='X_umap',
                                save_path=csv_path)
                                #gene_list=['Hba-x','Smim1']
else:
    adata = ad.read_h5ad(save_folder + 'pp.h5ad')
    cell_type_u_s = pd.read_csv(csv_path) 

print(cell_type_u_s)


if args_train: # to train model on each gene
    loss_df, cellDancer_df = cd.velocity(cell_type_u_s, permutation_ratio=0.5, n_jobs=n_jobs)
else:
    loss_df = pd.read_csv(save_folder+'loss.csv')
    cellDancer_df = pd.read_csv(save_folder+'cellDancer_estimation.csv')


if args_postp:
    # Compute cell velocity
    cellDancer_df=cd.compute_cell_velocity(cellDancer_df=cellDancer_df, projection_neighbor_size=100)

    # Plot cell velocity
    fig, ax = plt.subplots(figsize=(10,10))
    im = cdplt.scatter_cell(ax, cellDancer_df, colors=color_map, alpha=0.5, s=20, velocity=True, legend='on', min_mass=5, arrow_grid=(20,20))
    ax.axis('off')
    plt.savefig(save_folder+'arrowplot.png')
    plt.close()


    # set parameters
    dt = 0.001
    t_total = {dt: 10000}
    n_repeats = 10
    # estimate pseudotime
    cellDancer_df = cd.pseudo_time(cellDancer_df=cellDancer_df,
                                            grid=(30, 30),
                                            dt=dt,
                                            t_total=t_total[dt],
                                            n_repeats=n_repeats,
                                            speed_up=(60,60),
                                            n_paths = 5,
                                            psrng_seeds_diffusion=[i for i in range(n_repeats)],
                                            n_jobs=n_jobs)
    # plot pseudotime
    fig, ax = plt.subplots(figsize=(10,10))
    im=cdplt.scatter_cell(ax,cellDancer_df, colors='pseudotime', alpha=0.5, velocity=False)
    ax.axis('off')
    plt.savefig(save_folder+'pseudotime.png')
    plt.close()
                                            
    cellDancer_df.to_csv(os.path.join(save_folder, ('cellDancer_estimation_final.csv')),index=False)

else:
    cellDancer_df = pd.read_csv(save_folder+'cellDancer_estimation_final.csv')


