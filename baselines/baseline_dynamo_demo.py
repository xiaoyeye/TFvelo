import pandas as pd    
import scvelo as scv
import anndata as ad
import numpy as np
import scanpy as sc
import dynamo as dyn

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('AGG')

import os, sys

def data_type_tostr(adata, key=None):
    print(adata)
    if key is None:
        for key in list(adata.var):
            if adata.var[key][0] in [True, False]:
                print('Transfering', key, 'because True/False')
                adata.var[key] = adata.var[key].map({True: 'True', False:'False'})
            if adata.var[key][0] is None:
                print('Transfering', key, 'because None')
                for j in range(len(adata.var[key])):
                    if adata.var[key][j] is None:
                        adata.var[key][j] = 'None'
                    else:
                        adata.var[key][j] = str(adata.var[key][j])
            if adata.var[key][0] is np.nan:
                print('Transfering', key, 'because NaN')
                for j in range(len(adata.var[key])):
                    if adata.var[key][j] is None:
                        adata.var[key][j] = 'NaN'
                    else:
                        adata.var[key][j] = str(adata.var[key][j])
    elif key in adata.var.keys():
        if adata.var[key][0] in [True, False]:
            print('Transfering', key)
            adata.var[key] = adata.var[key].map({True: 'True', False:'False'})
    if 'cell_phase_genes' in adata.uns:
        del adata.uns['cell_phase_genes']
    return 

def dynamo_workflow_scNTseq(adata, **kwargs):
    preprocessor = dyn.pp.Preprocessor(cell_cycle_score_enable=True)
    preprocessor.preprocess_adata(adata, recipe='monocle', **kwargs)

    dyn.tl.dynamics(adata)

    dyn.tl.reduceDimension(adata)

    dyn.tl.cell_velocities(adata, calc_rnd_vel=True, transition_genes=adata.var_names)

    dyn.vf.VectorField(adata, basis='umap')
    return

def main(args):
    if args.dataset_name == 'pancreas':
        adata = scv.datasets.pancreas()
    elif args.dataset_name == 'gastrulation_erythroid':
        adata = scv.datasets.gastrulation_erythroid()   
        adata.obs['clusters'] = adata.obs['celltype'].copy()

    print(adata)

    dyn.pp.recipe_monocle(adata)
    dyn.tl.dynamics(adata, cores=3)

    dyn.tl.reduceDimension(adata)
    dyn.tl.cell_velocities(adata)

    dyn.tl.cell_wise_confidence(adata)
    dyn.vf.VectorField(adata)

    print(adata)

    data_type_tostr(adata)
    adata.write(args.result_path + 'dynamo.h5ad')
    return

def analysis(args):
    adata_TFv = ad.read_h5ad(args.result_path+'TFvelo.h5ad') 
    n_colors = len(adata_TFv.obs['clusters'].cat.categories)
    adata_TFv.uns['clusters_colors'] = adata_TFv.uns['clusters_colors'][:n_colors]

    method = 'dynamo'
    adata = ad.read_h5ad(args.result_path+method+'.h5ad')
    adata.uns['clusters_colors'] = adata_TFv.uns['clusters_colors'][adata_TFv.obs['clusters'].cat.categories.argsort()]
    
    dyn.vf.VectorField(adata, basis='umap', M=100)
    dyn.ext.ddhodge(adata, basis='umap')
    print(adata)

    save_kwargs = {"path": 'figures/', "prefix": 'dyn_'+args.dataset_name+'_embedding_stream', "dpi": 300, "ext": 'png'}
    dyn.pl.streamline_plot(adata, color=['clusters'], color_key=adata.uns['clusters_colors'], save_show_or_return='save', save_kwargs=save_kwargs)
    save_kwargs = {"path": 'figures/', "prefix": 'dyn_'+args.dataset_name+'_embedding_grid', "dpi": 300, "ext": 'png'}
    dyn.pl.grid_vectors(adata, color=['clusters'], color_key=adata.uns['clusters_colors'], save_show_or_return='save', save_kwargs=save_kwargs)
    #save_kwargs = {"path": 'figures/', "prefix": 'dyn_'+args.dataset_name+'_pseudotime', "dpi": 300, "ext": 'png'}
    #dyn.pl.streamline_plot(adata, color=['umap_ddhodge_potential'], save_show_or_return='save', save_kwargs=save_kwargs)
    scv.pl.scatter(adata, basis='umap', color='umap_ddhodge_potential', cmap='gnuplot', fontsize=20, save='dynamo_pseudotime.png')
    adata.write(args.result_path + 'dynamo.h5ad')
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="pancreas", help='pancreas, gastrulation_erythroid, pons, scNT_seq') 
    parser.add_argument( '--save_name', type=str, default='_demo', help='save_name')

    args = parser.parse_args() 
    args.result_path = 'TFvelo_'+ args.dataset_name + args.save_name+ '/'
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    print('********************************************************************************************************')
    print('********************************************************************************************************')
    print(args)

    #main(args) 
    analysis(args)
  
