from cmath import isnan
import TFvelo as TFv
from TFvelo.preprocessing.moments import get_connectivities
import anndata as ad
import scanpy as sc
import numpy as np

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation

import os, sys
#rootPath = os.path.dirname(sys.path[0])
#os.chdir(rootPath+'/RNA_velocity')

MOUSE_DATA = ['pancreas', 'mesc2', 'mesc1', 'dentategyrus', 'gastrulation_erythroid', 'scNT_seq']
HUMAN_DATA = ['hesc1', 'hesc2', 'bonemarrow', 'merfish']
FISH_DATA =  ['zebrafish']

def check_data_type(adata):
    for key in list(adata.var):
        if adata.var[key][0] in ['True', 'False']:
            print('Checking', key)
            adata.var[key] = adata.var[key].map({'True': True, 'False': False})
    return


def data_type_tostr(adata, key=None):
    if key is None:
        for key in list(adata.var):
            if adata.var[key][0] in [True, False]:
                print('Transfering', key)
                adata.var[key] = adata.var[key].map({True: 'True', False:'False'})
    elif key in adata.var.keys():
        if adata.var[key][0] in [True, False]:
            print('Transfering', key)
            adata.var[key] = adata.var[key].map({True: 'True', False:'False'})
    return   

from scipy import sparse
import sklearn
from typing import Union, Optional
from sklearn.preprocessing import normalize
Array = Union[np.ndarray, sparse.spmatrix]
def tfidf(X: Array) -> Array:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)

    Parameters
    ----------
    X
        Input matrix

    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

def lsi(
        adata: ad.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi
    return


def plot_TF_target(value1, value2, fig_name, cell_types, text=None, colors=None, save_name='', save_path='figures/TF_target/'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        cluster_map = list(cell_types.values.categories)
        cell_types_idx = []
        for cell_type in cell_types:
            cell_types_idx.append(cluster_map.index(cell_type))
        cell_types_idx = np.array(cell_types_idx)
        #adata.obs['clusters_ids'] = cell_types_idx
        #adata.uns['clusters_map'] = np.array(cluster_map)
        n_types = len(cluster_map)
        if colors is None:
            sc = plt.scatter(value2, value1, c=cell_types_idx, cmap='rainbow', alpha=0.7, s=5)
            plt.legend(handles=sc.legend_elements(num=n_types-1)[0], labels=cluster_map, bbox_to_anchor=(0.8,0.7), loc='center left', prop={'size': 9})
        else:
            color_idx = colors[cell_types_idx]
            sc = plt.scatter(value2, value1, c=color_idx, alpha=0.7, s=5)
    except:
        sc = plt.scatter(value2, value1, alpha=0.7, s=5)
    plt.text(0.1, 0.95, text, fontdict={'size':'20','color':'Red'},  transform = plt.gca().transAxes)
    #plt.title(fig_name)

    tmp = fig_name.split('_')
    target, TF = tmp[0]+' (Target)', tmp[2]+' (TF)'
    plt.xlabel(target, fontdict={'size':20})
    plt.ylabel(TF, fontdict={'size':20})

    plt.savefig(save_path +fig_name+save_name+ '.jpg', bbox_inches='tight', pad_inches=0.1)
    plt.close()


def weight_analysis(adata):
    import seaborn as sns
    weights = []
    TFs_times = []
    for ii in range(adata.shape[1]):
        if adata.var['velocity_genes'][ii]: 
            weights_gene = adata.varm['fit_weights'][ii, :adata.var['n_TFs'][ii]]
            #TFs_times_gene = adata.varm['TFs_times'][ii, :adata.var['n_TFs'][ii]]
            weights += list(weights_gene)
            #TFs_times += list(TFs_times_gene)
    weights = np.array(weights)
    '''    
    TFs_times = np.array(TFs_times)
    plt.scatter(TFs_times, weights)
    plt.savefig('figures/weights_times.jpg',bbox_inches='tight',pad_inches=0.1,dpi=300)
    plt.close()
    plt.clf()
    '''

    weights = weights[weights<4]
    weights = weights[weights>0]
    fig = sns.distplot(weights, kde=False)
    fig.set_xlim(0,4) 
    #fig.set_ylim(0,1) 
    plt.yticks([])
    plt.xlabel('Weights', fontdict={'size':20})
    plt.ylabel('Frequency', fontdict={'size':20})
    plt.title('Weights Distribution', fontdict={'size':20})
    plt.savefig('figures/weights.jpg',bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.close()
    plt.clf()
    return 


def main(args):
    adata = anndata.read_h5ad(args.data_path+"rc.h5ad") 
    adata.var_names_make_unique()
    num_cell, num_gene = adata.X.shape[0], adata.X.shape[1]
    check_data_type(adata)
    print(adata)
    if args.dataset_name == 'merfish':
        adata.uns['clusters_colors'] = np.array(['red', 'green', 'blue', 'purple'])
        args.n_bins = 26 
    if not os.path.exists('figures'):
        os.makedirs('figures')
    np.savetxt('figures/all_pp_genes.txt', np.array(adata.var_names), fmt='%s', delimiter='/t')
    adata.var['filter_count'] = (adata.layers['filtered']==1).sum(0)
    losses = adata.varm['loss'].copy()
    losses[np.isnan(losses)] = 1e6
    adata.var['min_loss'] = losses.min(1)
    velocity_genes = (adata.var['fit_alpha']>0) & (adata.var['filter_count']>=args.n_cell_thres) 
    np.savetxt('figures/all_velo_genes.txt', np.array(adata[:,velocity_genes].var_names), fmt='%s', delimiter='/t')
    print('num velocity_genes:', len(velocity_genes[velocity_genes==True]))
    adata.var['velocity_genes'] = velocity_genes

    weight_analysis(adata)

  
    n_cells = adata.layers['velo_hat'].shape[0]
    expanded_scaling_y = np.expand_dims(np.array(adata.var['fit_scaling_y']),0).repeat(n_cells,axis=0)
    adata.layers['velocity'] = adata.layers['velo_hat'] / expanded_scaling_y 
    #adata.layers['velocity'] = adata.layers['velo_normed']  #################################
    adata.layers['target_y'] = adata.layers['y'] / expanded_scaling_y
    
    if args.basis == 'lsi':
        lsi(adata, 50)
        adata.obsm['X_lsi'] = adata.obsm['X_lsi'][:,1:]
        TFv.pl.scatter(adata, basis='lsi',  save='lsi')
    if 'X_pca' not in adata.obsm.keys():
        print('PCA ing')
        sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
        # sc.pl.pca(adata, color=['clusters'], show=False, save='_clusters.png')
    if (args.basis == 'umap') and ('X_umap' not in adata.obsm.keys()):
        print('Umap ing')
        if args.dataset_name == 'hesc1':
            sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=30, n_pcs=5)
        sc.tl.umap(adata)
        sc.pl.umap(adata, color=['clusters'], show=False, save='_clusters.png')#, legend_loc='on data')
    if (args.basis == 'tsne') and ('X_tsne' not in adata.obsm.keys()):
        print('tsne ing')
        sc.tl.tsne(adata, use_rep="X_pca", n_pcs=10)
        sc.pl.tsne(adata, color=['clusters'], show=False, save='_clusters.png')#, legend_loc='on data')


    TFv.tl.velocity_graph(adata, basis=args.basis, vkey='velocity', xkey=args.layer) #  xkey='target_y'
    TFv.tl.velocity_pseudotime(adata, vkey='velocity', modality=args.layer) #  modality='target_y'
    TFv.pl.scatter(adata, basis=args.basis, color='velocity_pseudotime', cmap='gnuplot', fontsize=20, save='pseudotime')

    if args.dataset_name =='pancreas':
        cutoff_perc = 20
    else:
        cutoff_perc = 0
    smooth = 1 

    TFv.pl.velocity_embedding_stream(adata, vkey='velocity', use_derivative=True, density=2, basis=args.basis, \
        smooth=smooth, cutoff_perc=cutoff_perc, fontsize=20, save='embedding_stream', n_bins=args.n_bins) # 

    top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index#[:300]
    TFv.pl.heatmap(adata, var_names=top_genes, sortby='velocity_pseudotime', col_color='clusters', filter_start=0, filter_end=1, n_convolve=100, save='heatmap.png')

    top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index
    if args.dataset_name == 'pancreas':
        show_genes_main = ['LITAF', 'H19', 'CTSF', 'ECE1', 'RPS8'] 
        show_genes_SI = ['LDHA', 'NRTN', 'MAML3', 'CD47', 'TUBB4B', 'HMGN3', 'CCND3', 'GSTZ1', 'CLDN6', 'LRP11',
                       'BEX2', 'ANXA4', 'STXBP1', 'SURF4', 'TMSB4X', 'IMMP1L', 'GNS'] 
    elif args.dataset_name == 'gastrulation_erythroid':
        show_genes_main = ['TACC1', 'PFN1', 'TMEM14C']
        show_genes_SI = [ 'CHCHD10', 'CALR',  'TPST2', 'RAB11A', 'DIAPH1', 'GPX1', 'GCSH', 'FECH', 'RND2', 'PRTG', 
                         'CANX', 'SMTNL2', 'CTSB',  'KLF1', 'SPNS1', 'TSPAN4'] 
    elif args.dataset_name == 'hesc1':
        show_genes_main = ['GSTP1', 'SERTAD1','RGS2']
        show_genes_SI = ['ANKRD37', 'CCNA1', 'PTGES', 'CNNM2', 'LY6E', 'DNMT3L', 'CNST', 'TCL1B', 'FXYD4', 'SLCO4A1',
                      'CHKB', 'SOX15', 'RBM48', 'PPP1R14A', 'SNAPC2', 'HIST1H2AA', 'ZNF350']
    elif args.dataset_name == 'merfish':
        show_genes_main = ['CD44', 'ACER2', 'SERPINE2',  'COL5A1']
        show_genes_SI = ['LRP1', 'NOTCH3', 'NANOS1', 'ANTXR2', 'FAT4', 'INADL', 'COL4A4', 'PTPRH', 'OAF', 'CORO1C', 
                         'CAMK2N1', 'ITGA6','PRKCE', 'MSX1','TNRC18','PHLDA3', 'TBX3','CCND1', 'SEMA3C' ,'MET',
                         'THBS1','ST3GAL1','EFR3B','LOXL4','KLHL18', 'CCND2']
    else:
        show_genes_main = top_genes[:8]
        show_genes_SI = top_genes[:16]
    #if args.show_all_genes: 
    #    for i in range(50):
    #        TFv.pl.scatter(adata, basis=top_genes[i*20: (i+1)*20], ncols=5, frameon=False, save='top_genes_'+str(i)+'.png')
    print(adata)
    show_genes_SI = show_genes_SI[:16]
    TFv.pl.scatter(adata, basis=show_genes_SI, ncols=4, frameon=False, fontsize=20, save='WX_y_SI.png')
    TFv.pl.scatter(adata, x='velocity_pseudotime', y=show_genes_SI, ncols=4, frameon=True, fontsize=20, xlabel='t', ylabel='expression', save='y_t_SI.png')
    TFv.pl.velocity(adata, show_genes_SI, ncols=2, add_outline=True, layers=[args.layer], fontsize=12, dpi=300, save='info_SI.png') #layers='all'
    for show_gene in show_genes_main:
        #TFv.pl.scatter(adata, basis=show_gene, ncols=4, frameon=False, save=show_gene+'_WX_y.png')
        TFv.pl.velocity(adata, show_gene, ncols=2, add_outline=True, layers='na', dpi=300, fontsize=15, save=show_gene+'_WX_y.png') #layers='all'
        TFv.pl.scatter(adata, x='velocity_pseudotime', y=show_gene, ncols=4, frameon=True, fontsize=20, xlabel='t', ylabel='expression', fontsize=20, save=show_gene+'_y_t.png')
        TFv.pl.velocity(adata, show_gene, ncols=2, add_outline=True, layers=[args.layer], fontsize=15, dpi=300, save=show_gene+'_info.png') #layers='all'

    TFv.tl.velocity_confidence(adata)
    keys = 'velocity_confidence'#, 'velocity_confidence_transition' #'velocity_length'
    TFv.pl.scatter(adata, c=keys, cmap='coolwarm', fontsize=20, save='velocity_confidence.png')
    print(adata.obs['velocity_confidence'].mean())
    np.savetxt('figures/TFvelo_'+args.dataset_name+'_velocity_confidence.txt', np.array(adata.obs['velocity_confidence']))

    data_type_tostr(adata)
    adata.write(args.data_path + 'analysis.h5ad')

    return





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="merfish", help='pancreas, scNT_seq, merfish, hesc1, mesc2, gastrulation_erythroid') 
    parser.add_argument( '--layer', type=str, default="M_total", help='M_total, total') 
    parser.add_argument( '--basis', type=str, default="umap", help='umap, tsne, pca')
    parser.add_argument( '--n_bins', type=int, default=32, help='n_bins in stream')
    parser.add_argument( '--n_cell_thres', type=int, default=10, help='min num of filtered cells in modeling each gene')
    parser.add_argument( '--save_name', type=str, default='', help='save_name')
    args = parser.parse_args() 
    
    args.data_path = 'TFvelo_'+ args.dataset_name + args.save_name+ '/'
    print('------------------------------------------------------------')   
    print(args)   
    main(args)
