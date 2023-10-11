import TFvelo as TFv
import anndata as ad
import scanpy as sc
import numpy as np
import scipy
import matplotlib
matplotlib.use('AGG')


np.set_printoptions(suppress=True)


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



def get_pseudotime(adata):
    TFv.tl.velocity_graph(adata, basis=None, vkey='velocity', xkey='M_total')
    TFv.tl.velocity_pseudotime(adata, vkey='velocity', modality='M_total') 
    TFv.pl.scatter(adata, basis=args.basis, color='velocity_pseudotime', cmap='gnuplot', fontsize=20, save='pseudotime')
    return adata


def get_sort_positions(arr):
    positions = np.argsort(np.argsort(arr))
    positions_normed = positions/(len(arr)-1)
    return positions_normed


def get_metric_pseudotime(adata, t_key='normed_t_sort'):
    n_cells, n_genes = adata.shape
    adata.var['spearmanr_pseudotime'] = 0.0
    for i in range(n_genes):
        correlation, _ = scipy.stats.spearmanr(adata.layers[t_key][:,i], adata.obs['velocity_pseudotime'])
        adata.var['spearmanr_pseudotime'][i] = correlation
    return adata


def show_adata(args, adata, save_name, show_all=0):
    if show_all:
        for i in range(int((len(adata.var_names)-1)/20)+1):
            genes2show = adata.var_names[i*20: (i+1)*20]
            TFv.pl.velocity(adata, genes2show, ncols=4, add_outline=True, layers='na', dpi=300, fontsize=15, save='WX_y_'+save_name+'_'+str(i)) #layers='all'
    if len(adata.obs['clusters'].cat.categories) > 10:
        legend_loc = 'right margin'
    else:
        legend_loc = 'on data'
    cutoff_perc = 20
    TFv.pl.velocity_embedding_stream(adata, vkey='velocity', use_derivative=False, density=2, basis=args.basis, \
        cutoff_perc=cutoff_perc, smooth=0.5, fontsize=20, recompute=True, \
        legend_loc=legend_loc, save='embedding_stream_'+save_name) # 

    return



def get_sort_t(adata):
    t = adata.layers['fit_t_raw'].copy()
    normed_t = adata.layers['fit_t_raw'].copy()
    n_bins = 20
    n_cells, n_genes = adata.shape
    sort_t = np.zeros([n_cells, n_genes])
    non_blank_gene = np.zeros(n_genes, dtype=int)
    hist_all, bins_all = np.zeros([n_genes, n_bins]), np.zeros([n_genes, n_bins+1])
    for i in range(n_genes):
        gene_name = adata.var_names[i]
        tmp = t[:,i].copy()
        if np.isnan(tmp).sum():
            non_blank_gene[i] = 1 
            continue
        hist, bins = np.histogram(tmp, bins=n_bins)
        hist_all[i], bins_all[i] = hist, bins
        if not (0 in list(hist)):
            if (tmp.min() < 0.1) and (tmp.max() > 0.8):
                blank_start_bin_id = np.argmin(hist)
                blank_end_bin_id = blank_start_bin_id
                non_blank_gene[i] = 1
                blank_start_bin = bins[blank_start_bin_id]
                blank_end_bin = bins[blank_end_bin_id]
                tmp = (tmp < blank_start_bin)*1 + tmp 
            else:
                blank_end_bin = tmp.min()
        else:
            blank_start_bin_id = list(hist).index(0)
            for j in range(blank_start_bin_id+1, len(hist)):
                if hist[j] > 0:
                    blank_end_bin_id = j
                    break
            blank_start_bin = bins[blank_start_bin_id]
            blank_end_bin = bins[blank_end_bin_id]
            tmp = (tmp < blank_start_bin)*1 + tmp 
            
        t[:,i] = tmp
        tmp = tmp - blank_end_bin
        tmp = tmp/tmp.max()
        normed_t[:,i] = tmp
        sort_t[:,i] = get_sort_positions(tmp)

    adata.layers['fit_t_shift'] = t.copy() # x, ..., x+1
    adata.layers['normed_t'] = normed_t.copy() # 0, ..., 1
    adata.layers['normed_t_sort'] = sort_t.copy() # 0, 1/2000, 2/2000 ..., 1
    adata.varm['hist_all'] = hist_all.copy()
    adata.varm['bins_all'] = bins_all.copy()
    adata.var['non_blank_gene'] = non_blank_gene.copy()
    #adata = adata[:, adata.var['non_blank_gene']==0] 
    return adata




def main(args):
    adata = ad.read_h5ad(args.data_path+"rc.h5ad") 
    adata.var_names_make_unique()
    check_data_type(adata)
    print(adata)

    losses = adata.varm['loss'].copy()
    losses[np.isnan(losses)] = 1e6
    adata.var['min_loss'] = losses.min(1)

    n_cells = adata.shape[0]
    expanded_scaling_y = np.expand_dims(np.array(adata.var['fit_scaling_y']),0).repeat(n_cells,axis=0)
    adata.layers['velocity'] = adata.layers['velo_hat'] / expanded_scaling_y  

    if 'X_pca' not in adata.obsm.keys():
        print('PCA ing')
        sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
    if (args.basis=='umap') and ('X_umap' not in adata.obsm.keys()):
        print('Umap ing')
        if args.dataset_name == 'hesc1':
            sc.tl.pca(adata, n_comps=50, svd_solver='arpack')
            sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=30, n_pcs=5)
            sc.tl.umap(adata)
        else:
            sc.tl.umap(adata)  
            sc.pl.umap(adata, color='clusters', save=True)
    
    adata = get_pseudotime(adata)
    
    adata_copy = adata.copy()
    adata_copy = get_sort_t(adata_copy) 

    adata_copy_1 = adata_copy.copy()
    data_type_tostr(adata_copy_1)
    print(adata_copy_1)
    adata_copy_1.write(args.data_path + 'rc.h5ad')

    thres_loss = np.percentile(adata_copy.var['min_loss'], args.loss_percent_thres) 
    adata_copy = adata_copy[:, adata_copy.var['min_loss'] < thres_loss]

    thres_n_cells = adata_copy.X.shape[0] * 0.1
    adata_copy = adata_copy[:, adata_copy.var['n_cells'] > thres_n_cells]
    
    adata_copy = adata_copy[:, adata_copy.var['non_blank_gene']==0] 


    adata_copy = get_metric_pseudotime(adata_copy)
    adata_copy = adata_copy[:, adata_copy.var['spearmanr_pseudotime'] > args.spearmanr_thres] 

    TFv.tl.velocity_graph(adata_copy, basis=None, vkey='velocity', xkey='M_total')
    adata_copy.uns['clusters_colors'] = adata.uns['clusters_colors']
    show_adata(args, adata_copy, save_name='velo', show_all=1)


    data_type_tostr(adata_copy)
    print(adata_copy)
    adata_copy.write(args.data_path + 'TFvelo.h5ad')

    return




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="pancreas", help='pancreas, gastrulation_erythroid, 10x_mouse_brain, hesc1') 
    parser.add_argument( '--layer', type=str, default="M_total", help='M_total, total') 
    parser.add_argument( '--basis', type=str, default="umap", help='umap, tsne, pca')
    parser.add_argument( '--loss_percent_thres', type=int, default=50, help='max loss of each gene')
    parser.add_argument( '--spearmanr_thres', type=float, default=0.8, help='min spearmanr')
    parser.add_argument( '--save_name', type=str, default='_demo', help='save_name')
    args = parser.parse_args() 
    
    args.data_path = 'TFvelo_'+ args.dataset_name + args.save_name+ '/'
    print('------------------------------------------------------------')   

    print(args) 
    main(args)
