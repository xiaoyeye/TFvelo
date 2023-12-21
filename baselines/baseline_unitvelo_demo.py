import unitvelo as utv
import scvelo as scv
import scanpy as sc
import anndata as ad

velo = utv.config.Configuration()
velo.R2_ADJUST = True 
velo.IROOT = None
velo.FIT_OPTION = '1'
velo.GPU = 1



def main(args):
    if args.dataset_name == 'pancreas':
        path_to_adata = 'data/Pancreas/endocrinogenesis_day15.h5ad'
        label = 'clusters'
    elif args.dataset_name == 'gastrulation_erythroid':
        path_to_adata = 'data/Gastrulation/erythroid_lineage.h5ad'
        label = 'celltype'

    adata = utv.run_model(path_to_adata, label, config_file=velo)
    
    scv.pp.neighbors(adata)

    adata.write(args.result_path+'unitvelo.h5ad')

    #subvar = adata.var.loc[adata.var['velocity_genes'] == True]
    #sub = adata[:, subvar.index]

    return

def analysis(args):
    adata_TFv = ad.read_h5ad(args.result_path+'TFvelo.h5ad') 
    n_colors = len(adata_TFv.obs['clusters'].cat.categories)
    adata_TFv.uns['clusters_colors'] = adata_TFv.uns['clusters_colors'][:n_colors]

    method = 'unitvelo'
    adata = ad.read_h5ad(args.result_path+method+'.h5ad')
    adata.uns['clusters_colors'] = adata_TFv.uns['clusters_colors'].copy()
    adata.uns['label'] = 'clusters'

    if args.dataset_name == 'pancreas':
        label = 'clusters'
    elif args.dataset_name == 'gastrulation_erythroid':
        adata.obs['clusters'] = adata.obs['celltype'].copy()
    elif args.dataset_name == 'pons':
        adata.obs['clusters'] = adata.obs['celltype'].copy()
    
    scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot', fontsize=20, save=args.dataset_name+'_'+method+'_pseudotime.png')
    scv.pl.velocity_embedding_stream(adata, color='clusters', dpi=300, title='', save= args.dataset_name+'_'+method+'_embedding_stream.png')
    scv.pl.velocity_embedding_grid(adata, color='clusters', dpi=300,  arrow_size=10, title='', save= args.dataset_name+'_'+method+'_embedding_grid.png')
    adata.write(args.result_path+'unitvelo.h5ad')

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="gastrulation_erythroid", help='pancreas, gastrulation_erythroid, pons') 
    parser.add_argument( '--save_name', type=str, default='_demo', help='save_name')
    args = parser.parse_args() 

    args.result_path = 'TFvelo_'+ args.dataset_name + args.save_name+ '/'
    print('------------------------------------------------------------')   
    print(args)   
    #main(args)
    analysis(args)

