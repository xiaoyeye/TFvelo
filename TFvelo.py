import pandas as pd    
import TFvelo as TFv
import anndata as ad
import numpy as np
import scanpy as sc

import matplotlib
matplotlib.use('AGG')

import os, sys
rootPath = os.path.dirname(sys.path[0])
os.chdir(rootPath+'/RNA_velocity')

def check_data_type(adata):
    for key in list(adata.var):
        if adata.var[key][0] in ['True', 'False']:
            adata.var[key] = adata.var[key].map({'True': True, 'False': False})
    return          

def data_type_tostr(adata, key):
    #for key in list(adata.var):
    if key in adata.var.keys():
        if adata.var[key][0] in [True, False]:
            adata.var[key] = adata.var[key].map({True: 'True', False:'False'})
    return          

def batch_correction(adata, batch_corr_method, batch_keys):
    if batch_corr_method == 'bbknn':
        print('PCA')
        sc.tl.pca(adata, svd_solver='arpack')
        adata.obsm['X_pca']
        adata.varm['PCs']
        print('batch correction with bbknn')
        sc.external.pp.bbknn(adata, batch_key='batch')
    elif batch_corr_method == 'scanorama': 
        print('batch correction with scanorama')           
        adata_list = []
        for batch in batch_keys:
            adata_batch = adata[adata.obs['batch']==batch] #sc.read(args.save_path+'/Day'+str(Day)+'/rna.h5ad')
            #print(rna_Day)
            adata_list.append(adata_batch)
        import scanorama
        corrected_adata_list = scanorama.correct_scanpy(adata_list)
        corrected_adata = corrected_adata_list[0].concatenate(corrected_adata_list[1:])
        adata.X = corrected_adata.X

    elif batch_corr_method == 'harmony':
        print('correcting rna with harmony')
        from harmony import harmonize
        adata.X = harmonize(adata.X, adata.obs, batch_key = 'Days')
        #adata.raw = adata
        #sc.pp.scale(adata, max_value=10)
        #sc.tl.pca(adata)
        #Z = harmonize(adata.obsm['X_pca'], adata.obs, batch_key = 'Days')
        #adata.obsm['X_harmony'] = Z
        #sc.pp.neighbors(adata, use_rep="X_harmony", n_neighbors=15, n_pcs=30)
    return adata

def preprocessing(args):
    print('----------------------------------Preprocessing',args.dataset_name,'---------------------------------------------')
    if args.dataset_name == 'pancreas':
        adata = ad.read_h5ad("data/pancreas/endocrinogenesis_day15.h5ad") #adata = TFv.datasets.pancreas()

    elif args.dataset_name == 'gastrulation_erythroid':
        adata = TFv.datasets.gastrulation_erythroid()   
        adata.uns['clusters_colors'] = adata.uns['celltype_colors'].copy()
        adata.obs['clusters'] = adata.obs['celltype'].copy()

    elif args.dataset_name == 'merfish':
        all_data = np.loadtxt('data/merfish/pnas.1912459116.sd12.csv', str, delimiter = ",")
        all_features = all_data[1:, 1: ]
        all_features = all_features.astype(np.float)
        all_features = np.swapaxes(all_features, 0, 1)
        print('feature shape: ', all_features.shape)
        adata = ad.AnnData(all_features)
        adata.var_names = all_data[1:, 0]
        adata.obs_names = all_data[0, 1:]
        all_batch_info = []

        gene_names = adata.var_names
        for gene_name in gene_names:
            if 'Blank' in gene_name:
                adata = adata[:,adata.var_names!=gene_name]

        for cell_name in adata.obs_names:
            if 'B1' in cell_name:
                all_batch_info.append('1')
            elif 'B2' in cell_name:
                all_batch_info.append('2')
            elif 'B3' in cell_name:
                all_batch_info.append('3')
        adata.obs['batch'] = np.array(all_batch_info)

        info = np.loadtxt('data/merfish/types.txt', dtype=int)
        embed = np.load('data/merfish/lambdaI0.8_epoch5000_Embed_X.npy')
        adata.obsm['X_embed'] = embed

        cell_clusters_idx = info[:,2]
        cell_clusters_map = np.array(['M', 'S', 'NA', 'G2', 'G1'])
        cell_clusters = cell_clusters_map[cell_clusters_idx]
        adata.obs['clusters'] = pd.CategoricalIndex(list(cell_clusters), categories=["G1", "S", "G2", "M", 'NA'])
        
        adata.obsm['position'] = np.zeros([adata.shape[0], 2])
        import openpyxl
        workbook = openpyxl.load_workbook('data/merfish/pnas.1912459116.sd15.xlsx')
        tmp = 0
        for ii, sheet_name in enumerate(['Batch 1', 'Batch 2', 'Batch 3']):
            worksheet = workbook[sheet_name]
            sheet_X = [item.value for item in list(worksheet.columns)[1]]
            sheet_X = sheet_X[1:]
            sheet_Y = [item.value for item in list(worksheet.columns)[2]]
            sheet_Y = sheet_Y[1:]
            tmp_ = len(sheet_X) 
            adata.obsm['position'][tmp:tmp+tmp_, 0] = np.array(sheet_X).astype(np.float) + ii*5000
            adata.obsm['position'][tmp:tmp+tmp_, 1] = np.array(sheet_Y).astype(np.float)
            tmp += tmp_

        adata = adata[adata.obs['clusters']!='NA']
        adata = adata[adata.obs['clusters']!='M']

    elif args.dataset_name == 'hesc1':
        expression = pd.read_table("data/hesc1/rpkm.txt", header=0, index_col=0, sep="\t").T 
        adata = ad.AnnData(expression)
        adata.obs_names = expression.index
        adata.var_names = expression.columns
        adata.obs['time_gt'] = 'Nan'
        for ii, cell in enumerate(adata.obs_names):
            adata.obs['time_gt'][ii] = cell.split('.')[0]
        adata.obs['time_gt'] = adata.obs['time_gt'].astype('category') 
        adata.obs['clusters'] = adata.obs['time_gt'].copy()

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    np.savetxt(args.result_path+'all_genes.txt', np.array(adata.var_names), fmt='%s', delimiter='/t')

    if "spliced" in adata.layers:
        adata.layers["total"] = adata.layers["spliced"].todense() + adata.layers["unspliced"].todense()
    elif "new" in adata.layers:
        adata.layers["total"] = np.array(adata.layers["total"].todense())
    else:
        adata.layers["total"] = adata.X
    n_cells, n_genes = adata.X.shape
    if not args.dataset_name in ['pbmc68k']:
        sc.pp.filter_genes(adata, min_cells=int(n_cells/50))
        sc.pp.filter_cells(adata, min_genes=int(n_genes/50))
    TFv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000, log=True) #include the following steps
    adata.X = adata.layers["total"].copy()
    if args.dataset_name in ['merfish']:#'batch' in adata.obs_keys(): 
        adata = batch_correction(adata, batch_corr_method='scanorama', batch_keys=['1', '2', '3']) 
    adata.uns['clusters_colors'] = np.array(['red', 'orange', 'yellow', 'green','skyblue', 'blue','purple', 'pink', '#8fbc8f', '#f4a460', '#fdbf6f', '#ff7f00', '#b2df8a', '#1f78b4',
       '#6a3d9a', '#cab2d6'], dtype=object)
    print(adata)

    '''
    all_TF_names, TF_names, TF_idx = [], [], []
    with open("data/TF_names_v_1.01.txt", "r") as f:  # 打开文件
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            all_TF_names.append(line)
    f.close()
    '''

    gene_names = []
    for tmp in adata.var_names:
        gene_names.append(tmp.upper())
    adata.var_names = gene_names

    if args.dataset_name == 'merfish':
        sc.pp.neighbors(adata, use_rep="X_embed", n_neighbors=100)
        TFv.pp.moments(adata, n_pcs=30, n_neighbors=None)  
    else:
        TFv.pp.moments(adata, n_pcs=30, n_neighbors=30) 
    TFv.pp.get_TFs(adata, dataset='all', corr_check='all', thres=0.3) 

    adata.write(args.result_path + 'pp.h5ad')



def main(args):
    print('--------------------------------')
    adata = ad.read_h5ad(args.result_path + 'pp.h5ad')

    adata.layers['fit_t'] = np.zeros_like(adata.layers['total'])
    if os.cpu_count() < 16: # local
        n_jobs = 1
        args.n_top_genes = 2
        args.max_iter = 10
    else: # server
        if args.n_jobs >= 1:
            n_jobs = args.n_jobs
        else:
            n_jobs = int(os.cpu_count()/2-2)
    print('n_jobs:', n_jobs)
    flag = TFv.tl.recover_dynamics(adata, n_jobs=n_jobs, max_iter=args.max_iter, var_names=args.var_names,
        WX_method = args.WX_method, max_n_TF=args.max_n_TF, n_top_genes=args.n_top_genes,
        fit_scaling=True, use_raw=args.use_raw) ########################
    if flag==False:
        return adata, False
    if 'highly_variable_genes' in adata.var.keys():
        data_type_tostr(adata, key='highly_variable_genes')
    adata.write(args.result_path + 'rc.h5ad')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset_name', type=str, default="pancreas", help='pancreas, scNT_seq, merfish, bonemarrow') 
    parser.add_argument( '--preprocessing', type=int, default=0, help='preprocessing')
    parser.add_argument( '--n_jobs', type=int, default=16, help='n_jobs')
    parser.add_argument( '--var_names', type=str, default="all", help='all, highly_variable_genes')
    parser.add_argument( '--WX_method', type=str, default= "lsq_linear", help='Gaussian_Kernel, NN, LS, LASSO, Ridge, constant, LS_constrant, lsq_linear')
    parser.add_argument( '--n_top_genes', type=int, default=2000, help='n_top_genes')
    parser.add_argument( '--max_n_TF', type=int, default=99, help='max_n_TF')
    parser.add_argument( '--max_iter', type=int, default=20, help='max_iter in EM')
    parser.add_argument( '--save_name', type=str, default='', help='save_name')
    parser.add_argument( '--use_raw', type=int, default=0, help='use_raw')

    args = parser.parse_args() 
    args.result_path = 'TFvelo_'+ args.dataset_name + args.save_name+ '/'
    print('********************************************************************************************************')
    print('********************************************************************************************************')
    print(args)

    if args.preprocessing or (not os.path.exists(args.result_path+'pp.h5ad')):
        preprocessing(args)  
    #check_TF_target(args)
    main(args) 
  
