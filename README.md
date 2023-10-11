# TFvelo


This is the code of TFvelo, which is developed based on the implementation of scVelo. 

![Image text](https://github.com/xiaoyeye/TFvelo/blob/main/figures/demo.png)


TFvelo_run_demo.py provides the code for runing TFvelo, and TFvelo_analysis_demo.py is for results visualization. These code can be directly downloaded for usage.


Environment:
```
conda create -n TFvelo_env python=3.8.12
conda activate TFvelo_env
pip install pandas==1.2.3 
pip install anndata==0.8.0 
pip install scanpy==1.8.2
pip install numpy==1.21.6
pip install scipy==1.10.1 
pip install numba==0.57.0 
pip install matplotlib==3.3.4
pip install scvelo==0.2.4
pip install typing_extensions
```

Reproduce:
To reproduce TFvelo on pancreas:
```
python TFvelo_run_demo.py --dataset_name pancreas
```
This will automatically download, preprocess and run TF model on pancrease dataset. The result will be stored in 'TFvelo_pancreas/rc.h5ad'.


After that, the visualization of results can be obtained by 
```
python TFvelo_analysis_demo.py --dataset_name pancreas
```
The result will be stored in 'TFvelo_pancreas_demo/TFvelo.h5ad', and figures will be put in folder 'figures'.

Running the program with default parameters can reproduce the results in manuscript.


To apply TFvelo to other single cell data:
you can define a personalized name for the dataset, and simply add the following codes into the preprocess() function:
```
if args.dataset_name == your_dataset_name:
  adata = ad.read_h5ad(your_h5ad_file_path)   
```
Then run the code with:
```
python TFvelo_run_demo.py --dataset_name your_dataset_name
python TFvelo_analysis_demo.py --dataset_name your_dataset_name
```
As a result, all generated h5ad files will be puted in the folder named: "TFvelo_"+your_data_name+"_demo". And figures will be put in the folder "figures".

Hyperparameters:
--n_jobs: number of cpus to use
--init_weight_method: the method to initialize the weights. Correlation is adopted by default.
--WX_method: the method to optimize weight. lsq_linear is adopted by default.
--n_neighbors: number of neighbors.
--WX_thres: The max absolute value for weights.
--TF_databases: The way to select candidate TFs. use ENCODE and ChEA by default.
--max_n_TF: max number of TFs used for modeling each gene.
--max_iter: max number of iterations in the generalized EM algorithm.
--n_time_points: the number of time points in the time assinment (E step of the generalized EM algorithm). 
--save_name: the name of folder which all generated files will be put in.
