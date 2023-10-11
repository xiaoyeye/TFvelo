# TFvelo


This is the code of TFvelo, which is developed based on the implementation of scVelo. TFvelo.py provides the code for runing TFvelo, and TFvelo_analysis.py is for results visualization. These code can be directly downloaded for usage and being further developed without installation.


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
The result will be stored in 'TFvelo_pancreas/TFvelo.h5ad', and figures will be put in folder 'figures'.

Running the program with default parameters can reproduce the results in manuscript.


To apply TFvelo to other single cell data:
you can use a personalized name for the dataset by :
--dataset_name your_data_name 

and simply change the data loader in preprocess() function as:
adata = ad.read_h5ad(h5ad_file_path)   

As a result, all results will be puted in the folder $"TFvelo_"+your_data_name+"_demo"$
