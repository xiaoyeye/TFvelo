# TFvelo


This is the code of TFvelo, which is developed based on the implementation of scVelo. TFvelo.py provides the code for runing TFvelo, and TFvelo_analysis.py is for results visualization. These code can be directly downloaded for usage and being further developed without installation.


Dependencies:
pandas,
anndata,
scanpy,
numpy,
scikit-learn,
matplotlib.


The demo dataset Pancreas can be load by:
```
import TFvelo as TFv
adata = TFv.datasets.pancreas()
```

To test TFvelo on pancreas:
```
python TFvelo.py --dataset_name pancreas
```
This will automatically download, preprocess and run TF model on pancrease dataset. These processes may take 1-2 hours on a linux server with more than 16 CPUs. The result will be stored in 'TFvelo_pancreas/rc.h5ad'.


After that, the visualization of results can be obtained by 
```
python TFvelo_analysis.py --dataset_name pancreas
```
The result will be stored in 'TFvelo_pancreas/analysis.h5ad', and figures will be put in folder 'figures'.

Running the program with default parameters can reproduce the results in manuscript.
