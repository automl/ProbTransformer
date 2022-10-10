# Probabilistic Transformer
### *Modelling Ambiguities and Distributions for RNA Folding  and Molecule Design*
____

This repository contains the source code to the NeurIPS 2022 paper 
*Probabilistic Transformer: Modelling Ambiguities and Distributions for RNA Folding and Molecule Design*

[Paper on arXiv](https://arxiv.org/abs/2205.13927)

## Structure of the repository

##### *configs*
Contains the configuration files for our experiments on the Synthetic Sequential Distribution, RNA
 folding and molecule design task we reported in the paper.

##### *data*
Contains training, validation and test data for the RNA folding and molecule design task. 
We use the processed Guacamol dataset from https://github.com/devalab/molgpt and created the RNA folding data based on the description in the paper.

##### *prob_transformer*
Contains the source code of the ProbTransformer, the data handler and the training script `train_transformer.py`. 
The train script runs out of the box on a downscaled config and creates an *experiments* folder in the base directory.  


## Install conda environment 

Please adjust the cuda toolkit version in the `environment.yml` file to fit your setup. 
```
conda env create -n myenv -f environment.yml
conda activate myenv
pip install -e .
```

## Prepare data

#### RNA data
```
tar -xf data/rna_data.plk.xz -C data/
```

#### Molecule data

Download the [Guacamol dataset](https://drive.google.com/file/d/1gOSoKyGoYVdxtvy5cH2GNVDpLibk0lkS/view?usp=sharing) and extract  into `data`.
```
unzip data/guacamol2.csv.zip -d data/
```



## Train a ProbTransformer/Transformer model  
##### on the Synthetic Sequential Distribution Task
```
python prob_transformer/train_transformer.py -c configs/ssd_prob_transformer.yml
python prob_transformer/train_transformer.py -c configs/ssd_transformer.yml
```
##### on the RNA folding Task
```
python prob_transformer/train_transformer.py -c configs/rna_prob_transformer.yml
python prob_transformer/train_transformer.py -c configs/rna_transformer.yml
```
##### on the Molecule Design Task
```
python prob_transformer/train_transformer.py -c configs/mol_prob_transformer.yml
python prob_transformer/train_transformer.py -c configs/mol_transformer.yml
```