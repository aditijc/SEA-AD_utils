import os
import numpy as np
import anndata
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import scanpy as sc
import json
import scvi
import glob
import seaborn as sns
import copy
import random
import scipy.sparse as sp_sparse
import scipy.stats as sp_stats
from datetime import datetime
from joblib import parallel_backend
from joblib import Parallel, delayed
from igraph import *
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as scores
import pickle as pkl
import argparse
import wandb

import yaml
import time

def init_params():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_file", type=str, help="config file",
        default="../params.yaml",
    )
    params, _ = parser.parse_known_args()
    return params

def load_balanced_adata(params):
    adata = sc.read_h5ad(filename=params['data']['balanced_path'])

    ### Subset genes to 4000
    adata.obs["celltype"] = adata.obs["Supertype"]
    adata.obs["batch"] = adata.obs["Donor ID"]

    num_types = adata.obs["celltype"].unique().size
    # id2type = dict(enumerate(adata.obs["celltype"].cat.categories))
    celltypes = adata.obs["celltype"].unique()
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    adata.obs["celltype_id"] = celltype_id_labels
    adata.obs["batch_id"] = adata.obs["batch"].cat.codes.values
    adata.var["gene_name"] = adata.var.index.tolist()

    # get high variance genes
    sc.pp.highly_variable_genes(adata, n_top_genes=4000, flavor="cell_ranger", batch_key="batch")
    adata = adata[:, adata.var["highly_variable"]]

    return adata

def create_train_test_adata(adata, perc_train=0.8, splitby="Supertype", seed=42, output_train_path=None, output_val_path=None):
    """
    Splits an AnnData object into training and validation sets based on the specified category.
    Designed for SEA-AD dataset.

    Parameters:
    adata (anndata.AnnData): The input AnnData object containing single-cell data.
    perc_train (float, optional): The proportion of cells to include in the training set (default is 0.8).
    splitby (str, optional): The column name in adata.obs to split by (default is "Supertype").
    seed (int, optional): The random seed for reproducibility (default is 42).

    Returns:
    train_adata (anndata.AnnData): The training set AnnData object.
    val_adata (anndata.AnnData): The validation set AnnData object.
    """
    train_indices = []
    val_indices = []

    if seed is not None:
        np.random.seed(seed)

    # create random split for each category
    for category in adata.obs[splitby].unique():
        adata_category = adata.obs[adata.obs[splitby] == category]
        num_train = int(perc_train*len(adata_category))
        indices = list(adata_category.index)
        np.random.shuffle(indices)
        train_indices.extend(indices[:num_train])
        val_indices.extend(indices[num_train:])

    train_adata = adata[train_indices, :].copy()
    val_adata = adata[val_indices, :].copy()

    print("Train size:", train_adata.shape)
    print("Validation size:", val_adata.shape)

    if output_train_path is not None: train_adata.write_h5ad(filename=output_train_path)
    if output_val_path is not None: val_adata.write_h5ad(filename=output_val_path)

    return train_adata, val_adata, train_indices, val_indices

def refine_subclasses(adata):
    adata.obs["Subclass_refined"] = adata.obs["Subclass"]
    adata.obs["Subclass_refined"] = adata.obs["Subclass_refined"].cat.add_categories(["Astrocyte_A", "Astrocyte_B", "Astrocyte_C", "Microglia-PVM_A", "Microglia-PVM_B", "Microglia-PVM_C", "OPC_A", "OPC_B", "Oligodendrocyte_A", "Oligodendrocyte_B"])

    # Divide Astrocyte subclass
    adata.obs.loc[adata.obs["Supertype"] == "Astro_1", "Subclass_refined"] = "Astrocyte_A"
    adata.obs.loc[adata.obs["Supertype"] == "Astro_3", "Subclass_refined"] = "Astrocyte_A"
    adata.obs.loc[adata.obs["Supertype"] == "Astro_2", "Subclass_refined"] = "Astrocyte_B"
    adata.obs.loc[adata.obs["Supertype"] == "Astro_4", "Subclass_refined"] = "Astrocyte_B"
    adata.obs.loc[adata.obs["Supertype"] == "Astro_5", "Subclass_refined"] = "Astrocyte_C"
    adata.obs.loc[adata.obs["Supertype"] == "Astro_6-SEAAD", "Subclass_refined"] = "Astrocyte_C"
    adata.obs["Subclass_refined"] = adata.obs["Subclass_refined"].cat.remove_categories(["Astrocyte"])

    # Divide Microglia PVM subclass
    adata.obs.loc[adata.obs["Supertype"] == "Lymphocyte", "Subclass_refined"] = "Microglia-PVM_A"
    adata.obs.loc[adata.obs["Supertype"] == "Micro-PVM_1", "Subclass_refined"] = "Microglia-PVM_A"
    adata.obs.loc[adata.obs["Supertype"] == "Micro-PVM_2", "Subclass_refined"] = "Microglia-PVM_A"
    adata.obs.loc[adata.obs["Supertype"] == "Micro-PVM_2_1-SEAAD", "Subclass_refined"] = "Microglia-PVM_B"
    adata.obs.loc[adata.obs["Supertype"] == "Micro-PVM_2_3-SEAAD", "Subclass_refined"] = "Microglia-PVM_B"
    adata.obs.loc[adata.obs["Supertype"] == "Micro-PVM_3-SEAAD", "Subclass_refined"] = "Microglia-PVM_C"
    adata.obs.loc[adata.obs["Supertype"] == "Micro-PVM_4-SEAAD", "Subclass_refined"] = "Microglia-PVM_C"
    adata.obs.loc[adata.obs["Supertype"] == "Monocyte", "Subclass_refined"] = "Microglia-PVM_C"
    adata.obs["Subclass_refined"] = adata.obs["Subclass_refined"].cat.remove_categories(["Microglia-PVM"])

    # Divide OPC subclass
    adata.obs.loc[adata.obs["Supertype"] == "OPC_1", "Subclass_refined"] = "OPC_A"
    adata.obs.loc[adata.obs["Supertype"] == "OPC_2", "Subclass_refined"] = "OPC_A"
    adata.obs.loc[adata.obs["Supertype"] == "OPC_2_1-SEAAD", "Subclass_refined"] = "OPC_B"
    adata.obs.loc[adata.obs["Supertype"] == "OPC_2_2-SEAAD", "Subclass_refined"] = "OPC_B"
    adata.obs["Subclass_refined"] = adata.obs["Subclass_refined"].cat.remove_categories(["OPC"])

    # Divide Oligodendrocyte subclass
    adata.obs.loc[adata.obs["Supertype"] == "Oligo_1", "Subclass_refined"] = "Oligodendrocyte_A"
    adata.obs.loc[adata.obs["Supertype"] == "Oligo_2", "Subclass_refined"] = "Oligodendrocyte_A"
    adata.obs.loc[adata.obs["Supertype"] == "Oligo_2_1-SEAAD", "Subclass_refined"] = "Oligodendrocyte_B"
    adata.obs.loc[adata.obs["Supertype"] == "Oligo_3", "Subclass_refined"] = "Oligodendrocyte_B"
    adata.obs.loc[adata.obs["Supertype"] == "Oligo_4", "Subclass_refined"] = "Oligodendrocyte_B"
    adata.obs["Subclass_refined"] = adata.obs["Subclass_refined"].cat.remove_categories(["Oligodendrocyte"])
    
    return adata

def create_grouping(subclass, supertype, adata, params):
    grouping = adata.obs.loc[:, [subclass, supertype]].drop_duplicates().sort_values(by=supertype)
    grouping[subclass] = grouping[subclass].astype("category")
    grouping["subclass_codes"] = grouping[subclass].cat.codes
    grouping = grouping["subclass_codes"].to_numpy()
    random_grouping = np.random.choice(grouping, len(grouping), replace=False)
    np.save(os.path.join(params["results"]["runfiles"], "input", "random_subclass_grouping.npy"), random_grouping)
    adata.obs["supertypes_scrambled"] = np.random.choice(adata.obs[supertype], len(adata.obs[supertype]), replace=False)
    adata.obs["supertypes_scrambled"].to_csv(os.path.join(params["results"]["runfiles"], "input", "scrambled_supertypes.csv"))
    return grouping, adata

# adata.obs.loc[adata.obs["Subclass_refined"] == "Oligodendrocyte_B", "Supertype"].value_counts()[:10]
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ## Init Params/ Cell Model
    print("Initializing Params.")
    config_file = init_params().config_file

    with open(config_file, 'r') as file:
        params = yaml.safe_load(file)
    import sys
    sys.path.insert(0, params['cell_path']) # Path to cell-tools
    import cell_tools.model as cellmodel # import scvi_self.model as scvimodel
    import cell_tools.run as cellrun # import scvi_self.run as scvirun
    import cell_tools.tool as celltl # import scvi_self.tool as scvitl

    print('Loading Data.')
    adata = load_balanced_adata(params)

    ## Create hierarchical grouping
    grouping = None
    print('Grouping Data.')
    refine_subclasses = False
    if params["model"]["cell_grouping"] or params["model"]["blast_grouping"]:
        if refine_subclasses:
            adata = refine_subclasses(adata)
            grouping, adata = create_grouping("Subclass_refined", "Supertype", adata, params)
        else: grouping, adata = create_grouping("Subclass", "Supertype", adata, params)
    print('Creating Train/ Test Splits.')
    train_adata, val_adata, train_indices, val_indices = create_train_test_adata(adata)

    ## RUN CELL 
    print("Training Cell.")
    wandb_args_cell = {'project': 'CELL_tests', 'entity': 'aditijc', 'group': params["model"]["cellname"]}
    model_path = params["results"]["model_results"]
    cell_model_path = os.path.join(model_path, params["model"]["cellname"])
    batch_size = params["model"]["cell_batch_size"]
    max_epochs = params["model"]["cell_max_epochs"]
    cell_model, cell_kwargs = cellrun.run_CELL(adata, 
                                               cell_model_path, 
                                               batch_key="batch", # previously method
                                               categorical_covariate_keys=["library_prep"],
                                               batch_size=batch_size,
                                               max_epochs=max_epochs,
                                               adversarial_use_focal=params["model"]["cell_focal"],
                                               focal_schedule=params["model"]["cell_focal_schedule"],
                                               use_self_contrastive=params["model"]["cell_contrastive"],
                                               labels_key="celltype",
                                               use_labels_groups=params["model"]["cell_grouping"],
                                               labels_groups=grouping,
                                               wandb_args=wandb_args_cell,
                                            )
    wandb.finish()

    # ## Plot Cell Latents
    # print("Plotting Cell Latents.")
    # if os.path.exists(os.path.join(cell_model_path, "CELL.h5ad")) and os.path.exists(os.path.join(cell_model_path, 'CELL_umap.pkl')):
    #         print("CELL model has been clustered before!" )
    #         adata_CELL = sc.read_h5ad(filename=os.path.join(cell_model_path, "CELL.h5ad"))
    #         umap_model = pkl.load(open(os.path.join(cell_model_path, 'CELL_umap.pkl'),'rb'))
    # else:
    #         cell_model = cellmodel.CELL.load(cell_model_path, adata)
    #         adata_CELL = adata.copy()
    #         latent = cell_model.get_latent_representation(adata_CELL)
    #         adata_CELL.obsm["CELL"] = latent
    #         sc.pp.neighbors(adata_CELL, use_rep="CELL") # Using latent representation to form neighbors
    #         _, umap_model = celltl.umap(adata_CELL, min_dist=0.3, method='rapids')
    #         adata_CELL.write_h5ad(filename=os.path.join(cell_model_path, "CELL.h5ad"))
    #         pkl.dump(umap_model, open(os.path.join(cell_model_path, 'CELL_umap.pkl'), 'wb'))

    # sc.pl.umap(
    #         adata_CELL,
    #         color=["Supertype"], 
    #         palette = list(mpl.colors.CSS4_COLORS.values()), 
    #         frameon=False,
    #         title="CELL Trained"
    #         ) # show the cell before classification


    ## Create Train/ test col
    print("Creating train/ test split.")
    adata.obs["train"] = False
    adata.obs.loc[train_indices, "train"] = True
    adata.obs["train"] = adata.obs["train"].astype('category')

    ## RUN CELLBLAST
    print("Training Cellblast.")
    wandb_args_cellblast = {'project': 'CELLBLAST_tests', 'entity': 'aditijc', 'group': params["model"]["blastname"]}
    
    label_model_path = os.path.join(model_path, params["model"]["blastname"])
    blast_kwargs = cellrun.run_CELLBLAST(
            adata=adata,
            model_dir = cell_model_path,
            label_model_dir = label_model_path,
            CELL_kwargs = cell_kwargs,
            batch_key="batch", # previously method
            categorical_covariate_keys=["library_prep"],
            splitby="train",
            labels_key="celltype",
            cellblast_max_epochs = params["model"]["cellblast_max_epochs"],
            cellblast_batch_size = params["model"]["cellblast_batch_size"],
            verbose=True,
            use_labels_groups=params["model"]["blast_grouping"],
            labels_groups=grouping,
            wandb_args_cellblast=wandb_args_cellblast,
            use_contrastive_classification=params["model"]["blast_contrastive"], # Change to use contrastive classification
            use_focal_classification=params["model"]["blast_focal_classification"],
            cellblast_adversarial_use_focal=params["model"]["blast_focal"],
            focal_schedule=params["model"]["blast_focal_schedule"]
        )

    ## Plotting CellBlast latents
    print('Plotting blast latents.')
    label_model_path_full = label_model_path+"_0"
    if os.path.exists(os.path.join(label_model_path_full, "CELLBLAST.h5ad")) and os.path.exists(os.path.join(label_model_path_full, 'CELLBLAST_umap.pkl')):
            print("BLAST model has been clustered before!" )
            adata_CELLBLAST = sc.read_h5ad(filename=os.path.join(label_model_path_full, "CELLBLAST.h5ad"))
            umap_label = pkl.load(open(os.path.join(label_model_path_full, 'CELLBLAST_umap.pkl'),'rb'))
    else:
            label_model = cellmodel.CELLBLAST.load(label_model_path_full, adata)
            print(label_model)
            adata_CELLBLAST = adata.copy()
            latent_CELLBLAST = label_model.get_latent_representation(adata_CELLBLAST)
            adata_CELLBLAST.obsm["CELLBLAST"] = latent_CELLBLAST
            sc.pp.neighbors(adata_CELLBLAST, use_rep="CELLBLAST") # Using latent representation to form neighbors
            _, umap_label = celltl.umap(adata_CELLBLAST, min_dist=0.3, method='rapids')
            adata_CELLBLAST.write_h5ad(filename=os.path.join(label_model_path_full, "CELLBLAST.h5ad"))
            pkl.dump(umap_label, open(os.path.join(label_model_path_full, 'CELLBLAST_umap.pkl'), 'wb'))

    sc.pl.umap(
            adata_CELLBLAST,
            color=["Supertype"], 
            palette = list(mpl.colors.CSS4_COLORS.values()), 
            frameon=False,
            title="CELLBLAST Trained"
            ) # show the cell after cellblast classification

    ## Metrics
    print("Computing metrics.")
    models = [
        {
            "model_dir" : cell_model_path,
            "label_model_dir" : label_model_path,
            "CELL_kwargs" : cell_kwargs,
            "batch_key" : "method",
            "categorical_covariate_keys": ["library_prep"],
            "splitby": "train",
            "labels_key": "Supertype",
            "cellblast_max_epochs": 10,
            "cellblast_batch_size": 128
        }
    ]
    results, metrics = celltl.compile_results(
        adata,
        "Supertype",
        models=models
    )
    metrics = pd.DataFrame(metrics)
    print(f'Mean f1: {np.mean(metrics["f1"])}')