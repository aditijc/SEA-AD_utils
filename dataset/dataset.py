import anndata as ad
import numpy as np

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

    return train_adata, val_adata