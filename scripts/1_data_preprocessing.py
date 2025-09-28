#!/usr/bin/env python3
"""
data preprocessing
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io as sio
from scipy.sparse import csr_matrix
import os
from pathlib import Path
import argparse


def prepare_gmvae_data(input_h5ad, output_dir, subsample_fraction=None):
    """
    prepare data for GMVAE training

    args:
        input_h5ad: Path to input h5ad file
        output_dir: Output directory for processed files
        subsample_fraction: Optional fraction to subsample data
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # from: ProtoCell4P/src/load_data.py
    # og: adata = sc.read_h5ad(data_path)
    print(f"loading data from: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    print(f"loaded data: {adata.n_obs} cells x {adata.n_vars} genes")

    # subsampling (new)
    if subsample_fraction is not None:
        n_cells = adata.shape[0]
        n_subsample = int(n_cells * subsample_fraction)
        np.random.seed(42)
        indices = np.random.choice(n_cells, n_subsample, replace=False)
        adata = adata[indices, :]
        print(f"subsampled to {subsample_fraction*100}%: {n_subsample} cells from {n_cells}")

    # from: ProtoCell4P/src/load_data.py line 31-32
    # og: sc.pp.filter_genes(adata, min_cells=5)
    print(f"before filtering: {adata.shape[1]} genes")
    sc.pp.filter_genes(adata, min_cells=5)
    print(f"after filtering (min_cells=5): {adata.shape[1]} genes")

    # from: ProtoCell4P/src/load_data.py line 33
    # og: sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("Normalized to target_sum=1e4")

    # from: ProtoCell4P/src/load_data.py lines 36-37
    # og: if keep_sparse is False: adata.X = adata.X.toarray()
    # keeping sparse format (keep_sparse=True equivalent)
    X = adata.X

    # convert to genes x cells format for GMVAE
    X_transposed = X.T.tocsr()
    sio.mmwrite(os.path.join(output_dir, "matrix.mtx"), X_transposed)

    # from: ProtoCell4P/src/load_data.py line 46
    # og: cell_types = adata.obs["ct_cov"]
    if "ct_cov" in adata.obs.columns:
        cell_types = adata.obs["ct_cov"]
    elif "cell_type" in adata.obs.columns:
        cell_types = adata.obs["cell_type"]
    else:
        raise ValueError("no cell type column found. expected 'ct_cov' or 'cell_type'")

    # extract patient information
    if "ind_cov" in adata.obs.columns:
        patient_ids = adata.obs["ind_cov"]
    else:
        raise ValueError("no patient ID column found. expected 'ind_cov'")

    if "disease_cov" in adata.obs.columns:
        disease_labels = adata.obs["disease_cov"]
    else:
        raise ValueError("no disease label column found. expected 'disease_cov'")

    # from: ProtoCell4P/src/load_data.py lines 48-49
    # og: ct_id = sorted(set(cell_types))
    # og: mapping_ct = {c:idx for idx, c in enumerate(ct_id)}
    ct_id = sorted(set(cell_types))
    mapping_ct = {c: idx for idx, c in enumerate(ct_id)}
    cell_type_codes = [mapping_ct[ct] for ct in cell_types]

    # create disease label mapping
    disease_id = sorted(set(disease_labels))
    mapping_disease = {d: idx for idx, d in enumerate(disease_id)}
    disease_codes = [mapping_disease[d] for d in disease_labels]

    # save all labels
    labels_df = pd.DataFrame({
        'cluster': cell_type_codes,
        'patient_id': patient_ids,
        'disease': disease_codes
    })
    labels_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

    # from: ProtoCell4P/src/load_data.py line 44
    # og: genes = adata.var_names.tolist()
    genes = adata.var_names.tolist()
    with open(os.path.join(output_dir, "genes.txt"), 'w') as f:
        for gene in genes:
            f.write(f"{gene}\n")

    # save processed adata
    adata.write(os.path.join(output_dir, "processed_data.h5ad"))

    # save metadata
    metadata = {
        'n_cells': X_transposed.shape[1],
        'n_genes': X_transposed.shape[0],
        'n_cell_types': len(np.unique(cell_type_codes)),
        'cell_types': list(pd.Categorical(cell_types).categories),
        'n_patients': len(np.unique(patient_ids)),
        'patients': list(pd.Categorical(patient_ids).categories),
        'n_diseases': len(np.unique(disease_codes)),
        'diseases': list(pd.Categorical(disease_labels).categories),
        'disease_mapping': mapping_disease,
        'subsample_fraction': subsample_fraction
    }

    import json
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"prepared data: {X_transposed.shape[0]} genes x {X_transposed.shape[1]} cells")
    print(f"cell types: {len(np.unique(cell_type_codes))}")
    print(f"files saved to: {output_dir}")
    print("generated files:")
    print("matrix.mtx (genes x cells)")
    print("labels.csv (cell type labels)")
    print("genes.txt (gene names)")
    print("processed_data.h5ad (AnnData object)")
    print("metadata.json (dataset info)")

    return adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for GMVAE-4P')

    # from: ProtoCell4P/src/load_data.py line 25
    # og: "../data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad"
    parser.add_argument('--input', type=str, required=True,
                       help='input h5ad file path')
    parser.add_argument('--output', type=str, required=True,
                       help='output directory for processed data')
    parser.add_argument('--subsample', type=float, default=None,
                       help='fraction to subsample (e.g., 0.1 for 10%)')

    args = parser.parse_args()

    print("=" * 60)
    print("data preprocessing")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    if args.subsample:
        print(f"subsampling: {args.subsample*100}%")
    print()

    prepare_gmvae_data(args.input, args.output, subsample_fraction=args.subsample)

    print("\ndata preprocessed")
