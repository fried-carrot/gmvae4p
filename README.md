# GMVAE-4P

Three-module implementation for GMVAE-4P patient classification from scRNA-seq data.

## Quick Start

```bash
# install dependencies
pip install -r requirements.txt

# run complete pipeline
chmod +x run.sh
./run.sh
```

## Three Modules

### 1. Data Preprocessing (`scripts/1_data_preprocessing.py`)
- **Purpose**: Cleanses scRNA-seq data following existing P4P framework
- **Outputs**: matrix.mtx, labels.csv, genes.txt, processed_data.h5ad, metadata.json

### 2. GMVAE Training (`scripts/2_train_gmvae.py`)
- **Purpose**: Trains GMVAE with ZINB decoder and freezes for downstream use
- **Outputs**: Trained and frozen GMVAE model (.pth file)

### 3. Classification Training (`scripts/3_train_classifier.py`)
- **Purpose**: Trains patient classifier using frozen GMVAE embeddings
- **Outputs**: Trained classifier with performance metrics

## Reference Repositories

- **ProtoCell4P**: Original P4P implementation
- **bulk2sc_GMVAE**: Original Bulk2SC GMVAE implementation
