# Small qc dataset for test and debug purposes


## Manifest
- `openff-toolkit-0.10.6`: dgl graphs saved with openff-toolkit `0.10.6`  
    - `dgl1/`: Small dgl graph datasets converted from HDF5 downloaded from [Zenodo](https://zenodo.org/records/8148817) with precomputed AM1-BCC ELF10 partial charges using OpenEye Toolkits.
        - `protein-torsion-sm/`: 
            - Copied from `/home/takabak/data/exploring-rna/refit-espaloma/openff-default/01-create-dataset/TorsionDriveDataset/protein-torsion/data`
    - `dgl2/`: Small dgl graph datasets before filtering with pre-computed baseline force field energies and forces. These datasets are one before the preprocessed dataset uploaded in [Zenodo](https://zenodo.org/records/8150601).
        - `protein-torsion-sm/`:  
            - Copied from `/home/takabak/data/exploring-rna/refit-espaloma/openff-default/02-train/merge-data/openff-2.0.0/protein-torsion`
        - `gen2-torsion-sm/`:  
            - Copied from `/home/takabak/data/exploring-rna/refit-espaloma/openff-default/02-train/merge-data/openff-2.0.0/gen2-torsion`
        - `rna-diverse-sm/`:  
            - Copied from `/home/takabak/data/exploring-rna/refit-espaloma/openff-default/02-train/merge-data/openff-2.0.0/rna-diverse`

## Dependencies
openff-toolkit `0.10.6` (<0.11.0) required to load and manipulate dgl graphs

## Note
All dgl graphs were processed with openff-toolkit `0.10.6` which is incompatible with versions > `0.11.0`.