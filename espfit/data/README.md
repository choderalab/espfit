## Manifest

- `config/`: Sample configuration file to train espaloma

- `forcefield/`: Force field files

- `qcdata/`: Sample QC data for espaloma training
    - `openff-toolkit-0.10.6/`

- `sampler/`: Sample MD output files of `target/testsystem/nucleoside/pdbfixer_min.pdb`

- `target/`: Stores structures for nucleosides and testsystems
    - `nucleoside/`
        - `adenosine/`
        - `cytidine/`
        - `guanosine/`
        - `uridine/`
    - `testsystems/`
        - `nucleoside/`
        - `protein-ligand/`
        - `multi-protein-ligand/`