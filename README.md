espfit
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/espfit/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/espfit/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/espfit/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/espfit/branch/main)


Infrastruture to train espaloma with experimental observables

### Installation
>mamba create -n espfit  
>mamba install -c conda-forge espaloma=0.3.2  
>#uninstall openff-toolkit and install a customized version to support dgl graphs created using openff-toolkit=0.10.6  
>conda uninstall --force openff-toolkit  
>pip install git+https://github.com/kntkb/openff-toolkit.git@7e9d0225782ef723083407a1cbf1f4f70631f934  
>#install openeye-toolkit  
>mamba install openeye-toolkits -c openeye  
>#uninstall openmmforcefields if < 0.12.0
>conda uninstall --force openmmforcefields
>#use pip instead of mamba to avoid dependency issues with ambertools and python
>pip install git+https://github.com/openmm/openmmforcefields@0.12.0

### Quick Usage

### Change logging
```python
# load dgl graph data
from espfit.utils import logging
logging.get_logging_level()
#>'INFO'
logging.set_logging_level('DEBUG')
#>'DEBUG'
```
#### Train espaloma
```python
# load dgl graph data
from espfit.utils.graphs import CustomGraphDataset  
path = 'espfit/data/qcdata/openff-toolkit-0.10.6/dgl2/protein-torsion-sm/'
ds = CustomGraphDataset.load(path)
ds.reshape_conformation_size(n_confs=50)
ds.compute_relative_energy()
# create esplama model
from espfit.app.train import EspalomaModel
filename = 'espfit/data/config/config.toml'
model = EspalomaModel.from_toml(filename)
model.dataset_train = ds
# train
model.train()
```
#### Standard MD (default: espaloma-0.3.2 force field for solute molecules)
```python
from espfit.app.sampler import SetupSampler
c = SetupSampler()
filename = 'espfit/data/target/testsystems/nucleoside/pdbfixer_min.pdb'
c.create_system(biopolymer_file=filename)
c.minimize()
c.run()
```

### Prerequisite
- [Modified version](https://github.com/kntkb/openff-toolkit/tree/v0.14.5) of openff-toolkit `0.11.5` is required to train espaloma using dgl datasets ([Zenodo](https://zenodo.org/records/8150601)), which were used to train `espaloma-0.3`.
- OpenEye toolkit is required to load PBD files into OpenFF Molecule objects. Academic license can be obtained [here](https://www.eyesopen.com/academic-licensing).


### Copyright

Copyright (c) 2023, Ken Takaba


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
