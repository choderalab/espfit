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


### Quick Usage
```python
# load dgl graph data
from espfit.utils.graphs import CustomGraphDataset  
path = 'espfit/data/qcdata/openff-toolkit-0.10.6/dgl2/protein-torsion-sm/'
ds = CustomGraphDataset.load(paths)
ds.reshape_conformation_size(n_confs=50)
ds.compute_relative_energy()
# create esplama model
from espfit.app.train import EspalomaModel
filename = 'examples/config.toml'
model = EspalomaModel.from_toml(filename)
model.dataset_train = ds
# train
model.train()
```

### Copyright

Copyright (c) 2023, Ken Takaba


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
