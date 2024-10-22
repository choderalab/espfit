espfit
==============================
[//]: # (Badges)
[![CI](https://github.com/kntkb/espfit/actions/workflows/CI.yaml/badge.svg)](https://github.com/kntkb/espfit/actions/workflows/CI.yaml)
[![codecov](https://codecov.io/gh/kntkb/espfit/branch/main/graph/badge.svg)](https://codecov.io/gh/kntkb/espfit/branch/main)
<!--[![GitHub Actions Build Status](https://github.com/kntkb/espfit/workflows/CI/badge.svg)](https://github.com/kntkb/espfit/actions?query=workflow%3ACI)-->


Infrastruture to train espaloma with experimental observables


### Installation
>mamba create -n espfit python=3.11  
>mamba install -c conda-forge espaloma=0.3.2  
>conda uninstall --force openff-toolkit  
>pip install git+https://github.com/kntkb/openff-toolkit.git@7e9d0225782ef723083407a1cbf1f4f70631f934  
>mamba install openeye-toolkits -c openeye  
>conda uninstall --force openmmforcefields  
>pip install git+https://github.com/openmm/openmmforcefields@0.12.0  
>mamba install openmmtools  
>mamba install barnaba  

####  Notes
- `openff-toolkit` is re-installed with a customized version to support dgl graphs created using `openff-toolkit=0.10.6`
- `openmmforcefields` is reinstalled if the version is `<0.12.0` using pip to avoid dependency issues with `ambertools` and `python`. espaloma functionalities are better supported after `>=0.12.0`.


### Quick Usage
```python
from espfit.utils.graphs import CustomGraphDataset  
path = 'espfit/data/qcdata/openff-toolkit-0.10.6/dgl2/protein-torsion-sm/'
ds = CustomGraphDataset.load(path)
ds.reshape_conformation_size(n_confs=50, include_min_energy_conf=True)
ds.compute_relative_energy()
# Create esplama model
from espfit.app.train import EspalomaModel
filename = 'espfit/data/config/config.toml'
# Override training settings in config.toml
kwargs = {'output_directory_path': 'checkpoints', 'epochs': 100}
model = EspalomaModel.from_toml(filename, **kwargs)
model.dataset_train = ds
# Set sampler settings
model.train_sampler(sampler_patience=800, neff_threshold=0.2, sampler_weight=1)
```

### Standalone Usage
#### Change logging
```python
from espfit.utils import logging
logging.get_logging_level()
#>'INFO'
logging.set_logging_level('DEBUG')
```

#### Train espaloma
```python
# Load dgl graph data
from espfit.utils.graphs import CustomGraphDataset  
path = 'espfit/data/qcdata/openff-toolkit-0.10.6/dgl2/protein-torsion-sm/'
ds = CustomGraphDataset.load(path)
ds.reshape_conformation_size(n_confs=50)
ds.compute_relative_energy()
# Create esplama model
from espfit.app.train import EspalomaModel
filename = 'espfit/data/config/config.toml'
model = EspalomaModel.from_toml(filename)
model.dataset_train = ds
# Change default training settings
model.epochs = 100
model.output_directory_path = 'path/to/output'
# Train (default output directory is current path)
model.train()
```

#### Standard MD (default: espaloma-0.3.2 force field for solute molecules)
```python
# Create a new system and run simulation
from espfit.app.sampler import SetupSampler
c = SetupSampler()
filename = 'espfit/data/target/testsystems/nucleoside/pdbfixer_min.pdb'
c.create_system(biopolymer_file=filename)
c.minimize()
# Change default settings
c.nsteps = 1000
c.run()
# Export to XML
c.export_xml(exportSystem=True, exportState=True, exportIntegrator=True, output_directory_path='path/to/output')
```

#### Re-start MD from exisiting XML
```python
from espfit.app.sampler import SetupSampler
c = SetupSampler.from_xml(input_directory_path='path/to/input')
c.nsteps = 1000
c.run()
```

#### Compute RNA J-couplings from MD trajectory
```python
from experiment import RNASystem
rna = RNASystem()
rna.load_traj(input_directory_path='path/to/input')
couplings = rna.compute_jcouplings(couplings=['H1H2', 'H2H3', 'H3H4'], residues=['A_1_0'])
```

### Prerequisite
- [Modified version](https://github.com/kntkb/openff-toolkit/tree/v0.14.5) of openff-toolkit `0.11.5` is required to train espaloma using dgl datasets ([Zenodo](https://zenodo.org/records/8150601)), which were used to train `espaloma-0.3`.
- OpenEye toolkit is required to load PDB files into OpenFF Molecule objects. Academic license can be obtained [here](https://www.eyesopen.com/academic-licensing).


### Copyright

Copyright (c) 2023, Ken Takaba


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
