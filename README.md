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
>#uninstall openff-toolkit and install a customized version to support dgl graphs created using openff-toolkit=0.10.6  
>conda uninstall --force openff-toolkit  
>pip install git+https://github.com/kntkb/openff-toolkit.git@7e9d0225782ef723083407a1cbf1f4f70631f934  
>#install openeye-toolkit  
>mamba install openeye-toolkits -c openeye  
>#uninstall openmmforcefields if < 0.12.0  
>conda uninstall --force openmmforcefields  
>#use pip instead of mamba to avoid dependency issues with ambertools and python  
>pip install git+https://github.com/openmm/openmmforcefields@0.12.0  
>#install openmmtools  
>mamba install openmmtools  
>#install barnaba  
>mamba install barnaba  


### Quick Usage

#### Change logging
```python
# load dgl graph data
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
# Train
model.train(output_directory_path='path/to/output')
# To extend training, update the `epoch` in config.toml
# Alternatively, do the following:
model.config['espaloma']['train']['epochs'] = 50
model.train(output_directory_path='path/to/output')
```

#### Standard MD (default: espaloma-0.3.2 force field for solute molecules)
```python
# Create a new system and run simulation
from espfit.app.sampler import SetupSampler
c = SetupSampler()
filename = 'espfit/data/target/testsystems/nucleoside/pdbfixer_min.pdb'
c.create_system(biopolymer_file=filename)
c.minimize(maxIterations=10)
c.run(nsteps=10, output_directory_path='path/to/output')
# Export to XML
c.export_xml(exportSystem=True, exportState=True, exportIntegrator=True, output_directory_path='path/to/output')
```

#### Re-start MD from exisiting XML
```python
from espfit.app.sampler import SetupSampler
c = SetupSampler.from_xml(input_directory_path='path/to/input')
c.run(nsteps=10, output_directory_path='path/to/output')
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
- OpenEye toolkit is required to load PBD files into OpenFF Molecule objects. Academic license can be obtained [here](https://www.eyesopen.com/academic-licensing).


### Copyright

Copyright (c) 2023, Ken Takaba


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
