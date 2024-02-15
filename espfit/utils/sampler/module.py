import logging

_logger = logging.getLogger(__name__)


def check_effective_sample_size():
    # Compute effective sample size
    neff = 0.5
    
    return neff


def run_sampler(sampler_output_directory_path, biopolymer_file, ligand_file, maxIterations, nsteps, small_molecule_forcefield):
    import os
    from espfit.app.sampler import SetupSampler

    c = SetupSampler(output_directory_path=sampler_output_directory_path, small_molecule_forcefield=small_molecule_forcefield)
    c.create_system(biopolymer_file, ligand_file)
    c.minimize(maxIterations)
    c.run(nsteps=nsteps)    
    c.export_xml()


def compute_observable(input_directory_path):
    from espfit.app.experiment import RNASystem
    target = RNASystem()
    target.load_traj(input_directory_path=input_directory_path)
    val = target.compute_jcouplings()
    _logger.info(f'Computed observable: {val}')
    
    import os
    import yaml
    with open(os.path.join(input_directory_path, 'observable.yaml'), 'w') as f:
        yaml.dump(val, f, allow_unicode=True)

    return val


def compute_loss(input_directory_path):
    # Compute observable
    val = compute_observable(input_directory_path)
    _logger.info(f'Computed observable: {val}')

    # Compute loss
    import torch
    loss = torch.tensor(0.0)

    return loss
