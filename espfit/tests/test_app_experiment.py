import pytest
from importlib.resources import files
from espfit.app.experiment import RNASystem


def test_load_traj():
    input_directory_path = files('espfit').joinpath('data/sampler')   # PosixPath
    data = RNASystem(input_directory_path=input_directory_path)
    data.load_traj(reference_pdb='solvated.pdb', trajectory_netcdf='traj.nc')

    # TODO: Better test
    return data


def test_compute_jcouplings_1():
    input_directory_path = files('espfit').joinpath('data/sampler')   # PosixPath
    data = RNASystem(input_directory_path=input_directory_path)    
    data.load_traj()
    couplings = data.compute_jcouplings(couplings=['H1H2', 'H2H3', 'H3H4'])
    
    # TODO: Better test
    return couplings


def test_compute_jcouplings_2():
    input_directory_path = files('espfit').joinpath('data/sampler')   # PosixPath
    data = RNASystem()
    data.input_directory_path = str(input_directory_path)
    data.load_traj()
    couplings = data.compute_jcouplings(couplings=None)
    
    # TODO: Better test
    return couplings