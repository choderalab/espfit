import pytest
from importlib.resources import files
from espfit.app.analysis import RNASystem


@pytest.fixture
def _get_input_directory_path():
    input_directory_path = files('espfit').joinpath('data/sampler')   # PosixPath
    return input_directory_path 


def test_load_traj(_get_input_directory_path):
    # TODO: Better test
    input_directory_path = _get_input_directory_path
    data = RNASystem(input_directory_path=input_directory_path)
    data.load_traj(reference_pdb='solvated.pdb', trajectory_netcdf='traj.nc')

    assert data.traj is not None


def test_compute_jcouplings(_get_input_directory_path):
    # TODO: Better test
    input_directory_path = _get_input_directory_path
    data = RNASystem(input_directory_path=input_directory_path)    
    data.load_traj()
    couplings = data.compute_jcouplings(couplings=['H1H2', 'H2H3', 'H3H4'])

    assert couplings is not None


def test_compute_jcouplings_all(_get_input_directory_path):
    # TODO: Better test
    input_directory_path = _get_input_directory_path
    data = RNASystem()
    data.input_directory_path = str(input_directory_path)
    data.load_traj()
    couplings = data.compute_jcouplings()
    
    assert couplings is not None
    