import pytest
from importlib.resources import files
from espfit.utils.graphs import CustomGraphDataset
from espfit.app.train import EspalomaModel


@pytest.fixture
def test_create_espaloma_model():
    """Test function to load a TOML configuration file and create an EspalomaModel object.

    Returns
    -------
    model : espfit.app.train.EspalomaModel
        The created EspalomaModel object.
    """
    filename = files('espfit').joinpath('data/config/config.toml')   # PosixPath
    model = EspalomaModel.from_toml(str(filename))

    return model


@pytest.fixture
def test_load_dataset(tmpdir):
    """Test function to load a dataset and prepare it for training.

    Parameters
    ----------
    tmpdir : py._path.local.LocalPath   # IS THIS CORRECT?
        Temporary directory.

    Notes
    -----
    This function is not intended for production use. It is a minimal example for testing purposes.

    Returns
    -------
    ds : espfit.utils.graphs.CustomGraphDataset
        The loaded dataset.
    """
    # load dataset
    path = 'data/qcdata/openff-toolkit-0.10.6/dgl2/gen2-torsion-sm'
    mydata = files('espfit').joinpath(path)
    ds = CustomGraphDataset.load(str(mydata))

    # Prepare input dataset ready for training
    temporary_directory = tmpdir.mkdir('misc')
    ds.drop_duplicates(isomeric=False, keep=True, save_merged_dataset=True, dataset_name='misc', output_directory_path=str(temporary_directory))
    ds.compute_relative_energy()
    ds.reshape_conformation_size(n_confs=50)
    
    return ds


def test_train(test_load_dataset, test_create_espaloma_model, tmpdir):
    """Test function to train an EspalomaModel object.

    Parameters
    ----------
    test_load_dataset : espfit.utils.graphs.CustomGraphDataset
        The loaded dataset.

    test_create_espaloma_model : espfit.app.train.EspalomaModel
        The created EspalomaModel object.

    tmpdir : py._path.local.LocalPath   # IS THIS CORRECT?
        Temporary directory.
    """
    import glob

    # Load dataset and model
    ds = test_load_dataset
    model = test_create_espaloma_model
    model.dataset_train = ds

    # Create temporary checkpoint directory
    checkpoint_directory = tmpdir.mkdir('checkpoints')   # PosixPath
    model.output_directory_path=str(checkpoint_directory)

    # Train model with arbitrary number of epochs and checkpoint frequency
    model.epochs = 20
    model.checkpoint_frequency = 5
    model.train()

    # Test if the model has been trained
    n_checkpoints = len(glob.glob(str(checkpoint_directory.join('*.pt'))))
    expected_n_checkpoints = int(model.epochs/model.checkpoint_frequency)
    assert expected_n_checkpoints == n_checkpoints == 4   # 20/5 = 4


def test_train_extend(test_load_dataset, test_create_espaloma_model, tmpdir):
    """Test function to extend training an EspalomaModel object.

    Parameters
    ----------
    test_load_dataset : espfit.utils.graphs.CustomGraphDataset
        The loaded dataset.

    test_create_espaloma_model : espfit.app.train.EspalomaModel
        The created EspalomaModel object.

    tmpdir : py._path.local.LocalPath   # IS THIS CORRECT?
        Temporary directory.
    """
    import glob

    # Load dataset and model
    ds = test_load_dataset
    model = test_create_espaloma_model
    model.dataset_train = ds

    # Create temporary checkpoint directory
    checkpoint_directory = tmpdir.mkdir('checkpoints')   # PosixPath
    model.output_directory_path=str(checkpoint_directory)

    # Train model with arbitrary number of epochs and checkpoint frequency
    model.epochs = 10
    model.checkpoint_frequency = 2
    model.train()

    # Test if the model has been trained
    n_checkpoints = len(glob.glob(str(checkpoint_directory.join('*.pt'))))
    expected_n_checkpoints = int(model.epochs/model.checkpoint_frequency)
    assert n_checkpoints == expected_n_checkpoints == 5   # 10/2 = 5

    # Extend training
    model.epochs = 40
    model.train()
    n_checkpoints = len(glob.glob(str(checkpoint_directory.join('*.pt'))))
    expected_n_checkpoints = int(model.epochs/model.checkpoint_frequency)
    assert n_checkpoints == expected_n_checkpoints == 20   # 40/2 = 20


def test_train_extend_failure(test_load_dataset, test_create_espaloma_model, tmpdir):
    """Test function to extend training an EspalomaModel object.

    Parameters
    ----------
    test_load_dataset : espfit.utils.graphs.CustomGraphDataset
        The loaded dataset.

    test_create_espaloma_model : espfit.app.train.EspalomaModel
        The created EspalomaModel object.

    tmpdir : py._path.local.LocalPath   # IS THIS CORRECT?
        Temporary directory.
    """
    import glob

    # Load dataset and model
    ds = test_load_dataset
    model = test_create_espaloma_model
    model.dataset_train = ds

    # Create temporary checkpoint directory
    checkpoint_directory = tmpdir.mkdir('checkpoints')   # PosixPath
    model.output_directory_path=str(checkpoint_directory)

    # Train model
    model.epochs = 20
    model.checkpoint_frequency = 10
    model.train()

    # Test if the model has been trained
    n_checkpoints = len(glob.glob(str(checkpoint_directory.join('*.pt'))))
    expected_n_checkpoints = int(model.epochs/model.checkpoint_frequency)
    assert n_checkpoints == expected_n_checkpoints == 2   # 20/10 = 2

    # Extend training
    # The training should not extend because the given new number of epoch (i.e. 10) is less than the 
    # last epoch of the checkpoint file (i.e. 20). 
    with pytest.raises(SystemExit) as excinfo:
        model.epochs = 10
        model.train()
    assert excinfo.value.code == 0
    #n_checkpoints = len(glob.glob(str(checkpoint_directory.join('*.pt'))))
    #assert n_checkpoints == expected_n_checkpoints == 2   # 20/10 = 2