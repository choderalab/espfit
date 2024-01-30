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
    ds.drop_and_merge_duplicates(save_merged_dataset=True, dataset_name='misc', output_directory_path=str(temporary_directory))
    ds.reshape_conformation_size(n_confs=50)
    #ds.compute_relative_energy()

    return ds


def test_train(test_load_dataset, test_create_espaloma_model, tmpdir):
    """Test function to train an EspalomaModel object.
    """

    #
    # TODO: STILL WORKING ON THIS FUNCTION
    #
    
    # Load dataset and model
    ds = test_load_dataset
    model = test_create_espaloma_model

    # create temporary checkpoint directory
    checkpoint_directory = tmpdir.mkdir('checkpoints')
    #model.config['espaloma']['output_prefix'] = checkpoint_directory
    
    # Train model
    model.dataset_train = ds
    model.train(output_prefix=str(checkpoint_directory))   # overwrite output_prefix

    # Test if the model has been trained
    import glob
    n_checkpoints = len(glob.glob(str(checkpoint_directory.join('*.pt'))))
    assert n_checkpoints < 0
