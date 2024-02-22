import pytest
from importlib.resources import files
from espfit.utils.graphs import CustomGraphDataset
from espfit.app.train import EspalomaModel


@pytest.fixture
def test_create_espaloma_from_toml(tmpdir):
    """Test function to load a TOML configuration file and create an EspalomaModel object.

    Returns
    -------
    model : espfit.app.train.EspalomaModel
        The created EspalomaModel object.
    """
    filename = files('espfit').joinpath('data/config/config.toml')   # PosixPath
    model = EspalomaModel.from_toml(str(filename))
    model.output_directory_path = str(tmpdir)   # Update output directory path

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
    ds.compute_relative_energy()

    return ds


def test_train_sampler(test_load_dataset, test_create_espaloma_from_toml):

    """
    TODO
    ----

    * sampler.py needs to support loading temporary espaloma model during training
    """

    # Load dataset and model
    ds = test_load_dataset
    model = test_create_espaloma_from_toml

    # Set espaloma parameters
    model.dataset_train = ds
    model.epochs = 10

    # Train
    model.train_sampler(sampler_patience=3, neff_threshold=0.2)   # fails if sampler_patience is < epochs

    # Check outputs
    import glob
    #assert len(glob.glob(model.output_directory_path + '/*')) > 0
    #assert model.sampler is not None