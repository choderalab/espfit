import pytest
from importlib.resources import files
from espfit.utils.graphs import CustomGraphDataset


paths = [
    'data/qcdata/openff-toolkit-0.10.6/dgl2/gen2-torsion-sm',
    'data/qcdata/openff-toolkit-0.10.6/dgl2/protein-torsion-sm',
    'data/qcdata/openff-toolkit-0.10.6/dgl2/rna-diverse-sm',
]


@pytest.fixture
def mydata_gen2_torsion_sm():
    """Fixture function to load gen2-torsion-sm dataset.
    
    Returns
    -------
    ds : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `gen2 torsion`.
    """
    mydata = files('espfit').joinpath(paths[0])   # PosixPath
    ds = CustomGraphDataset.load(str(mydata))
    return ds


@pytest.fixture
def mydata_protein_torsion_sm():
    """Fixture function to load protein-torsion-sm dataset.
        
    Returns
    -------
    ds : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `protein torsion`.
    """
    mydata = files('espfit').joinpath(paths[1])   # PosixPath
    ds = CustomGraphDataset.load(str(mydata))
    return ds


@pytest.fixture
def mydata_rna_diverse_sm():
    """Fixture function to load rna-diverse-sm dataset.
        
    Returns
    -------
    ds : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `rna diverse`.
    """
    mydata = files('espfit').joinpath(paths[2])   # PosixPath
    ds = CustomGraphDataset.load(str(mydata))
    return ds


def test_load_dataset(mydata_gen2_torsion_sm):
    """Test the loading of a single dataset.

    Parameters
    ----------
    mydata_gen2_torsion_sm : espfit.utils.graphs.CustomGraphDataset

    Raises
    ------
    AssertionError : If the number of molecular conformers does not match.

    Returns
    -------
    """
    ds = mydata_gen2_torsion_sm
    nconfs = [g.nodes['g'].data['u_ref'].shape[1] for g in ds]
    assert nconfs == [24, 24, 24, 13, 24, 24, 24, 24], 'Number of molecular conformers does not match'


def test_load_dataset_multiple(mydata_gen2_torsion_sm, mydata_protein_torsion_sm, mydata_rna_diverse_sm):
    """Load multiple datasets.

    This function loads multiple datasets and performs assertions on the loaded data.

    Parameters
    ----------
    mydata_gen2_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `gen2 torsion`.

    mydata_protein_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `protein torsion`.
    
    mydata_rna_diverse_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `rna diverse`

    Raises
    ------
    AssertionError : If the total number of molecules or conformations does not match.

    Returns
    -------
    """
    ds = mydata_gen2_torsion_sm
    ds += mydata_protein_torsion_sm
    ds += mydata_rna_diverse_sm
    nconfs = [g.nodes['g'].data['u_ref'].shape[1] for g in ds]
    assert len(nconfs) == 23, 'Total number of molecules does not match'
    assert sum(nconfs) == 5636, 'Total number of conformations does not match'


def test_drop_and_merge_duplicates(mydata_gen2_torsion_sm, tmpdir):
    """Test function to drop and merge duplicate molecules.

    Parameters
    ----------
    mydata_gen2_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset of `gen2 torsion`.

    tmpdir : tmpdir fixture from pytest

    Returns
    -------
    ds : CustomGraphDataset
        The modified unique dataset without any duplicate molecules.
    """
    ds = mydata_gen2_torsion_sm
    temporary_directory = tmpdir.mkdir('misc')
    ds.drop_and_merge_duplicates(save_merged_dataset=True, dataset_name='misc', output_directory_path=str(temporary_directory))
    nconfs = [ g.nodes['g'].data['u_ref'].shape[1] for g in ds ]
    assert nconfs == [24, 13, 24, 24, 24, 72], 'Number of molecular conformers does not match'

    # return dataset to test espaloma refitting
    return ds


def test_subtract_nonbonded_interactions(mydata_gen2_torsion_sm):
    """Test the subtract_nonbonded_interactions function.
    
    This function checks if the 'u_qm' and 'u_qm_prime' attributes are correctly cloned from 'u_ref' and 'u_ref_prime'
    after subtracting nonbonded interactions.
    
    Parameteres
    -----------
    mydata_gen2_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `gen2 torsion`.
    
    Raises
    ------
    AssertionError : If 'u_qm' or 'u_qm_prime' attributes are not found in the test data.

    Returns
    -------
    """
    ds = mydata_gen2_torsion_sm
    ds.subtract_nonbonded_interactions(subtract_vdw=False, subtract_ele=True)   # default settings
    assert 'u_qm' in ds[0].nodes['g'].data.keys(), "Cannot find u_qm in g.nodes['g'].data.keys()"
    assert 'u_qm_prime' in ds[0].nodes['n1'].data.keys(), "Cannot find u_qm in g.nodes['n1'].data.keys()"


def test_filter_high_energy_conformers(mydata_gen2_torsion_sm):
    """Test function to filter high energy conformers.

    Parameters
    ----------
    mydata_gen2_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `gen2 torsion`.

    Raises
    ------
    AssertionError : If the number of molecular conformers does not match.

    Returns
    -------
    """
    ds = mydata_gen2_torsion_sm
    # set relative_energy_thershold very small to ensure some conformers will be filtered
    ds.filter_high_energy_conformers(relative_energy_threshold=0.01, node_feature='u_ref')
    nconfs = [ g.nodes['g'].data['u_ref'].shape[1] for g in ds ]
    assert nconfs == [14, 19, 19, 5, 14, 19, 24, 24], 'Number of molecular conformers does not match'


def test_filter_minimum_conformers(mydata_gen2_torsion_sm):
    """Test case for filtering molecules with conformers less than a certain threshold.

    Parameters
    ----------
    mydata_gen2_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `gen2 torsion`.

    Raises
    ------
    AssertionError : If the number of molecular conformers does not match.

    Returns
    -------
    """
    ds = mydata_gen2_torsion_sm
    nconfs = [g.nodes['g'].data['u_ref'].shape[1] for g in ds]
    ds.filter_minimum_conformers(n_conformer_threshold=20)
    assert nconfs != [24, 24, 24, 24, 24, 24, 24]   # gen2-torsion-sm


def test_compute_baseline_energy_force(mydata_protein_torsion_sm):
    """Test case for computing energy and force using other force fields.

    Parameters
    ----------
    mydata_protein_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `gen2 torsion`.

    Raises
    ------
    AssertionError : If the key for other force field is not found in any of the graph nodes.

    Returns
    -------
    """
    ds = mydata_protein_torsion_sm
    # remove all baseline forcefields
    keys = list(ds[0].nodes['g'].data.keys())
    for g in ds:
        for key in keys:
            if key not in ['u_ref', 'sum_q', 'u_ref']:
                g.nodes['g'].data.pop(key)
                g.nodes['n1'].data.pop(key+'_prime')
    ds.compute_baseline_energy_force(ds.available_forcefields)
    # check if baseline force field energies and forces are added
    for g in ds:
        assert list(g.nodes['g'].data.keys())[-7:] == [ 
            'u_gaff-1.81', 'u_gaff-2.11', 
            'u_openff-1.2.0', 'u_openff-2.0.0', 'u_openff-2.1.0', 
            'u_amber14sb', 'u_amber14sb_onlysc'
            ]


def test_reshape_conformation_size(mydata_gen2_torsion_sm):
    """Test function to reshape all dgl graphs to have the same number of conformations.

    Parameters
    ----------
    mydata_gen2_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `gen2 torsion`.

    Raises
    ------
    AssertionError : If the number of conformations does not match.

    Returns
    -------
    """
    # Test 1) reshape all dgl graphs to have 30 conformations
    ds = mydata_gen2_torsion_sm
    ds.reshape_conformation_size(n_confs=30)
    nconfs = [g.nodes['g'].data['u_ref'].shape[1] for g in ds]
    assert nconfs == [30, 30, 30, 30, 30, 30, 30, 30], 'All molecules should have 30 conformers'
    del ds, nconfs
    # Test 2) reshape all dgl graphs to have 30 conformations
    mydata = files('espfit').joinpath(paths[0])   # PosixPath
    ds = CustomGraphDataset.load(str(mydata))
    ds.reshape_conformation_size(n_confs=20)
    nconfs = [g.nodes['g'].data['u_ref'].shape[1] for g in ds]
    assert nconfs == [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], 'All molecules should have 20 conformers'
    del ds, nconfs


def test_compute_relative_energy(mydata_gen2_torsion_sm):
    """Test the compute_relative_energy method of the dataset.

    Parameters
    ----------
    mydata_gen2_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `gen2 torsion`.

    Raises
    ------
    AssertionError : If the 'u_ref_relative' key is not found in any of the graph nodes.

    Returns
    -------
    """
    ds = mydata_gen2_torsion_sm
    ds.compute_relative_energy()
    for g in ds:
        assert 'u_ref_relative' in g.nodes['g'].data.keys(), "Could not find g.nodes['g'].data['u_ref_relative']"


@pytest.mark.skip("Data split with precision issue")
def test_split(mydata_gen2_torsion_sm, mydata_protein_torsion_sm, mydata_rna_diverse_sm):
    """
    Test data split

    This function tests the data split functionality. It combines three datasets, `mydata_gen2_torsion_sm`,
    `mydata_protein_torsion_sm`, and `mydata_rna_diverse_sm`, and splits them into training, validation, and
    testing sets using the `split` method of the dataset object. It then asserts that the total number of entries
    in the original dataset matches the sum of the lengths of the training, validation, and testing sets.

    Parameters
    ----------
    mydata_gen2_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `gen2 torsion`.

    mydata_protein_torsion_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `protein torsion`.
    
    mydata_rna_diverse_sm : espfit.utils.graphs.CustomGraphDataset
        Small dataset from `rna diverse`

    Raises
    ------
    AssertionError : If the total number of entries does not match.

    Returns
    -------
    """
    ds = mydata_gen2_torsion_sm
    ds += mydata_protein_torsion_sm
    ds += mydata_rna_diverse_sm
    ds_tr, ds_vl_te = ds.split([0.8, 0.2])
    ds_vl, ds_te = ds_vl_te.split([0.1, 0.1])
    assert len(ds) == len(ds_tr) + len(ds_vl) + len(ds_te), 'Total number of entries does not match'
