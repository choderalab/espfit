import pytest
from espfit.utils.data.graphs import CustomGraphDataset


paths = [
    'espfit/data/qcdata/openff-toolkit-0.10.6/dgl2/gen2-torsion-sm',
    'espfit/data/qcdata/openff-toolkit-0.10.6/dgl2/protein-torsion-sm',
    'espfit/data/qcdata/openff-toolkit-0.10.6/dgl2/rna-diverse-sm',
]


def test_load_dataset():
    """Load a single dataset"""
    ds = CustomGraphDataset.load(paths[0])
    nconfs = [ g.nodes['g'].data['u_ref'].shape[1] for g in ds ]
    assert nconfs == [24, 24, 24, 13, 24, 24, 24, 24], 'Number of molecular conformers does not match'


def test_load_dataset_multiple():
    """Load multiple datasets"""
    ds = CustomGraphDataset.load(paths[0])
    for path in paths[1:]:
        ds += CustomGraphDataset.load(path)
    nconfs = [ g.nodes['g'].data['u_ref'].shape[1] for g in ds ]
    assert len(nconfs) == 23, 'Total number of molecules does not match'
    assert sum(nconfs) == 5636, 'Total number of conformations does not match'


def test_drop_and_merge_duplicates():
    """Drop and merge duplicate molecules"""
    ds = CustomGraphDataset.load(paths[0])
    nconfs = [ g.nodes['g'].data['u_ref'].shape[1] for g in ds ]
    # Temporary directory will be automatically cleaned up
    from espaloma.data.utils import make_temp_directory
        with make_temp_directory() as tmpdir:
            import os
            ds.drop_and_merge_duplicates(save_merged_dataset=True, dataset_name='misc', output_directory_path=tempdir)
    assert nconfs == [24, 13, 24, 24, 24, 72], 'Number of molecular conformers does not match'


def test_subtract_nonbonded_interactions():
    """Subtract nonbonded interactions
    
    Check if u_qm and u_qm_prime cloned from u_ref and u_ref_prime
    """
    ds = CustomGraphDataset.load(paths[0])
    # default settings
    ds.subtract_nobonded_interactions(subtract_vdw=False, subtract_ele=True) 
    assert 'u_qm' in ds[0].nodes['g'].data.keys(), "Cannot find u_qm in g.nodes['g'].data.keys()"
    assert 'u_qm_prime' in ds[0].nodes['n1'].data.keys(), "Cannot find u_qm in g.nodes['n1'].data.keys()"


def test_filter_high_energy_conformers():
    """Filter high energy conformers"""
    ds = CustomGraphDataset.load(paths[0])
    # set relative_energy_thershold very small to ensure some conformers will be filtered
    ds.filter_high_energy_conformers(relative_energy_threshold=0.01, node_feature='u_ref')
    nconfs = [ g.nodes['g'].data['u_ref'].shape[1] for g in ds ]
    assert nconfs == [14, 19, 19, 5, 14, 19, 24, 24], 'Number of molecular conformers does not match'


def test_filter_minimum_conformers():
    """Filter molecules with conformers less than a certain threshold"""
    ds = CustomGraphDataset.load(paths[0])
    nconfs = [ g.nodes['g'].data['u_ref'].shape[1] for g in ds ]
    ds.filter_minimum_conformers(n_conformer_threshold=20)
    assert nconfs != [24, 24, 24, 24, 24, 24, 24]   # gen2-torsion-sm


def test_compute_baseline_energy_force():
    """Compute energy and force using other force fields"""
    # peptides
    ds = CustomGraphDataset.load(paths[1])
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


def test_reshape_conformation_size():
    """Reshape all dgl graphs to have same number of conformations"""
    # 1
    ds = CustomGraphDataset.load(paths[0])
    ds.reshape_conformation_size(n_confs=30)
    nconfs = [ g.nodes['g'].data['u_ref'].shape[1] for g in ds ]
    assert nconfs == [30, 30, 30, 30, 30, 30, 30, 30], 'All molecules should have 30 conformers'
    del ds, nconfs
    # 2
    ds = CustomGraphDataset.load(paths[0])
    ds.reshape_conformation_size(n_confs=20)
    nconfs = [ g.nodes['g'].data['u_ref'].shape[1] for g in ds ]
    assert nconfs == [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], 'All molecules should have 20 conformers'
    del ds, nconfs


def test_compute_relative_energy():
    """Compute relative energy"""
    ds = CustomGraphDataset.load(paths[0])
    ds.compute_relative_energy()
    for g in ds:
        assert 'u_ref_relative' in g.nodes['g'].data.keys(), "Could not find g.nodes['g'].data['u_ref_relative']"


def test_split():
    """Test data split"""
    ds = test_load_dataset_multiple()
    ds_tr, ds_vl_te = ds.split([0.8, 0.2])
    ds_vl, ds_te = ds_vl_te.split([0.1, 0.1])
    assert len(ds) == len(ds_tr) + len(ds_vl) + len(ds_te), 'Total number of entries does not match'

