import os
import json
import numpy as np

from jestr.data.transforms import SpecBinner, SpecBinnerLog
from massspecgym.data.transforms import SpecTransform, MolTransform
from jestr.data.transforms import MolToGraph
import jestr.data.datasets as jestr_datasets
import typing as T
import matchms

def get_spec_featurizer(spectra_view: T.Union[str, list[str]],
                         params) -> T.Union[SpecTransform, T.Dict[str, SpecTransform]]:
    
    featurizers = {"BinnedSpectra": SpecBinner,
        "SpecBinnerLog": SpecBinnerLog}

    spectra_featurizer = {}

    if isinstance(spectra_view, str):
        spectra_view = [spectra_view]

    for view in spectra_view:
        featurizer_params = {'max_mz': params['max_mz']}
        if view in ["BinnedSpectra", "SpecBinnerLog"]:
            featurizer_params.update({'bin_width': params['bin_width']})
        else:
            raise Exception(f'Spectra view {view} is not supported')
        
        spectra_featurizer[view] = featurizers[view](**featurizer_params)

    return spectra_featurizer

def get_mol_featurizer(molecule_view: T.Union[str, T.List[str]], params) -> MolTransform:
    featurizes = {'MolGraph':MolToGraph}
    mol_featurizer = {}

    if isinstance(molecule_view, str):
        molecule_view = [molecule_view]
    for view in molecule_view:
        featurizer_params = {}
        if view in ('MolGraph'):
            featurizer_params.update({'atom_feature': params['atom_feature'], 'bond_feature': params['bond_feature'], 'element_list': params['element_list']})
        
        if len(molecule_view) == 1:
            return featurizes[view](**featurizer_params)

        mol_featurizer[view] = featurizes[view](**featurizer_params)
    
    return mol_featurizer

def get_test_ms_dataset(spectra_view: T.Union[str, T.List[str]],
                 mol_view: T.Union[str, T.List[str]],
                 spectra_featurizer: SpecTransform,
                 mol_featurizer: MolTransform,
                 params):

    dataset_params = {'spectra_view': spectra_view, 'pth': params['dataset_pth'], 'spec_transform': spectra_featurizer, 'mol_transform': mol_featurizer, "candidates_pth": params['candidates_pth']}

    return jestr_datasets.ExpandedRetrievalDataset(**dataset_params)
    
def get_ms_dataset(spectra_view: str,
                 mol_view: str,
                 spectra_featurizer: SpecTransform,
                 mol_featurizer: MolTransform,
                 params):
    

    # set up dataset_parameters
    dataset_params = {'pth': params['dataset_pth'], 'spec_transform': spectra_featurizer, 'mol_transform': mol_featurizer, 'spectra_view': spectra_view}
    
    return jestr_datasets.JESTR1_MassSpecDataset(**dataset_params)

class PrepMatchMS:
    def __init__(self, spectra_view) -> None:
        if spectra_view in ('SpecBinnerLog', 'BinnedSpectra'):
            self.prepare = self.specMzInt
        else:
            raise Exception("Spectra view is not supported.")
        
    def specMzInt(self, row):
        return matchms.Spectrum(
            mz = row['mzs'],
            intensities = row['intensities'],
            metadata = {'precursor_mz': row['precursor_mz']}
        )