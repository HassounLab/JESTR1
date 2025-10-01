import numpy as np
import torch
import matchms
from typing import Optional
from rdkit.Chem import AllChem as Chem
from jestr.definitions import CHEM_ELEMS_SMALL
from massspecgym.data.transforms import MolTransform, SpecTransform, default_matchms_transforms
from massspecgym.data.transforms import SpecBinner

import dgllife.utils as chemutils
import re

class SpecBinnerLog(SpecTransform):
    def __init__(
        self,
        max_mz: float = 1005,
        bin_width: float = 1,
    ) -> None:
        self.max_mz = max_mz
        self.bin_width = bin_width
        if not (max_mz / bin_width).is_integer():
            raise ValueError("`max_mz` must be divisible by `bin_width`.")
        
    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        return default_matchms_transforms(spec, mz_to=self.max_mz, n_max_peaks=None)
    
    def matchms_to_torch(self, spec: matchms.Spectrum) -> torch.Tensor:
        """
        Bin the spectrum into a fixed number of bins.
        """
        binned_spec = self._bin_mass_spectrum(
            mzs=spec.peaks.mz,
            intensities=spec.peaks.intensities,
            max_mz=self.max_mz,
            bin_width=self.bin_width,
        )
        return torch.from_numpy(binned_spec).to(dtype=torch.float32)

    def _bin_mass_spectrum(
        self, mzs, intensities, max_mz, bin_width
    ):

        # Calculate the number of bins
        num_bins = int(np.ceil(max_mz / bin_width))

        # Calculate the bin indices for each mass
        bin_indices = np.floor(mzs -1 / bin_width).astype(int)

        # Filter out mzs that exceed max_mz
        valid_indices = bin_indices[mzs <= max_mz]
        valid_intensities = intensities[mzs <= max_mz]

        # Clip bin indices to ensure they are within the valid range
        valid_indices = np.clip(valid_indices, 0, num_bins - 1)

        # Initialize an array to store the binned intensities
        binned_intensities = np.zeros(num_bins)

        # Use np.add.at to sum intensities in the appropriate bins
        np.add.at(binned_intensities, valid_indices, valid_intensities)

        binned_intensities = binned_intensities/np.max(binned_intensities) * 999

        binned_intensities = np.log10(binned_intensities + 1) / 3

        return binned_intensities 
    
class MolToGraph(MolTransform):
    def __init__ (self, atom_feature: str = "full", bond_feature: str = "full", element_list: list = CHEM_ELEMS_SMALL):
        self.atom_feature = atom_feature
        self.bond_feature = bond_feature
        self.node_featurizer = self._get_atom_featurizer(element_list=element_list) 
        self.edge_featurizer = self._get_bond_featurizer()
    
    def from_smiles(self, mol:str):
        mol = Chem.MolFromSmiles(mol)
        g = chemutils.mol_to_bigraph(mol, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer, add_self_loop = True,
                             num_virtual_nodes = 0, canonical_atom_order=False)

        return g

    def _get_atom_featurizer(self, element_list) -> dict:
        feature_mode = self.atom_feature
        atom_mass_fun = chemutils.ConcatFeaturizer(
            [chemutils.atom_mass]
        )
        def atom_bond_type_one_hot(atom):
            bs = atom.GetBonds()
            bt = np.array([chemutils.bond_type_one_hot(b) for b in bs])
            return [any(bt[:, i]) for i in range(bt.shape[1])]

        def atom_type_one_hot(atom):
            return chemutils.atom_type_one_hot(
                atom, allowable_set = element_list, encode_unknown = True
            )
        
        if feature_mode == 'light':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                atom_type_one_hot
            ])
        elif feature_mode == 'full':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                atom_type_one_hot, 
                atom_bond_type_one_hot,
                chemutils.atom_degree_one_hot, 
                chemutils.atom_total_degree_one_hot,
                chemutils.atom_explicit_valence_one_hot,
                chemutils.atom_implicit_valence_one_hot,
                chemutils.atom_hybridization_one_hot,
                chemutils.atom_total_num_H_one_hot,
                chemutils.atom_formal_charge_one_hot,
                chemutils.atom_num_radical_electrons_one_hot,
                chemutils.atom_is_aromatic_one_hot,
                chemutils.atom_is_in_ring_one_hot,
                chemutils.atom_chiral_tag_one_hot
            ])
        elif feature_mode == 'medium':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                atom_type_one_hot, 
                atom_bond_type_one_hot,
                chemutils.atom_total_degree_one_hot,
                chemutils.atom_total_num_H_one_hot,
                chemutils.atom_is_aromatic_one_hot,
                chemutils.atom_is_in_ring_one_hot,
            ])
        return chemutils.BaseAtomFeaturizer(
        {"h": atom_featurizer_funs, 
        "m": atom_mass_fun}
    )

    def _get_bond_featurizer(self, self_loop=True) -> dict:
        feature_mode = self.bond_feature
        if feature_mode == 'light':
            return chemutils.BaseBondFeaturizer(
                featurizer_funcs = {'e': chemutils.ConcatFeaturizer([
                    chemutils.bond_type_one_hot
                ])}, self_loop = self_loop
            )
        elif feature_mode == 'full':
            return chemutils.CanonicalBondFeaturizer(
                bond_data_field='e', self_loop = self_loop
            )
