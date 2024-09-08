import re
import json

import numpy as np
from ase import Atoms, Atom
from ...data import _keys
from ...register import Register
from ..parse import Parser, ParserRegister
from .gaussian_tools import *


@ParserRegister.register("gaussian")
class GaussianParser(Parser):
    def __init__(self, root, prefix, convention_file=None, valid_gau_info_path=None, **kwargs):
        super(GaussianParser, self).__init__(root, prefix)
        self.is_fixed_convention = False
        self.on_the_fly_convention_done = False
        if convention_file:
            self.is_fixed_convention = True
            with open(convention_file, 'r') as f:
                self.convention = json.load(f)
        self.atomic_symbols = {}
        self.nbasis = {}
        if valid_gau_info_path:
            self.raw_datas = []
            with open(valid_gau_info_path, 'r') as file:
                for line in file.readlines():
                    self.raw_datas.append(line.strip())
        else:
            with open('valid_gaussian_logs.txt', 'w') as f:
                for a_raw_datapath in self.raw_datas:
                    f.write(a_raw_datapath + '\n')

    def get_structure(self, idx):
        file_path = self.raw_datas[idx]
        nbasis, atoms = get_basic_info(file_path)
        self.atomic_symbols[idx] = atoms.symbols
        self.nbasis[idx] = nbasis
        structure = {
            _keys.ATOMIC_NUMBERS_KEY: atoms.numbers,
            _keys.PBC_KEY: np.array([False, False, False]),
            _keys.POSITIONS_KEY: atoms.positions.reshape(1, -1, 3).astype(np.float32),
            _keys.CELL_KEY: atoms.cell.reshape(1,3,3).astype(np.float32)
        }
        return structure

    def get_eigenvalue(self):
        psas

    def get_basis(self, idx):
        if not self.is_fixed_convention:
            file_path = self.raw_datas[idx]
            self.convention = get_convention(file_path)
            self.on_the_fly_convention_done = True
        return self.convention['atom_to_dftio_orbitals']

    def get_blocks(self, idx, hamiltonian=True, overlap=False, density_matrix=False):
        file_path = self.raw_datas[idx]
        if idx not in self.nbasis.keys():
            nbasis, atoms = get_basic_info(file_path)
            atomic_symbols = atoms.symbols
        else:
            nbasis = self.nbasis[idx]
            atomic_symbols = self.atomic_symbols[idx]
        if self.is_fixed_convention == False and self.on_the_fly_convention_done == False:
            self.convention = get_convention(file_path)
        molecule_transform_indices, atom_in_mo_indices = generate_molecule_transform_indices(atom_types=atomic_symbols,
                                            atom_to_transform_indices=self.convention['atom_to_transform_indices'])
        ham_dict, overlap_dict, density_dict = None, None, None
        if hamiltonian:
            hamiltonian_matrix = read_fock_from_gau_log(file_path, nbf=nbasis)
            hamiltonian_matrix = transform_matrix(matrix=hamiltonian_matrix, transform_indices=molecule_transform_indices)
            ham_dict = cut_matrix(full_matrix=hamiltonian_matrix, atom_in_mo_indices=atom_in_mo_indices)
        if overlap:
            overlap_matrix = read_int1e_from_gau_log(file_path, matrix_type=0, nbf=nbasis)
            overlap_matrix = transform_matrix(matrix=overlap_matrix, transform_indices=molecule_transform_indices)
            overlap_dict = cut_matrix(full_matrix=overlap_matrix, atom_in_mo_indices=atom_in_mo_indices)
        if density_matrix:
            density_matrix = read_density_from_gau_log(file_path, nbf=nbasis)
            density_matrix = transform_matrix(matrix=density_matrix, transform_indices=molecule_transform_indices)
            density_dict = cut_matrix(full_matrix=density_matrix, atom_in_mo_indices=atom_in_mo_indices)

        return [ham_dict], [overlap_dict], [density_dict]
