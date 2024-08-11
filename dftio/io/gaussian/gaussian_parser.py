import re
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, Atom
from ase.visualize import view
from ...data import _keys
from ...register import Register
from ..parse import Parser, ParserRegister
from .gaussian_tools import *
from .gaussian_conventionns import *


@ParserRegister.register("gaussian")
class GaussianParser(Parser):
    def __init__(self, root, prefix, convention_file, **kwargs):
        super(GaussianParser, self).__init__(root, prefix)
        with open(convention_file, 'r') as f:
            self.convention = json.load(f)
        self.atomic_symbols = {}
        self.nbasis = {}

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
        return self.convention['atom_to_sorted_orbitals']

    def get_blocks(self, idx, hamiltonian=True, overlap=False, density=False):
        file_path = self.raw_datas[idx]
        if idx not in self.nbasis.keys():
            nbasis, atoms = get_basic_info(file_path)
            atomic_symbols = atoms.symbols
        else:
            nbasis = self.nbasis[idx]
            atomic_symbols = self.atomic_symbols[idx]
        molecule_transform_indices = generate_molecule_transform_indices(atom_types=atomic_symbols,
                                            atom_to_transform_indices=self.convention['atom_to_transform_indices'])
        hamiltonian_matrix, overlap_matrix, density_matrix = None, None, None
        if hamiltonian:
            hamiltonian_matrix = read_int1e_from_gau_log(file_path, matrix_type=3, nbf=nbasis)
            hamiltonian_matrix = hamiltonian_matrix[..., molecule_transform_indices, :]
            hamiltonian_matrix = hamiltonian_matrix[..., :, molecule_transform_indices]
        if overlap:
            overlap_matrix = read_int1e_from_gau_log(file_path, matrix_type=0, nbf=nbasis)
            overlap_matrix = overlap_matrix[..., molecule_transform_indices, :]
            overlap_matrix = overlap_matrix[..., :, molecule_transform_indices]
        if density:
            density_matrix = read_density_from_gau_log(file_path, nbf=nbasis)
            density_matrix = density_matrix[..., molecule_transform_indices, :]
            density_matrix = density_matrix[..., :, molecule_transform_indices]
        return hamiltonian_matrix, overlap_matrix, density_matrix
