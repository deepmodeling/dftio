import re
import json

import numpy as np
from ase import Atoms, Atom
from ...data import _keys
from ...register import Register
from ..parse import Parser, ParserRegister
from .gaussian_tools import *
from .gaussian_conventionns import orbital_sign_map


@ParserRegister.register("gaussian")
class GaussianParser(Parser):
    def __init__(self, root, prefix, convention_file=None, valid_gau_info_path=None, add_phase_transfer=False, **kwargs):
        super(GaussianParser, self).__init__(root, prefix)
        self.add_phase_transfer = add_phase_transfer
        self.is_fixed_convention = False
        self.on_the_fly_convention_done = False
        if convention_file:
            self.is_fixed_convention = True
            with open(convention_file, 'r') as f:
                self.convention = json.load(f)
        self.atomic_symbols = {}
        self.nbasis = {}
        if valid_gau_info_path:
            self.raw_datas = get_gau_logs(valid_gau_info_path)
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

        # Cache basic info
        if idx not in self.nbasis.keys():
            self.nbasis[idx], atoms = get_basic_info(file_path)
            self.atomic_symbols[idx] = atoms.symbols
        nbasis = self.nbasis[idx]
        atomic_symbols = self.atomic_symbols[idx]

        # Get convention if needed
        if not self.is_fixed_convention and not self.on_the_fly_convention_done:
            self.convention = get_convention(file_path)

        # Generate indices
        molecule_transform_indices, atom_in_mo_indices = generate_molecule_transform_indices(
            atom_types=atomic_symbols,
            atom_to_transform_indices=self.convention['atom_to_transform_indices']
        )

        # Get phase sign list if needed
        phase_sign_list = None
        if self.add_phase_transfer:
            atom_to_sorted_orbitals = convert_to_sorted_orbitals(self.convention['atom_to_dftio_orbitals'])
            phase_sign_list = get_phase_sign_list(
                atomic_symbols=atomic_symbols,
                atom_to_sorted_orbitals=atom_to_sorted_orbitals,
                orbital_sign_map=orbital_sign_map
            )

        results = []
        for matrix_type, should_compute in [
            ('ham', hamiltonian),
            ('overlap', overlap),
            ('density', density_matrix)
        ]:
            if should_compute:
                if matrix_type == 'ham':
                    matrix = read_fock_from_gau_log(file_path, nbf=nbasis)
                elif matrix_type == 'overlap':
                    matrix = read_int1e_from_gau_log(file_path, matrix_type=0, nbf=nbasis)
                else:
                    matrix = read_density_from_gau_log(file_path, nbf=nbasis)

                matrix = transform_matrix(matrix=matrix, transform_indices=molecule_transform_indices)

                if self.add_phase_transfer:
                    matrix = apply_phase_signs_to_matrix(matrix, phase_sign_list)

                result = cut_matrix(full_matrix=matrix, atom_in_mo_indices=atom_in_mo_indices)
            else:
                result = None
            results.append([result])

        return results
