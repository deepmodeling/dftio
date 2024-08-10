import re
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
    def __init__(
            self,
            root,
            prefix,
            **kwargs
            ):
        super(GaussianParser, self).__init__(root, prefix)


    def get_structure(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        in_standard_orientation = False
        atoms = Atoms()

        for idx, line in enumerate(lines):
            # Extract atomic coordinates in standard orientation
            if "Standard orientation:" in line:
                in_standard_orientation = True
                atom_data = []  # Reset atom data for the latest standard orientation
            elif in_standard_orientation and "---" in line:
                if len(atoms) > 0:
                    break
            elif in_standard_orientation:
                parts = line.split()
                if len(parts) == 6:  # We expect 6 parts in a valid atom data line
                    try:
                        atomic_number = int(parts[1])
                        x = float(parts[3])
                        y = float(parts[4])
                        z = float(parts[5])
                        atoms.append(Atom(symbol=atomic_number, position=(x, y, z)))
                    except ValueError:
                        continue
        return atoms


    def get_eigenvalue(self):
        psas

    # Key word Pop=Full is required
    def get_basis(self):
        pass

    def get_blocks(self, logname, hamiltonian=True, overlap=False, density_matrix=False, convention=gau_6311_plus_gdp_convention):
        nbasis = get_nbasis(logname)
        atoms = self.get_structure(logname)
        molecule_transform_indices = generate_molecule_transform_indices(atom_types=atoms.symbols,
                                            atom_to_transform_indices=convention['atom_to_transform_indices'])
        hamiltonian_matrix, overlap_matrix, density_matrix = None, None, None
        if hamiltonian:
            hamiltonian_matrix = read_int1e_from_gau_log(logname, matrix_type=3, nbf=nbasis)
            hamiltonian_matrix = hamiltonian_matrix[..., molecule_transform_indices, :]
            hamiltonian_matrix = hamiltonian_matrix[..., :, molecule_transform_indices]
        if overlap:
            overlap_matrix = read_int1e_from_gau_log(logname, matrix_type=0, nbf=nbasis)
            overlap_matrix = overlap_matrix[..., molecule_transform_indices, :]
            overlap_matrix = overlap_matrix[..., :, molecule_transform_indices]
        if density_matrix:
            density_matrix = read_density_from_gau_log(logname, nbf=nbasis)
            density_matrix = density_matrix[..., molecule_transform_indices, :]
            density_matrix = density_matrix[..., :, molecule_transform_indices]
        return hamiltonian_matrix, overlap_matrix, density_matrix
