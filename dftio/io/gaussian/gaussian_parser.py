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
    def __init__(self):
        pass

    def get_structure(self, file_path):
        return get_atoms(file_path)

    def get_eigenvalue(self):
        psas

    # Key word Pop=Full is required
    def get_basis(self):
        pass

    def get_blocks(self, logname, hamiltonian=True, overlap=False, density=False, convention=gau_6311_plus_gdp_convention):
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
        if density:
            density_matrix = read_density_from_gau_log(logname, nbf=nbasis)
            density_matrix = density_matrix[..., molecule_transform_indices, :]
            density_matrix = density_matrix[..., :, molecule_transform_indices]
        return hamiltonian_matrix, overlap_matrix, density_matrix
