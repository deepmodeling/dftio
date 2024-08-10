import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, Atom
from ase.visualize import view
from pprint import pprint
from .gaussian_conventionns import orbital_idx_map


def get_nbasis(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    nbasis = None
    for line in lines:
        # Extract NBasis
        if line.strip().startswith("NBasis"):
            nbasis = int(line.split()[2])
            break
    if nbasis is None:
        print("NBasis keyword not found in the log file.")
    else:
        print(f"NBasis = {nbasis}")
    return nbasis


def find_basis_set(file_path):
    target_line = 'Standard basis:'
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith(target_line):
                return line.strip()[len(target_line):].strip()
    return None


def simplify_orbitals(orbitals):
    simplified = ''
    seen_pdf = set()  # To keep track of 's' orbitals we've already added
    for orbital in orbitals:
        letter = orbital[-1]  # Get the last character (s, p, d, or f)
        if letter == 's':
            simplified += letter
        elif letter in 'pdf':
            if orbital not in seen_pdf:
                simplified += letter
                seen_pdf.add(orbital)
        else:
            raise NotImplementedError(f'Orbital {letter} has not been supported.')
    return simplified


def process_atomic_orbitals(orbital_list, orbital_idx_map):
    # Create a dictionary to store orbitals by type
    orbitals_by_type = {'s': [], 'p': [], 'd': [], 'f': []}

    # Sort orbitals into types
    for i, orbital in enumerate(orbital_list):
        orbital_type = orbital[-1]
        orbitals_by_type[orbital_type].append((i, orbital))

    transform_indices = []
    orbital_counts = Counter()

    # Process each orbital type in order: s, p, d, f
    for orbital_type in ['s', 'p', 'd', 'f']:
        # Sort orbitals of this type by principal quantum number
        sorted_orbitals = sorted(orbitals_by_type[orbital_type], key=lambda x: int(x[1][0]))

        if orbital_type == 's':
            # For s orbitals, just add the indices
            transform_indices.extend([x[0] for x in sorted_orbitals])
        else:
            # For p, d, f orbitals, apply the shifts
            shifts = orbital_idx_map[orbital_type]
            for i in range(0, len(sorted_orbitals), len(shifts)):
                for shift in shifts:
                    if i + shift < len(sorted_orbitals):
                        transform_indices.append(sorted_orbitals[i + shift][0])

        # Count the orbitals
        orbital_counts[orbital_type] = len(sorted_orbitals)

    # Simplify and check orbital counts
    s_sets = orbital_counts['s']
    assert abs(orbital_counts['p'] % 3) < 1e-6, "p orbital is not multiple of 3"
    p_sets = orbital_counts['p'] // 3
    assert abs(orbital_counts['d'] % 5) < 1e-6, "d orbital is not multiple of 5"
    d_sets = orbital_counts['d'] // 5
    assert abs(orbital_counts['f'] % 7) < 1e-6, "f orbital is not multiple of 7"
    f_sets = orbital_counts['f'] // 7

    sorted_orbital_str = s_sets * 's' + p_sets * 'p' + d_sets * 'd' + f_sets * 'f'

    return transform_indices, sorted_orbital_str


# Key word Pop=Full is required
def parse_orbital_populations(filename, nbf, orbital_idx_map):
    target_pattern = re.compile(r'\s*Gross orbital populations:')
    with open(filename, 'r') as f:
        # Search for the target pattern
        for line in f:
            if target_pattern.search(line):
                break
        else:
            raise ValueError(f"No match for orbital population block found in file {filename}")
        orbitals = []
        atoms_list = []
        next(f)  # Skip the line with column numbers
        for i in range(nbf):
            parts = next(f).split()
            if len(parts) >= 5:
                an_orbital = parts[3][:2].lower()
                atoms_list.append(parts[2])
            else:
                an_orbital = parts[1][:2].lower()
            orbitals.append(an_orbital)
        atom_to_orbitals = {}
        atom_to_simplified_orbitals = {}
        atom_to_sorted_orbitals = {}
        atom_to_transform_indices = {}
        orbital_index = 1
        for atom in atoms_list:
            # Get the corresponding orbitals for the current atom
            unit_orbitals = ['1s']
            while orbital_index < len(orbitals):
                orbital = orbitals[orbital_index]
                orbital_index += 1
                if orbital == '1s':
                    break
                unit_orbitals.append(orbital)
            if not atom in atom_to_orbitals.keys():
                atom_to_orbitals[atom] = unit_orbitals
                transform_indices, sorted_orbital_str = process_atomic_orbitals(unit_orbitals, orbital_idx_map)
                atom_to_sorted_orbitals[atom] = sorted_orbital_str
                atom_to_transform_indices[atom] = transform_indices
                atom_to_simplified_orbitals[atom] = simplify_orbitals(unit_orbitals)
    return orbitals, atom_to_orbitals, atom_to_simplified_orbitals, atom_to_sorted_orbitals, atom_to_transform_indices



def generate_molecule_transform_indices(atom_types, atom_to_transform_indices):
    molecule_transform_indices = []
    current_offset = 0

    for atom_type in atom_types:
        # Get the transform indices for this atom type
        atom_indices = atom_to_transform_indices[atom_type]

        # Add the current offset to each index
        adjusted_indices = [index + current_offset for index in atom_indices]

        # Add these adjusted indices to the molecule transform indices
        molecule_transform_indices.extend(adjusted_indices)

        # Update the offset for the next atom
        current_offset += max(atom_indices) + 1

    return molecule_transform_indices


def matrix_to_image(matrix, filename='matrix_image.png'):
    # Convert the matrix to a numpy array
    matrix = np.array(matrix)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(20, 20))

    # Create a color-mapped image of the matrix
    im = ax.imshow(matrix, cmap='viridis')

    # Add a colorbar
    plt.colorbar(im)

    # Add annotations for each cell
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.annotate(f'{matrix[i, j]:.2f}',
                        xy=(j, i),
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center',
                        va='center',
                        color='w',
                        fontweight='bold')

    # Set title and labels
    ax.set_title('Matrix Visualization')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the image
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Image saved as {filename}")


# modified from Mokit,
# see https://github.com/1234zou/MOKIT/blob/7499356b1ff0f9d8b9efbb846395059867dbba4c/src/rwwfn.f90#L3405
# Key word IOp(5/33=2) is required
def read_density_from_gau_log(logname, nbf):
    target_pattern = re.compile(r'\s*Density Matrix:')
    mat = np.zeros((nbf, nbf))
    with open(logname, 'r') as f:
        # Search for the target pattern
        for line in f:
            if target_pattern.search(line):
                break
        else:
            raise ValueError(f"No match for Density matrix found in file {logname}")
        # Read the matrix data
        n = (nbf + 4) // 5  # Equivalent to ceiling division
        for i in range(n):
            next(f)  # Skip the line with column numbers
            k = 5 * i
            for j in range(k, nbf):
                line = next(f)[20:].split()
                m = min(5, nbf - k)
                actual_line_len = len(line[0:m])
                mat[k:k + actual_line_len, j] = [float(x.replace('D', 'E')) for x in line[0:m]]

    # Mirror the upper triangle to the lower triangle
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat


# modified from Mokit,
# see https://github.com/1234zou/MOKIT/blob/7499356b1ff0f9d8b9efbb846395059867dbba4c/src/rwwfn.f90#L895
# Key word IOp(3/33=1) is required
def read_int1e_from_gau_log(logname, matrix_type, nbf):
    matrix_types = {
        0: r"Overlap",
        1: r"Kinetic Energy",
        2: r"Potential Energy",
        3: r"Core Hamiltonian",
    }

    if matrix_type not in matrix_types:
        raise ValueError(
            f"Invalid matrix_type = {matrix_type}. Allowed values are 1/2/3/4 for Overlap/Kinetic/Potential/Core Hamiltonian.")
    target_pattern = re.compile(rf"\*+\s*{matrix_types[matrix_type]}\s*\*+")
    mat = np.zeros((nbf, nbf))
    with open(logname, 'r') as f:
        # Search for the target pattern
        for line in f:
            if target_pattern.search(line):
                break
        else:
            raise ValueError(f"No match for '{matrix_types[matrix_type]}' found in file {logname}")

        # Read the matrix data
        n = (nbf + 4) // 5  # Equivalent to ceiling division
        for i in range(n):
            next(f)  # Skip the line with column numbers
            k = 5 * i
            for j in range(k, nbf):
                line = next(f).split()
                m = min(5, nbf - k)
                actual_line_len = len(line[1:m + 1])
                mat[k:k + actual_line_len, j] = [float(x.replace('D', 'E')) for x in line[1:m + 1]]
    # Mirror the upper triangle to the lower triangle
    mat = mat + mat.T - np.diag(mat.diagonal())
    return mat


def get_atoms(file_path):
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


def get_convention(filename):
    nbasis = get_nbasis(filename)
    basis_name = find_basis_set(filename)
    orbitals, atom_to_orbitals, atom_to_simplified_orbitals, atom_to_sorted_orbitals, atom_to_transform_indices = parse_orbital_populations(
        filename, nbasis, orbital_idx_map)
    convention = {
        'atom_to_simplified_orbitals': atom_to_simplified_orbitals,
        'atom_to_sorted_orbitals': atom_to_sorted_orbitals,
        'atom_to_transform_indices': atom_to_transform_indices,
        'basis_name': basis_name,
    }
    pprint(convention, compact=True)
    return convention


def check_eigenvalue_consistency(matrix, manipulated_matrix):
    original_eigenvalues = np.linalg.eigvals(matrix)
    manipulated_eigenvalues = np.linalg.eigvals(manipulated_matrix)
    original_eigenvalues.sort()
    manipulated_eigenvalues.sort()
    tolerance = 1e-10
    is_consistent = np.allclose(original_eigenvalues, manipulated_eigenvalues, atol=tolerance)
    return is_consistent


def check_transform(filepath):
    convention = get_convention(filename='gau.log')
    nbasis = get_nbasis(filepath)
    atoms = get_atoms(filepath)
    molecule_transform_indices = generate_molecule_transform_indices(atom_types=atoms.symbols,
                                                                     atom_to_transform_indices=convention[
                                                                         'atom_to_transform_indices'])
    hamiltonian_matrix = read_int1e_from_gau_log(filepath, matrix_type=3, nbf=nbasis)
    new_hamiltonian_matrix = hamiltonian_matrix[..., molecule_transform_indices, :]
    new_hamiltonian_matrix = new_hamiltonian_matrix[..., :, molecule_transform_indices]
    consistent_flag = check_eigenvalue_consistency(hamiltonian_matrix, new_hamiltonian_matrix)
    if consistent_flag:
        print('Hamiltonian matrix transform is consistent for eigenvalues.')
    else:
        print('Hamiltonian matrix transform is non-consistent for eigenvalues.')

    overlap_matrix = read_int1e_from_gau_log(filepath, matrix_type=0, nbf=nbasis)
    new_overlap_matrix = overlap_matrix[..., molecule_transform_indices, :]
    new_overlap_matrix = new_overlap_matrix[..., :, molecule_transform_indices]
    consistent_flag = check_eigenvalue_consistency(overlap_matrix, new_overlap_matrix)
    if consistent_flag:
        print('Overlap matrix transform is consistent for eigenvalues.')
    else:
        print('Overlap matrix transform is non-consistent for eigenvalues.')

    density_matrix = read_density_from_gau_log(filepath, nbf=nbasis)
    new_density_matrix = density_matrix[..., molecule_transform_indices, :]
    new_density_matrix = new_density_matrix[..., :, molecule_transform_indices]
    consistent_flag = check_eigenvalue_consistency(density_matrix, new_density_matrix)
    if consistent_flag:
        print('Density matrix transform is consistent for eigenvalues.')
    else:
        print('Density matrix transform is non-consistent for eigenvalues.')

