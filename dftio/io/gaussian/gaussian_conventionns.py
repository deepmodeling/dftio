
__all__ = [
    'gau_6311_plus_gdp_convention',
]

orbital_idx_map = {
    's': [0],
    'p': [1, 2, 0],
    'd': [4, 2, 0, 1, 3],
    'f': [6, 4, 2, 0, 1, 3, 5],
}

gau_6311_plus_gdp_convention = {'atom_to_simplified_orbitals': {'C': 'sspspspspd', 'H': 'sssp', 'O': 'sspspspspd'},
                                'atom_to_sorted_orbitals': {'C': 'sssssppppd', 'H': 'sssp', 'O': 'sssssppppd'},
                                'atom_to_transform_indices': {'C': [0, 1, 5, 9, 13, 3, 4, 2, 7, 8, 6, 11, 12,
                                                                    10, 15, 16, 14, 21, 19, 17, 18, 20],
                                                              'H': [0, 1, 2, 4, 5, 3],
                                                              'O': [0, 1, 5, 9, 13, 3, 4, 2, 7, 8, 6, 11, 12,
                                                                    10, 15, 16, 14, 21, 19, 17, 18, 20]},
                                'basis_name': '6-311+g(d,p)', }


