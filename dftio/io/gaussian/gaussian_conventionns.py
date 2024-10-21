__all__ = [
    'gau_6311_plus_gdp_convention',
]

orbital_idx_map_2_pyscf = {
    's': [0],
    'p': [0, 1, 2],
    'd': [4, 2, 0, 1, 3],
    'f': [6, 4, 2, 0, 1, 3, 5],
}

orbital_idx_map = {
    's': [0],
    'p': [1, 2, 0],
    'd': [4, 2, 0, 1, 3],
    'f': [6, 4, 2, 0, 1, 3, 5],
}

orbital_sign_map = {
    's': [1],
    'p': [-1, 1, -1],
    'd': [1, -1, 1, -1, 1],
    'f': [-1, 1, -1, 1, -1, 1, -1],
}

gau_6311_plus_gdp_convention = {'atom_to_simplified_orbitals': {'C': 'sspspspspd', 'H': 'sssp', 'O': 'sspspspspd'},
                                'atom_to_sorted_orbitals': {'C': 'sssssppppd', 'H': 'sssp', 'O': 'sssssppppd'},
                                'atom_to_transform_indices': {'C': [0, 1, 5, 9, 13, 3, 4, 2, 7, 8, 6, 11, 12,
                                                                    10, 15, 16, 14, 21, 19, 17, 18, 20],
                                                              'H': [0, 1, 2, 4, 5, 3],
                                                              'O': [0, 1, 5, 9, 13, 3, 4, 2, 7, 8, 6, 11, 12,
                                                                    10, 15, 16, 14, 21, 19, 17, 18, 20]},
                                'basis_name': '6-311+g(d,p)', }

pyscf_6311_plus_gdp_convention = {'atom_to_dftio_orbitals': {'C': '5s4p1d', 'H': '3s1p', 'O': '5s4p1d'},
                                  'atom_to_simplified_orbitals': {'C': 'sssssppppd', 'H': 'sssp', 'O': 'sssssppppd'},
                                  'atom_to_transform_indices': {'C': [0, 1, 2, 3, 4, 6, 7, 5, 9, 10, 8, 12, 13, 11, 15, 16, 14, 17, 18, 19, 20, 21],
                                                                'H': [0, 1, 2, 4, 5, 3],
                                                                'O': [0, 1, 2, 3, 4, 6, 7, 5, 9, 10, 8, 12, 13, 11, 15, 16, 14, 17, 18, 19, 20, 21]},
                                  'basis_name': '6-311+G(d,p) (5D, 7F)'}


gau_6311_plus_gdp_to_pyscf_convention = {
    'atom_to_simplified_orbitals': {'C': 'sspspspspd', 'H': 'sssp', 'O': 'sspspspspd'},
    'atom_to_sorted_orbitals': {'C': 'sssssppppd', 'H': 'sssp', 'O': 'sssssppppd'},
    'atom_to_transform_indices': {'C': [0, 1, 5, 9, 13, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 21, 19, 17, 18, 20],
                                  'H': [0, 1, 2, 3, 4, 5],
                                  'O': [0, 1, 5, 9, 13, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 21, 19, 17, 18, 20]},
    'basis_name': '6-311+g(d,p)', }


gau_def2svp_to_pyscf_convention = {'atom_to_simplified_orbitals': {'C': 'sssppd', 'O': 'sssppd', 'H': 'ssp'},
                                   'atom_to_dftio_orbitals': {'C': '3s2p1d', 'O': '3s2p1d', 'H': '2s1p'},
                                   'atom_to_transform_indices': {'C': [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 11, 9, 10, 12],
                                                                 'O': [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 11, 9, 10, 12],
                                                                 'H': [0, 1, 2, 3, 4]},
                                   'basis_name': 'def2SVP (5D, 7F)'}
