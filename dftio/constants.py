import numpy as np
import ase.data as data
import ase

atomic_num_dict = ase.atom.atomic_numbers
atomic_numbers_r = dict(zip(data.atomic_numbers.values(), data.atomic_numbers.keys()))

orbitalId = {0:'s',1:'p',2:'d',3:'f',4:'g',5:'h'}
anglrMId = {'s':0,'p':1,'d':2,'f':3,'g':4,'h':5}
Bohr2Ang = 0.529177249

norb_dict = {'s':1,'e':2,'p':3,'q':4,'d':5,'j':6,'f':7,'o':8,'g':9,'h':11, "i":15, "k":21, "l": 25, "m": 35}
norb_dict_r = {1:'s', 2:'e', 3:'p', 4:'q', 5:'d', 6:'j', 7:'f', 8:'o', 9:'g', 11:'h', 15:"i", 21:"k", 25:"l", 35:"m"}

ABACUS_orbital_number_m = {
    "s": [0],
    "p": [0, 1, -1],
    "d": [0, 1, -1, 2, -2],
    "f": [0, 1, -1, 2, -2, 3, -3],
    "g": [0, 1, -1, 2, -2, 3, -3, 4, -4],
    "h": [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
}

DeePTB_orbital_number_m = {
    "s": [0],
    "p": [-1, 0, 1],
    "d": [-2, -1, 0, 1, 2],
    "f": [-3, -2, -1, 0, 1, 2, 3],
    "g": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    "h": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
}

ABACUS2DFTIO = {
            0: np.eye(1, dtype=np.float32),
            1: np.eye(3, dtype=np.float32)[[2, 0, 1]],
            2: np.eye(5, dtype=np.float32)[[4, 2, 0, 1, 3]],
            3: np.eye(7, dtype=np.float32)[[6, 4, 2, 0, 1, 3, 5]],
            4: np.eye(9, dtype=np.float32)[[8, 6, 4, 2, 0, 1, 3, 5, 7]],
            5: np.eye(11, dtype=np.float32)[[10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9]]
        }

ABACUS2DFTIO[1][[0, 2]] *= -1
ABACUS2DFTIO[2][[1, 3]] *= -1
ABACUS2DFTIO[3][[0, 6, 2, 4]] *= -1
ABACUS2DFTIO[4][[1, 7, 3, 5]] *= -1
ABACUS2DFTIO[5][[0, 10, 8, 2, 6, 4]] *= -1

RESCU2DFTIO = {
            0: np.eye(1, dtype=np.float32),
            1: np.eye(3, dtype=np.float32),
            2: np.eye(5, dtype=np.float32),
            3: np.eye(7, dtype=np.float32),
            4: np.eye(9, dtype=np.float32),
            5: np.eye(11, dtype=np.float32)
        }

RESCU2DFTIO[1][[0, 2]] *= -1
RESCU2DFTIO[2][[1, 3]] *= -1
RESCU2DFTIO[3][[0, 6, 2, 4]] *= -1
RESCU2DFTIO[4][[1, 7, 3, 5]] *= -1
RESCU2DFTIO[5][[0, 10, 8, 2, 6, 4]] *= -1



SIESTA2DFTIO = {
            0: np.eye(1, dtype=np.float32),
            1: np.eye(3, dtype=np.float32),
            2: np.eye(5, dtype=np.float32),
            3: np.eye(7, dtype=np.float32),
            4: np.eye(9, dtype=np.float32),
            5: np.eye(11, dtype=np.float32)
        }

SIESTA2DFTIO[1][[0, 2]] *= -1
SIESTA2DFTIO[2][[1, 3]] *= -1
SIESTA2DFTIO[3][[0, 6, 2, 4]] *= -1
SIESTA2DFTIO[4][[1, 7, 3, 5]] *= -1
SIESTA2DFTIO[5][[0, 10, 8, 2, 6, 4]] *= -1

OPENMX_orbital_number_m = {
    "s": [0],
    "p": [1, -1, 0],
    "d": [0, 2, -2, 1, -1],
    "f": [0, 1, -1, 2, -2, 3, -3]
}

OPENMX2DFTIO = {
            0: np.eye(1, dtype=np.float32),
            1: np.eye(3, dtype=np.float32)[[1, 2, 0]],
            2: np.eye(5, dtype=np.float32)[[2, 4, 0, 3, 1]],
            3: np.eye(7, dtype=np.float32)[[6, 4, 2, 0, 1, 3, 5]]
        }