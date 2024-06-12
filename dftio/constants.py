import numpy as np

orbitalId = {0:'s',1:'p',2:'d',3:'f',4:'g',5:'h'}
anglrMId = {'s':0,'p':1,'d':2,'f':3,'g':4,'h':5}
Bohr2Ang = 0.529177249

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