from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import re
from tqdm import tqdm
from collections import Counter
from ...constants import orbitalId, RESCU2DFTIO, anglrMId
import ase
import dpdata
import glob
import h5py
import os
import numpy as np
from ..parse import Parser, ParserRegister, find_target_line
from ...data import _keys
from ...register import Register

Hartree2eV = 27.21138602
Bohr2Angstrom = 0.52917721067

@ParserRegister.register("rescu")
class RescuParser(Parser):
    def __init__(self, root, prefix, **kwargs):
        super(RescuParser, self).__init__(root, prefix)
        # the root + prefix should locate the directory of the output file of each calculation
        # the list of path will be saved in self.raw_datas

    def calculation_type(self, output):
        with h5py.File(output, "r") as f:
            calT = "".join([chr(i) for i in f["info"]["calculationType"][:].reshape(-1)])

        return calT


    def get_structure(self, idx):
        global_lists = glob.glob(self[idx] + "/*.mat")
        for fs in global_lists:
            calT = self.calculation_type(fs)
            if calT == "self-consistent":
                path = fs
                break

        with h5py.File(path, "r") as f:
            pos = f["atom"]["xyz"][:].T.astype(np.float32)
            element_map = f["atom"]["element"][:].reshape(-1).astype(np.int32) - 1

            if isinstance(f["element"]["species"][:][0][0], np.uint16):
                species = ["".join([chr(i) for i in f["element"]["species"][:].reshape(-1)])]
            elif isinstance(f["element"]["species"][:][0][0], h5py.h5r.Reference):
                species = ["".join([chr(j) for j in f[i][:].reshape(-1)]) for i in f["element"]["species"][:].reshape(-1)]
            ans = np.array([ase.atom.atomic_numbers[i] for i in species], dtype=np.int32)
            atomic_number = ans[element_map]

            pbc = []
            for i in f["domain"]["boundary"][:].flatten():
                if i == 1:
                    pbc.append(True)
                elif i == 3:
                    pbc.append(False)
                else:
                    raise ValueError("Unknown boundary condition")
            pbc = np.array(pbc)

            cell = f["domain"]["latvec"][:].T.astype(np.float32)
        
        structure = {
            _keys.ATOMIC_NUMBERS_KEY: atomic_number,
            _keys.PBC_KEY: pbc,
            _keys.POSITIONS_KEY: pos.reshape(1, -1, 3) * Bohr2Angstrom,
            _keys.CELL_KEY: cell.reshape(1,3,3) * Bohr2Angstrom
        }

        return structure
    
    def get_eigenvalue(self, idx):
        global_lists = glob.glob(self[idx] + "/*.mat")
        for fs in global_lists:
            calT = self.calculation_type(fs)
            if calT == "band-structure":
                path = fs
                break
        with h5py.File(path, "r") as f:
            kpt = f["band"]["kdirect"][:].T[np.newaxis, :, :]
            eigs = f["band"]["ksnrg"][:].T
        
        return {_keys.KPOINT_KEY: kpt, _keys.ENERGY_EIGENVALUE_KEY: eigs}
    
    def get_basis(self, idx):
        global_lists = glob.glob(self[idx] + "/*.mat")
        for fs in global_lists:
            calT = self.calculation_type(fs)
            if calT == "self-consistent":
                path = fs
                break
        with h5py.File(path, "r") as f:
            Aorb = f["LCAO"]["orbInfo"]["Aorb"][:].flatten()
            Lorb = f["LCAO"]["orbInfo"]["Lorb"][:].flatten()

        stru = self.get_structure(idx=idx)
        atomic_number = stru[_keys.ATOMIC_NUMBERS_KEY]
        an = list(set(atomic_number))
        an_ind = [atomic_number.tolist().index(i)+1 for i in an]# since Aorb info start from 1
        basis = {}
        for i, at in enumerate(an_ind):
            ix = np.where(Aorb == at)[0]
            basis[ase.atom.chemical_symbols[an[i]]] = Lorb[ix]
        
        for k in basis:
            bb = basis[k]
            bc = Counter(bb)
            bs = ""
            for l in sorted(bc.keys()):
                bs += str(int(bc[l] / (2*l+1))) + orbitalId[l]
            basis[k] = bs
        
        return basis
    
    def get_blocks(self, idx, hamiltonian: bool = False, overlap: bool = False, density_matrix: bool = False):
        # rescu use Spherical Harmonic basis with the phase factor, and the order of basis is defined in the form from small to large,
        # i.e. for p orbital, the m is ranked as [-1, 0, 1]
        global_lists = glob.glob(self[idx] + "/*.h5")
        for fs in global_lists:
            with h5py.File(fs, "r") as f:
                if "LCAO" in f and "hamiltonian1" in f["LCAO"]:
                    Rvec = f["LCAO"]["Rvec"][:].T
                    path = fs
                    break

        if any([hamiltonian, overlap, density_matrix]):
            basis = self.get_basis(idx=idx)
        else:
            return [{}], [{}], [{}]
        
        l_dict = {}
        # count norbs
        count = {}
        for at in basis:
            count[at] = 0
            l_dict[at] = []
            for iorb in range(int(len(basis[at]) / 2)):
                n, o = int(basis[at][2*iorb]), basis[at][2*iorb+1]
                count[at] += n * (2*anglrMId[o]+1)
                l_dict[at] += [anglrMId[o]] * n
        
        an = self.get_structure(idx=idx)[_keys.ATOMIC_NUMBERS_KEY]

        hamiltonian_dict = {}
        overlap_dict = {}
        density_matrix = {}

        
        with h5py.File(path, "r") as f:
            if hamiltonian:
                hamil = []
                hamil_mask = []
                for i in range(Rvec.shape[0]):
                    if np.abs(f["LCAO"]["hamiltonian" + str(i+1)][:]).max() > 1e-8:
                        hamil.append(f["LCAO"]["hamiltonian" + str(i+1)][:].T)
                        hamil_mask.append(True)
                    else:
                        hamil_mask.append(False)
                hamil_mask = np.array(hamil_mask)
                hamil = np.stack(hamil).astype(np.float32) * Hartree2eV
                hamil_Rvec = Rvec[hamil_mask]
                xcount = 0
                for i, ai in enumerate(an):
                    si = ase.atom.chemical_symbols[ai]
                    ycount = 0
                    for j, aj in enumerate(an):
                        sj = ase.atom.chemical_symbols[aj]
                        keys = map(lambda x: "_".join([str(i),str(j),str(x[0].astype(np.int32)),str(x[1].astype(np.int32)),str(x[2].astype(np.int32))]), hamil_Rvec)
                        blocks = self.transform(hamil[:, xcount:xcount+count[si], ycount:ycount+count[sj]], l_dict[si], l_dict[sj])

                        hamiltonian_dict.update(dict(zip(keys, blocks)))

                        ycount += count[sj]
                    xcount += count[si]
            

            if overlap:
                ovp = []
                ovp_mask = []
                for i in range(Rvec.shape[0]):
                    if np.abs(f["LCAO"]["overlap" + str(i+1)][:]).max() > 1e-8:
                        ovp.append(f["LCAO"]["overlap" + str(i+1)][:].T)
                        ovp_mask.append(True)
                    else:
                        ovp_mask.append(False)
                ovp_mask = np.array(ovp_mask)
                ovp = np.stack(ovp).astype(np.float32)
                ovp_Rvec = Rvec[ovp_mask]
                xcount = 0
                for i, ai in enumerate(an):
                    si = ase.atom.chemical_symbols[ai]
                    ycount = 0
                    for j, aj in enumerate(an):
                        sj = ase.atom.chemical_symbols[aj]
                        keys = map(lambda x: "_".join([str(i),str(j),str(x[0].astype(np.int32)),str(x[1].astype(np.int32)),str(x[2].astype(np.int32))]), ovp_Rvec)
                        blocks = self.transform(ovp[:, xcount:xcount+count[si], ycount:ycount+count[sj]], l_dict[si], l_dict[sj])
                        overlap_dict.update(dict(zip(keys, blocks)))

                        ycount += count[sj]
                    xcount += count[si]

            if density_matrix:
                raise NotImplementedError("Density matrix is not implemented yet.")
            
        return [hamiltonian_dict], [overlap_dict], [density_matrix]

    def transform(self, mat, l_lefts, l_rights):

        if max(*l_lefts, *l_rights) > 5:
            raise NotImplementedError("Only support l = s, p, d, f, g, h.")
        block_lefts = block_diag(*[RESCU2DFTIO[l_left] for l_left in l_lefts])
        block_rights = block_diag(*[RESCU2DFTIO[l_right] for l_right in l_rights])

        return block_lefts @ mat @ block_rights.T