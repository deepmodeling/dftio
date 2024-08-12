from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import re
from tqdm import tqdm
from collections import Counter
from ...constants import orbitalId, SIESTA2DFTIO,anglrMId
import ase
import dpdata
import os
import numpy as np
from collections import Counter
from ..parse import Parser, ParserRegister, find_target_line
from ...data import _keys
import sisl


@ParserRegister.register("siesta")
class SiestaParser(Parser):
    def __init__(
            self,
            root,
            prefix,
            **kwargs
            ):
        super(SiestaParser, self).__init__(root, prefix)
               

    def find_content(self,path,str_to_find):
        # 用于存储包含SystemLabel标签的文件及其内容
        fdf_files_with_system_label_content = {}
        file_path = None
        system_label_content = None
        # 遍历给定路径及其子目录
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.fdf'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # 使用正则表达式匹配SystemLabel及其后面的内容
                            match = re.search(r'\b'+str_to_find+r'\b\s*(\S+)', content)
                            if match:
                                system_label_content = match.group(1)
                                fdf_files_with_system_label_content[file_path] = system_label_content
                                break
                    except:
                        print(f"don't find {str_to_find} in {file_path}")
        
        if system_label_content is None:
            print(f"don't find {str_to_find} in {file_path}, use the default value: siesta")

        return file_path, system_label_content   


    # essential
    def get_structure(self,idx):
        path = self.raw_datas[idx]
        struct,_ = self.find_content(path= path,str_to_find='AtomicCoordinatesAndAtomicSpecies')
        struct = sisl.get_sile(struct).read_geometry()
        structure = {
            _keys.ATOMIC_NUMBERS_KEY: np.array([struct.atoms[i].Z for i in range(struct.na)], dtype=np.int32),
            _keys.PBC_KEY: np.array([True, True, True]) # abacus does not allow non-pbc structure
        }
        structure[_keys.POSITIONS_KEY] = struct.xyz.astype(np.float32)[np.newaxis, :, :]
        structure[_keys.CELL_KEY] = struct.cell.astype(np.float32)[np.newaxis, :, :]

        return structure
    
    # essential
    def get_eigenvalue(self, idx):
        pass
    
    # essential
    def get_basis(self,idx):
        # {"Si": "2s2p1d"}
        path = self.raw_datas[idx]
        _,system_label = self.find_content(path=path,str_to_find='SystemLabel')
        if system_label is None:
            system_label = "siesta"

        # tshs = self.raw_datas[idx]+ "/"+system_label+".TSHS"
        # hamil =  sisl.Hamiltonian.read(tshs)
        ORB_INDX = self.raw_datas[idx]+ "/"+system_label+".ORB_INDX"
        ORB_INDX  = sisl.get_sile(ORB_INDX ).read_basis()
        na = len(ORB_INDX)
        basis_siesta = {}
        basis = {}
        for i in range(na):
            if ORB_INDX[i].tag not in basis_siesta.keys():
                basis_siesta[ORB_INDX[i].tag] = []
                for j in range(ORB_INDX[i].no):
                    basis_siesta[ORB_INDX[i].tag].append(ORB_INDX[i].orbitals[j].name())


        for atom_type in basis_siesta.keys():
            split_basis = []
            for i in range(len(basis_siesta[atom_type])):
                split_basis.append(list(basis_siesta[atom_type][i])[1])
            
            counted_basis = Counter(split_basis)
            
            counted_basis_list = []
            for basis_type in counted_basis.keys():
                if basis_type == 's':
                    counted_basis_list.append(str(int(counted_basis['s']/1))+'s')
                elif basis_type == 'p':
                    assert abs(counted_basis['p']%3)<1e-6, "p orbital is not multiple of 3"
                    counted_basis_list.append(str(int(counted_basis['p']/3))+'p')
                elif basis_type == 'd':
                    assert abs(counted_basis['d']%5)<1e-6, "d orbital is not multiple of 5"
                    counted_basis_list.append(str(int(counted_basis['d']/5))+'d')
                elif basis_type == 'f':
                    assert abs(counted_basis['f']%7)<1e-6, "f orbital is not multiple of 7"
                    counted_basis_list.append(str(int(counted_basis['f']/7))+'f')
            
            basis[atom_type] = "".join(counted_basis_list)
        
        return basis


    # essential
    def get_blocks(self, idx, hamiltonian: bool = False, overlap: bool = False, density_matrix: bool = False):
        path = self.raw_datas[idx]
        _,system_label = self.find_content(path=self.raw_datas[idx],str_to_find='SystemLabel')
        if system_label is None:
            system_label = "siesta"
        hamiltonian_dict, overlap_dict, density_matrix_dict = None, None, None
        struct,_ = self.find_content(path= path,str_to_find='AtomicCoordinatesAndAtomicSpecies')
        struct = sisl.get_sile(struct).read_geometry()
        na = struct.na
        element = [struct.atoms[i].Z for i in range(struct.na)]
        
        tshs = path+ "/"+system_label+".TSHS"
        if os.path.exists(tshs):
            hamil =  sisl.Hamiltonian.read(tshs)
        else:
            raise FileNotFoundError("Hamiltonian file not found.")        
        site_norbits = np.array([hamil.atoms[i].no for i in range(hamil.na)])
        site_norbits_cumsum = site_norbits.cumsum()

        basis = self.get_basis(idx)      
        spinful = False #TODO: add support for spinful


        central_cell = [int(np.floor(hamil.nsc[i]/2)) for i in range(3)]
        Rvec_list = []
        for rx in range(central_cell[0],hamil.nsc[0]):
            for ry in range(hamil.nsc[1]):
                for rz in range(hamil.nsc[2]):
                    Rvec_list.append([rx-central_cell[0],ry-central_cell[1],rz-central_cell[2]])
        Rvec = np.array(Rvec_list)


        hamiltonian_dict = {}
        overlap_dict = {}
        density_matrix_dict = {}


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

        cut_tol_ham = 1e-5
        cut_tol_ovp = 1e-5
        cut_tol_dm = 1e-5


        if hamiltonian:
           
            hamil_csr = hamil.tocsr()
            hamil_blocks = []
            hamil_mask = []
            for i in range(Rvec.shape[0]):
                off = hamil.geometry.sc_index(Rvec[i]) * hamil.geometry.no
                if np.abs(hamil_csr[:,off:off+hamil.geometry.no].toarray()).max() > cut_tol_ham:
                    hamil_mask.append(True)
                    hamil_blocks.append(hamil_csr[:,off:off+hamil.geometry.no].toarray())
                else:
                    hamil_mask.append(False)

            hamil_mask = np.array(hamil_mask)
            hamil_Rvec = Rvec[hamil_mask]
            hamil_blocks = np.stack(hamil_blocks).astype(np.float32)

            for i in range(na):
                si = ase.atom.chemical_symbols[element[i]]
                for j in range(na):
                    sj = ase.atom.chemical_symbols[element[j]]
                    keys = map(lambda x: "_".join([str(i),str(j),str(x[0].astype(np.int32)),\
                                str(x[1].astype(np.int32)),str(x[2].astype(np.int32))]), hamil_Rvec)
                    i_norbs = site_norbits[i]
                    i_orbs_start =site_norbits_cumsum[i] - i_norbs
                    j_norbs = site_norbits[j]
                    j_orbs_start =site_norbits_cumsum[j] - j_norbs
                    block = self.transform(hamil_blocks[:,i_orbs_start:i_orbs_start+i_norbs,j_orbs_start:j_orbs_start+j_norbs],\
                                            l_dict[si], l_dict[sj])
                    # block = hamil_blocks[:,i_orbs_start:i_orbs_start+i_norbs,j_orbs_start:j_orbs_start+j_norbs]
                    block_mask = np.abs(block).max(axis=(1,2)) > cut_tol_ham

                    if np.any(block_mask):
                        keys = list(keys)
                        keys = [keys[k] for k,t in enumerate(block_mask) if t]
                        hamiltonian_dict.update(dict(zip(keys, block[block_mask])))
            
        if overlap:
            if os.path.exists(tshs):
                ovp =  sisl.Overlap.read(tshs)
            else:
                raise FileNotFoundError("Overlap file not found.")
            
            ovp_csr = ovp.tocsr()
            ovp_blocks = []
            ovp_mask = []
            for i in range(Rvec.shape[0]):
                off = ovp.geometry.sc_index(Rvec[i]) * ovp.geometry.no
                if np.abs(ovp_csr[:,off:off+ovp.geometry.no].toarray()).max() > cut_tol_ovp:
                    ovp_mask.append(True)
                    ovp_blocks.append(ovp_csr[:,off:off+ovp.geometry.no].toarray())
                else:
                    ovp_mask.append(False)
                
            ovp_blocks = np.stack(ovp_blocks).astype(np.float32)
            ovp_mask = np.array(ovp_mask)
            ovp_Rvec = Rvec[ovp_mask]

            for i in range(na):
                si = ase.atom.chemical_symbols[element[i]]
                for j in range(na):
                    sj = ase.atom.chemical_symbols[element[j]]
                    keys = map(lambda x: "_".join([str(i),str(j),str(x[0].astype(np.int32)),\
                                str(x[1].astype(np.int32)),str(x[2].astype(np.int32))]), ovp_Rvec)
                    i_norbs = site_norbits[i]
                    i_orbs_start =site_norbits_cumsum[i] - i_norbs
                    j_norbs = site_norbits[j]
                    j_orbs_start =site_norbits_cumsum[j] - j_norbs
                    block = self.transform(ovp_blocks[:,i_orbs_start:i_orbs_start+i_norbs,j_orbs_start:j_orbs_start+j_norbs],\
                                             l_dict[si], l_dict[sj])
                    # block = ovp_blocks[:,i_orbs_start:i_orbs_start+i_norbs,j_orbs_start:j_orbs_start+j_norbs]
                    block_mask = np.abs(block).max(axis=(1,2)) > cut_tol_ovp

                    if np.any(block_mask):
                        keys = list(keys)
                        keys = [keys[k] for k,t in enumerate(block_mask) if t]
                        overlap_dict.update(dict(zip(keys, block[block_mask])))

        if density_matrix:
            _,system_label = self.find_content(path=self.raw_datas[idx],str_to_find='SystemLabel')
            if system_label is None:
                system_label = "siesta"
            DM_path = self.raw_datas[idx]+ "/"+system_label+".DM"
            if os.path.exists(DM_path):
                DM =  sisl.DensityMatrix.read(DM_path)
            else:
                raise FileNotFoundError("Density Matrix file not found.")
            
            DM_csr = DM.tocsr()
            DM_blocks = []
            DM_mask = []
            for i in range(Rvec.shape[0]):
                off = DM.geometry.sc_index(Rvec[i]) * DM.no
                if np.abs(DM_csr[:,off:off+DM.geometry.no].toarray()).max() > cut_tol_dm:
                    DM_mask.append(True)
                    DM_blocks.append(DM_csr[:,off:off+DM.geometry.no].toarray())
                else:
                    DM_mask.append(False)
                
            DM_blocks = np.stack(DM_blocks).astype(np.float32)
            DM_mask = np.array(DM_mask)
            DM_Rvec = Rvec[DM_mask]

            for i in range(na):
                si = ase.atom.chemical_symbols[element[i]]
                for j in range(na):
                    sj = ase.atom.chemical_symbols[element[j]]
                    keys = map(lambda x: "_".join([str(i),str(j),str(x[0].astype(np.int32)),\
                                str(x[1].astype(np.int32)),str(x[2].astype(np.int32))]), DM_Rvec)
                    i_norbs = site_norbits[i]
                    i_orbs_start =site_norbits_cumsum[i] - i_norbs
                    j_norbs = site_norbits[j]
                    j_orbs_start =site_norbits_cumsum[j] - j_norbs
                    block = self.transform(DM_blocks[:,i_orbs_start:i_orbs_start+i_norbs,j_orbs_start:j_orbs_start+j_norbs],\
                                            l_dict[si], l_dict[sj])
                    
                    block_mask = np.abs(block).max(axis=(1,2)) > cut_tol_dm

                    if np.any(block_mask):
                        keys = list(keys)
                        keys = [keys[k] for k,t in enumerate(block_mask) if t]
                        density_matrix_dict.update(dict(zip(keys, block)))

            
        
        return [hamiltonian_dict], [overlap_dict], [density_matrix_dict]
    

    
    def transform(self, mat, l_lefts, l_rights):
        # ssppd   l_lefts=[0,0,1,1,2] l_rights=[0,0,1,1,2]

        if max(*l_lefts, *l_rights) > 5:
            raise NotImplementedError("Only support l = s, p, d, f, g, h.")
        block_lefts = block_diag(*[SIESTA2DFTIO[l_left] for l_left in l_lefts])
        block_rights = block_diag(*[SIESTA2DFTIO[l_right] for l_right in l_rights])

        return block_lefts @ mat @ block_rights.T