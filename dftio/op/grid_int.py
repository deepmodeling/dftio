from ..datastruct import PrimitiveFieldsNeighborList
import torch
import ase.data as data
from torch_scatter import scatter_sum
import numpy as np

atomic_numbers_r = dict(zip(data.atomic_numbers.values(), data.atomic_numbers.keys()))

class SingleGridIntegrator:
    def __init__(self, atomic_numbers, pbc, cell, coordinates, grids, atomic_basis, skin=0.0, sorted=False, use_scaled_positions=False, dtype=torch.float32) -> None:
        self.atomic_basis = atomic_basis
        cutoffs = [self.atomic_basis[atomic_numbers_r[i]].rcut for i in atomic_numbers]
        self.nblist = PrimitiveFieldsNeighborList(cutoffs=cutoffs, sorted=sorted, skin=skin, use_scaled_positions=use_scaled_positions)
        self.atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long)
        self.pbc = pbc
        self.cell = torch.as_tensor(cell, dtype=dtype)
        self.coordinates = torch.as_tensor(coordinates, dtype=dtype)
        self.grids = torch.as_tensor(grids, dtype=dtype)
        # generate index, (grid, atom)
        self.index = [[],[]]
        self.dtype = dtype
        self.cell_shift = []

        self.nblist.update(pbc, cell, coordinates, grids)

        for i in range(len(self.nblist.neighbors)):
            self.index[0] += [i]*len(self.nblist.neighbors[i])
            self.index[1] += self.nblist.neighbors[i].tolist()
            self.cell_shift.append(self.nblist.displacements[i])

        self.index = torch.tensor(self.index, dtype=torch.long)
        self.cell_shift = torch.from_numpy(np.concatenate(self.cell_shift, axis=0, dtype=np.int32))

    def integrate(self, weights=None):
        
        ngrid = len(self.grids)
        results = torch.zeros(ngrid, dtype=weights.dtype)
        norbs = [self.atomic_basis[atomic_numbers_r[int(i)]].irreps.dim for i in self.atomic_numbers]
        cnorbs = torch.cumsum(torch.tensor([0]+norbs), dim=0)[:-1]
        for element in self.atomic_basis:
            # integrate over element i
            mask = self.atomic_numbers[self.index[1]].eq(data.atomic_numbers[element])
            rel_pos = self.coordinates[self.index[1][mask]] - self.grids[self.index[0][mask]] + self.cell_shift[mask].to(self.dtype) @ self.cell
            if weights is not None:
                
                assert len(weights) == sum(norbs) and len(weights.shape) == 1
                # here we assume that weights are arranged in one dimensional array, where the orders prioritize (atom index, angular momentum, magnetic momentum)
                weights_mask = torch.zeros(len(weights), dtype=torch.bool)
                
                for i in cnorbs[self.atomic_numbers.eq(data.atomic_numbers[element])]:
                    weights_mask[i:i+self.atomic_basis[element].irreps.dim] = True
                
                # turn the index of one element into the element order index
                n_eatoms = self.atomic_numbers.eq(data.atomic_numbers[element]).sum()
                convert_mask = -torch.ones_like(self.atomic_numbers)
                convert_mask[self.atomic_numbers.eq(data.atomic_numbers[element])] = torch.arange(n_eatoms)

                results += scatter_sum(
                    self.atomic_basis[element](rel_pos) * weights[weights_mask].reshape(-1, self.atomic_basis[element].irreps.dim)[convert_mask[self.index[1][mask]]], # the first term shaped [nrel_pos, norb]
                    self.index[0][mask],
                    dim=0,
                    dim_size=ngrid
                ).sum(-1)
            else:

                results += scatter_sum(
                    self.atomic_basis[element](rel_pos),
                    self.index[0][mask],
                    dim=0,
                    dim_size=ngrid
                ).sum(-1)
        
        return results



class DoubleGridIntegrator:
    def __init__(self) -> None:
        pass

    def integrate():
        pass