from dftio.op import SingleGridIntegrator, make_uniform_grid, make_simple_grid
from dftio.datastruct import AtomicBasis
from math import ceil, floor
from dftio.constants import atomic_numbers_r
import torch

class LDOS:
    def __init__(
            self, 
            atomicbasis: AtomicBasis, 
            atomic_numbers, 
            pbc, 
            cell, 
            coordinates, 
            grids,
            z_valence=None,
            nspin=2,
            skin=0.0, 
            sorted=False, 
            use_scaled_positions=False,
            dtype=torch.float32,
            ):

        self.atomic_basis = atomicbasis
        self.atomic_numbers = atomic_numbers
        self.pbc = pbc
        self.cell = cell
        self.coordinates = coordinates
        self.grids = grids
        self.z_valence = z_valence
        self.nspin = nspin
        self.dtype=dtype

        self.sgint = SingleGridIntegrator(
            atomic_numbers=atomic_numbers, 
            pbc=pbc, 
            cell=cell, 
            coordinates=coordinates, 
            grids=grids, 
            atomic_basis=atomicbasis, 
            skin=skin,
            sorted=sorted, 
            use_scaled_positions=use_scaled_positions,
            dtype=dtype
        )

        self.natoms = len(atomic_numbers)
        self.n_valbands = []
        if z_valence is not None:
            self.n_valbands = sum([z_valence[atomic_numbers_r[i]] for i in self.atomic_numbers]) / self.nspin
        
    def get(self, E: float, coefficients: torch.Tensor, eigenvalues: torch.Tensor, sigma: float=0.1):
        """Compute the local density of states at energy E

        Parameters
        ----------
        E : float
            _description_
        coefficients : torch.Tensor
            _description_
        eigenvalues : torch.Tensor
            _description_
        bias : float, optional
            _description_, by default 0.0
        sigma : float, optional
            _description_, by default 0.1

        Returns
        -------
        _type_
            _description_
        """

        ldos = torch.zeros(self.grids.shape[0])
        
        

        k, n, m = coefficients.shape # [nk, nbands, norbs]
        # compute E_fermi if self.n_valbands is available
        if self.n_valbands:
            neigvalan = k * self.n_valbands
            eigsort = eigenvalues.reshape(-1).sort().values
            if ceil(neigvalan) - neigvalan < 1e-6: # int
                E_fermi = eigsort[floor(neigvalan-1)] + eigsort[ceil(neigvalan+1)]
                E_fermi /= 2
            else:
                E_fermi = eigsort[floor(neigvalan)]
            
        else:
            E_fermi = 0.

        print("Computed fermi energy: ", E_fermi)
        

        E_range = [E-5*sigma, E+5*sigma]
        assert n == eigenvalues.shape[1] and k == eigenvalues.shape[0] and len(eigenvalues.shape)==2, "Number of bands and kpoints of coeff must be the same as the number of eigenvalues"
        if self.n_valbands:
            assert n >= self.n_valbands, f"Number of bands must be at least {self.n_valbands}"
            n = self.n_valbands
        else:
            print("Warning: Number of valence bands not provided. All input coeff and eigenvalues are considered corresponding to valence bands.")
        assert m >= n

        coefficients = coefficients[:,:n]
        eigenvalues = eigenvalues[:,:n]

        mask = torch.logical_and(eigenvalues > E_range[0], eigenvalues < E_range[1])
        mask = torch.logical_and(mask, coefficients.norm(dim=2) > 1e-6)

        for ik in range(k):
            for ib in range(n):
                if mask[ik, ib]:
                    ll = self.nspin * self.sgint.integrate(weights=coefficients[ik, ib])
                    ldos += (ll * ll.conj()).real * torch.exp(-0.5*((eigenvalues[ik, ib] - E) / sigma) ** 2)

        return ldos / k
    
    def get_wbias(self, coefficients: torch.Tensor, eigenvalues: torch.Tensor, bias: float=0.0):
        """Compute the local density of states at energy E

        Parameters
        ----------
        coefficients : torch.Tensor
            _description_
        eigenvalues : torch.Tensor
            _description_
        bias : float, optional
            _description_, by default 0.0
        sigma : float, optional
            _description_, by default 0.1

        Returns
        -------
        _type_
            _description_
        """
        ldos = torch.zeros(self.grids.shape[0])
        
        k, n, m = coefficients.shape
        # compute E_fermi if self.n_valbands is available
        if self.n_valbands:
            neigvalan = k * self.n_valbands
            eigsort = eigenvalues.reshape(-1).sort().values
            if ceil(neigvalan) - neigvalan < 1e-6: # int
                E_fermi = eigsort[floor(neigvalan-1)] + eigsort[ceil(neigvalan+1)]
                E_fermi /= 2
            else:
                E_fermi = eigsort[floor(neigvalan)]
            
        else:
            E_fermi = 0.

        print("Computed fermi energy: ", E_fermi)
        
        eigenvalues -= E_fermi

        E_range = [min(0.0, bias), max(0.0, bias)]

        assert n == eigenvalues.shape[1] and k == eigenvalues.shape[0] and len(eigenvalues.shape)==2, "Number of bands and kpoints of coeff must be the same as the number of eigenvalues"
        # if self.n_valbands:
        #     assert n >= self.n_valbands, f"Number of bands must be at least {self.n_valbands}"
        #     n = self.n_valbands
        # else:
        #     print("Warning: Number of valence bands not provided. All input coeff and eigenvalues are considered corresponding to valence bands.")
        assert m >= n

        # coefficients = coefficients[:,:n]
        # eigenvalues = eigenvalues[:,:n]

        mask = torch.logical_and(eigenvalues > E_range[0], eigenvalues < E_range[1])
        mask = torch.logical_and(mask, coefficients.norm(dim=2) > 1e-6)

        for ik in range(k):
            for ib in range(n):
                if mask[ik, ib]:
                    ll = self.nspin * self.sgint.integrate(weights=coefficients[ik, ib])
                    ldos += (ll * ll.conj()).real

        return ldos / k
    
    def scan(self, ldos_wbias, current: float):

        # here we assert the z axis are perpendicular to x and y direction
        assert len(ldos_wbias.shape) == 3, "scan only works for 3D LDOS"

        avg_left_current = ldos_wbias[:,:,0].mean()
        avg_right_current = ldos_wbias[:,:,-1].mean()
        # if left current is higher than right current, the calculation step would need a little correction
        left = True
        if avg_left_current > avg_right_current:
            left = False

        z0 = self.grids[:,2].min()
        z1 = self.grids[:,2].max()
        nz = ldos_wbias.shape[-1]
        dz = (z1 - z0) / (nz - 1)

        if left:

            # compute derivative of the current
            ldos_wbias_diff = ldos_wbias[:,:,1:] - ldos_wbias[:,:,:-1]

            # filter the negative ones
            mask = torch.zeros_like(ldos_wbias_diff[:,:,0], dtype=torch.bool)
            for i in range(1,nz):
                mask = torch.logical_or(mask, ldos_wbias_diff[:,:,i] < 0)
                ldos_wbias[:,:,i][mask] = 100

            # min_up = ldos_wbias[:,:,nz-1].min()
            # max_down = ldos_wbias[:,:,0].max()
        else:

            # compute derivative of the current
            ldos_wbias_diff = ldos_wbias[:,:,:-1] - ldos_wbias[:,:,1:]

            mask = torch.zeros_like(ldos_wbias_diff[:,:,0], dtype=torch.bool)
            for i in range(2,nz+1):
                mask = torch.logical_or(mask, ldos_wbias_diff[:,:,nz-i] < 0)
                ldos_wbias[:,:,nz-i][mask] = 100

            # min_up = ldos_wbias[:,:,0].min()
            # max_down = ldos_wbias[:,:,nz-1].max()

        # assert max_down <= current <= min_up, "Current is outside the energy range ({0}, {1})".format(max_down, min_up)

        current_center = ldos_wbias - current
        steps = current_center.abs().argmin(dim=-1)

        if left:
            assert steps.gt(0).all(), "Current is outside the energy range (too low)"
            assert steps.lt(nz-1).all(), "Current is outside the energy range (too high)"
        else:
            assert steps.gt(0).all(), "Current is outside the energy range (too high)"
            assert steps.lt(nz-1).all(), "Current is outside the energy range (too low)"
        
        delta_current = torch.gather(input=current_center, index=steps.unsqueeze(-1), dim=-1).squeeze(-1)

        if left:
            left_steps = steps - delta_current.gt(0).long()
            right_steps = steps + delta_current.le(0).long()
        else:
            left_steps = steps - delta_current.le(0).long()
            right_steps = steps + delta_current.gt(0).long()

        left_ldos = torch.gather(input=ldos_wbias, index=left_steps.unsqueeze(-1), dim=-1).squeeze(-1)
        right_ldos = torch.gather(input=ldos_wbias, index=right_steps.unsqueeze(-1), dim=-1).squeeze(-1)

        if left:
            z = (nz - right_steps) * dz + dz / (right_ldos - left_ldos) * (right_ldos - current) 
        else:
            z = left_steps * dz + dz / (left_ldos - right_ldos) * (left_ldos - current)

        return z # here z is the relative height to the lowest scanning surface defined by the grid

