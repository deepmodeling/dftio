import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc
from ..op.make_grid import make_simple_grid
from .neighbourlist import PrimitiveFieldsNeighborList
import torch

def _getline(cube):
    """Read a line from cube file where first field is an int
    and the remaining fields are floats.

    Parameters
    ----------
    cube :
        file object of the cube file

    Returns
    -------
    type
        

    """
    line = cube.readline().strip().split()
    return int(line[0]), np.array(list(map(float, line[1:])))


def read_cube(fname):
    """Read cube file into numpy array

    Parameters
    ----------
    fname :
        filename of cube file

    Returns
    -------
    type
        

    """
    bohr = 0.529177
    meta = {}
    with open(fname, 'r') as cube:
        cube.readline()
        cube.readline()  # ignore comments
        natm, meta['org'] = _getline(cube)
        nx, meta['xvec'] = _getline(cube)
        ny, meta['yvec'] = _getline(cube)
        nz, meta['zvec'] = _getline(cube)

        if nx > 0:
            meta['xvec'] = meta['xvec'] * bohr * nx
        if ny > 0:
            meta['yvec'] = meta['yvec'] * bohr * ny
        if nz > 0:
            meta['zvec'] = meta['zvec'] * bohr * nz

        meta['atoms'] = [_getline(cube) for i in range(natm)]
        data = np.zeros((nx * ny * nz))
        idx = 0
        for line in cube:
            for val in line.strip().split():
                data[idx] = float(val)
                idx += 1

    data = np.reshape(data, (nx, ny, nz))
    return data, meta


class Field(object):
    """ """

    def __init__(self, data, cell, na, nb, nc, pos: torch.Tensor=None, atomic_numbers: torch.Tensor=None, origin=(0, 0, 0)):
        self._origin_shift = torch.Tensor([0.0, 0.0, 0.0])
        self._cube = []
        self._origin_has_changed = False
        self._rot_mat = []
        self._atoms = []
        self._cell = cell
        self.na = na
        self.nb = nb
        self.nc = nc

        if isinstance(data, torch.Tensor):
            data = data.numpy()
        elif isinstance(data, list):
            data = np.array(data)
        else:
            assert isinstance(data, np.ndarray), "data must be a numpy array, torch.Tensor, or list"


        assert data.shape == (na, nb, nc), "data is not the same shape as the grid"
        
        x = torch.linspace(0., 1., na)
        y = torch.linspace(0., 1., nb)
        z = torch.linspace(0., 1., nc)

        self.origin = torch.Tensor(origin)
        self._atomic_numbers = atomic_numbers
        self._pos = pos

        data[data > 100] = 100
        data[data < -100] = -100
        self._interpolant = RegularGridInterpolator((x, y, z), data, bounds_error=False)



    @classmethod
    def from_cube(cls, path):

        assert path.endswith('.cube') or path.endswith('.cub'), "The input file path is not a cube file."

        data, meta = read_cube(path)
        _cell = torch.as_tensor(np.array([meta['xvec'], meta['yvec'], meta['zvec']]), dtype=torch.float32)
        na, nb, nc = data.shape

        pos = torch.as_tensor(np.array([atom[1][1:] for atom in meta['atoms']]), dtype=torch.float32)
        ans = torch.as_tensor(np.array([atom[0] for atom in meta['atoms']]), dtype=torch.float32)

        return cls(
                    data, 
                    cell=_cell, 
                    na=na, 
                    nb=nb, 
                    nc=nc, 
                    pos=pos, 
                    atomic_numbers=ans, 
                    origin=(0, 0, 0)
                )

    def set_origin(self, origin):
        """Set the coordinates of the center of the molecule

        Parameters
        ----------
        origin :
            return:

        Returns
        -------

        """

        if isinstance(origin, list):
            origin = torch.Tensor(origin)

        self._origin_shift = origin + self._origin_shift - torch.Tensor(self.origin)
        self._origin_has_changed = True

    def rotate(self, axis, theta):
        """Set the coordinates of the center of the molecule

        Parameters
        ----------
        origin :
            return:
        axis :
            
        theta :
            

        Returns
        -------

        """
        if axis == 'x':
            rot_mat = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(theta), -np.sin(theta)],
                                [0.0, np.sin(theta), np.cos(theta)]])
            rot_mat = torch.as_tensor(rot_mat, dtype=torch.float32)
        elif axis == 'y':
            rot_mat = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                                [0.0, 1.0, 0.0],
                                [-np.sin(theta), 0.0, np.cos(theta)]])
            rot_mat = torch.as_tensor(rot_mat, dtype=torch.float32)
        elif axis == 'z':
            rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                                [np.sin(theta), np.cos(theta), 0.0],
                                [0.0, 0.0, 1.0]])
            rot_mat = torch.as_tensor(rot_mat, dtype=torch.float32)
        else:
            raise ValueError('Wrong axis')

        self._rot_mat.append(rot_mat)

    def reset_rotations(self):
        """ """

        self._rot_mat = []

    def _transform(self, coords1, translate):
        """

        Parameters
        ----------
        coords1 :
            
        translate :
            

        Returns
        -------

        """

        if isinstance(coords1, list):
            coords = np.array(coords1)
        else:
            coords = coords1
        
        coords = torch.as_tensor(coords, dtype=torch.float32)

        if len(coords.shape) < 2:
            coords = coords.unsqueeze(0)

        coords += self.origin.reshape(1, 3)

        if self._origin_has_changed:
            coords = coords + self._origin_shift.reshape(1, 3)

        if isinstance(translate, np.ndarray):
            coords = coords - torch.as_tensor(np.squeeze(translate), dtype=torch.float32)

        if len(self._rot_mat) > 0:
            for item in self._rot_mat:
                coords = (item @ coords.T).T

        return coords

    def _inv_transform(self, coords1, translate):
        """

        Parameters
        ----------
        coords1 :
            
        translate :
            

        Returns
        -------

        """
        if isinstance(coords1, list):
            coords = np.array(coords1)
        else:
            coords = coords1

        coords = torch.as_tensor(coords, dtype=torch.float32)

        if len(coords.shape) < 2:
            coords = coords.unsqueeze(0)

        if len(self._rot_mat) > 0:
            for item in reversed(self._rot_mat):
                coords = (torch.linalg.inv(item) @ coords.T).T

        if isinstance(translate, np.ndarray):
            translate = torch.as_tensor(translate, dtype=torch.float32)
        else:
            translate = 0.

        coords = coords + translate

        if self._origin_has_changed:
            coords = coords - self._origin_shift.reshape(1, 3)

        coords -= self.origin.reshape(1, 3)

        return coords

    def __call__(self, coords1, translate=None):
        """

        Parameters
        ----------
        coords1 :
            
        translate :
             (Default value = None)

        Returns
        -------

        """

        # get the coordinates in orginal grid
        coords = self._transform(coords1, translate)
        
        # get the fractional coordinates
        coords = coords @ torch.linalg.inv(self._cell)

        # transform to the unit_cell
        values = torch.as_tensor(self._interpolant(coords - coords.floor()), dtype=coords.dtype)

        return torch.nan_to_num(values)

    def integrate(self, field):
        pass

    @property
    def atomic_numbers(self):
        """ """
        assert self._atomic_numbers is not None, 'Atomic numbers not set'
        assert isinstance(self._atomic_numbers, torch.Tensor), 'Atomic numbers must be a numpy array'

        return self._atomic_numbers
    
    @property
    def positions(self):
        """ """

        assert self._pos is not None, 'Positions not set'
        assert isinstance(self._pos, torch.Tensor) and len(self._pos.shape) == 2, 'Positions must be a numpy array'

        return self._pos
    
    @property
    def grids(self):
        """ """

        return make_simple_grid(cell=self._cell, nx=self.na, ny=self.nb, nz=self.nc)[1].reshape(self.na, self.nb, self.nc, 3)