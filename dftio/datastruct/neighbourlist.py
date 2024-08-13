import numpy as np
import matplotlib.pyplot as plt
import os
from dptb.utils._xitorch.interpolate import Interp1D
import torch
import itertools
from scipy.interpolate import RegularGridInterpolator

import scipy.sparse.csgraph as csgraph
from scipy import sparse as sp
from scipy.spatial import cKDTree

from ase.cell import Cell
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import (
    complete_cell,
    find_mic,
    minkowski_reduce,
    wrap_positions,
)

class PrimitiveFieldsNeighborList:
    """Neighbor list that works without Atoms objects.

    This is less fancy, but can be used to avoid conversions between
    scaled and non-scaled coordinates which may affect cell offsets
    through rounding errors.

    Attributes
    ----------
    nupdates : int
        Number of updated times.
    """

    def __init__(self, cutoffs, skin=0.0, sorted=False, use_scaled_positions=False):
        self.cutoffs = np.asarray(cutoffs)
        self.skin = skin
        self.sorted = sorted
        self.nupdates = 0
        self.use_scaled_positions = use_scaled_positions
        self.nneighbors = 0
        self.npbcneighbors = 0

    def update(self, pbc, cell, coordinates, grids):
        """Make sure the list is up to date.

        Returns
        -------
        bool
            True if the neighbor list is updated.
        """

        if self.nupdates == 0:
            self.build(pbc, cell, coordinates, grids)
            return True

        if ((self.pbc != pbc).any() or (self.cell != cell).any() or (
                (self.coordinates
                 - coordinates)**2).sum(1).max() > self.skin**2):
            self.build(pbc, cell, coordinates, grids)
            return True

        return False


    def build(self, pbc, cell, coordinates, grids):
        """Build the list.

        Coordinates are taken to be scaled or not according
        to self.use_scaled_positions.
        """
        self.pbc = pbc = np.array(pbc, copy=True)
        self.cell = cell = Cell(cell)
        self.coordinates = coordinates = np.array(coordinates, copy=True)
        self.grids = grids = np.array(grids, copy=True)

        if len(self.cutoffs) != len(coordinates):
            raise ValueError('Wrong number of cutoff radii: {} != {}'
                             .format(len(self.cutoffs), len(coordinates)))

        if len(self.cutoffs) > 0:
            rcmax = self.cutoffs.max()
        else:
            rcmax = 0.0

        if self.use_scaled_positions:
            positions0 = cell.cartesian_positions(coordinates)
            grids0 = cell.cartesian_positions(grids)
        else:
            positions0 = coordinates
            grids0 = grids

        rcell, op = minkowski_reduce(cell, pbc)
        positions = wrap_positions(positions0, rcell, pbc=pbc, eps=0) # make some atom outside the cell goes into the cell
        grid_positions = wrap_positions(grids0, rcell, pbc=pbc, eps=0)

        natoms = len(positions)
        ngrids = len(grids)

        self.nneighbors = 0
        self.npbcneighbors = 0
        self.neighbors = [np.empty(0, int) for _ in range(ngrids)]
        self.displacements = [np.empty((0, 3), int) for _ in range(ngrids)]
        self.nupdates += 1
        if ngrids == 0 or natoms == 0:
            return

        N = []
        ircell = np.linalg.pinv(rcell)
        for i in range(3):
            if self.pbc[i]:
                v = ircell[:, i]
                h = 1 / np.linalg.norm(v)
                n = int(2 * rcmax / h) + 1
            else:
                n = 0
            N.append(n)

        tree = cKDTree(positions, copy_data=True)
        offsets = cell.scaled_positions(positions - positions0)
        offsets = offsets.round().astype(int)

        grid_offsets = cell.scaled_positions(grid_positions - grids0)
        grid_offsets = grid_offsets.round().astype(int)

        for n1, n2, n3 in itertools.product(range(-N[0], N[0] + 1),
                                            range(-N[1], N[1] + 1),
                                            range(-N[2], N[2] + 1)):
            # if n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0):
            #     continue

            displacement = (n1, n2, n3) @ rcell
            for g in range(ngrids):

                indices = tree.query_ball_point(grid_positions[g] - displacement,
                                                r=rcmax)
                if not len(indices):
                    continue

                indices = np.array(indices)
                delta = positions[indices] + displacement - grid_positions[g]
                cutoffs = self.cutoffs[indices]
                
                i = indices[np.linalg.norm(delta, axis=1) < cutoffs]

                self.nneighbors += len(i)
                self.neighbors[g] = np.concatenate((self.neighbors[g], i))

                disp = (n1, n2, n3) @ op + offsets[i] - grid_offsets[g]
                self.npbcneighbors += disp.any(1).sum()
                self.displacements[g] = np.concatenate((self.displacements[g],
                                                        disp))
                # rel_pos = positions[i] - self.grids[g] + disp @ cell.array
                # print(np.linalg.norm(rel_pos, axis=-1), np.linalg.norm(delta, axis=1))

        if self.sorted:
            for g in range(ngrids):
                # sort first by neighbors and then offsets
                keys = (
                    self.displacements[g][:, 2],
                    self.displacements[g][:, 1],
                    self.displacements[g][:, 0],
                    self.neighbors[g],
                )
                mask = np.lexsort(keys)
                self.neighbors[g] = self.neighbors[g][mask]
                self.displacements[g] = self.displacements[g][mask]


    def get_neighbors(self, g):
        """Return neighbors of atom number a.

        A list of indices and offsets to neighboring atoms is
        returned.  The positions of the neighbor atoms can be
        calculated like this:

        >>> from ase.build import bulk
        >>> from ase.neighborlist import NewPrimitiveNeighborList

        >>> nl = NewPrimitiveNeighborList([2.3, 1.7])
        >>> atoms = bulk('Cu', 'fcc', a=3.6)
        >>> nl.update(atoms.pbc, atoms.get_cell(), atoms.positions)
        True
        >>> indices, offsets = nl.get_neighbors(0)
        >>> for i, offset in zip(indices, offsets):
        ...     print(
        ...           atoms.positions[i] + offset @ atoms.get_cell()
        ...     )  # doctest: +ELLIPSIS
        [3.6 ... 0. ]

        Notice that if get_neighbors(a) gives atom b as a neighbor,
        then get_neighbors(b) will not return a as a neighbor - unless
        bothways=True was used."""

        return self.neighbors[g], self.displacements[g]