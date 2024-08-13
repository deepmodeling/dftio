import torch

# here we should implementing several grid making methods

def make_simple_grid(cell, nx, ny, nz):
    """
    Make a simple grid with nx, ny, nz points in each direction.

    Parameters
    ----------
    cell : torch.Tensor
        The cell.
    nx : int
        The number of points in the x direction.
    ny : int
        The number of points in the y direction.
    nz : int
        The number of points in the z direction.

    Returns
    -------
    torch.Tensor
        The grid.
    """
    x = torch.linspace(0.0, cell[0].norm(), nx)
    y = torch.linspace(0.0, cell[1].norm(), ny)
    z = torch.linspace(0.0, cell[2].norm(), nz)

    grid = torch.meshgrid(x, y, z)
    grid = torch.stack(grid, dim=-1)
    grid = grid.reshape(-1, 3)

    return (nx, ny, nz), grid

def make_simple_grid2(x0, y0, z0, x1, y1, z1, nx, ny, nz):
    """
    Make a simple grid with nx, ny, nz points in each direction.

    Parameters
    ----------
    cell : torch.Tensor
        The cell.
    nx : int
        The number of points in the x direction.
    ny : int
        The number of points in the y direction.
    nz : int
        The number of points in the z direction.

    Returns
    -------
    torch.Tensor
        The grid.
    """

    x = torch.linspace(x0, x1, nx)
    y = torch.linspace(y0, y1, ny)
    z = torch.linspace(z0, z1, nz)

    grid = torch.meshgrid(x, y, z)
    grid = torch.stack(grid, dim=-1)
    grid = grid.reshape(-1, 3)

    return (nx, ny, nz), grid

def make_uniform_grid(cell, dr):
    """
    Make a uniform grid with dr spacing.

    Parameters
    ----------
    cell : torch.Tensor
        The cell.
    dr : float
        The spacing.

    Returns
    -------
    torch.Tensor
        The grid.
    """

    assert dr > 0.0 and dr < cell.norm(dim=1).min()

    nx = int(cell[0].norm() / dr)
    ny = int(cell[1].norm() / dr)
    nz = int(cell[2].norm() / dr)

    return make_simple_grid(cell, nx, ny, nz)
