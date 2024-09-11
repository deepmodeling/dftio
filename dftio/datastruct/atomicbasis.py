from dftio.dep._xitorch.interpolate import Interp1D
import torch

import scipy.sparse.csgraph as csgraph
import h5py
from e3nn.o3 import SphericalHarmonics, Irreps

def find_target_line(f, target):
    line = f.readline()
    while line:
        if target in line:
            return line
        line = f.readline()
    return None

def read_orbfile(path):
    au2angstrum = 0.529177249
    Lname = ["S", "P", "D", "F", "G"]

    with open(path) as f:
        element = find_target_line(f, "Element").split()[-1]
        radius = float(find_target_line(f, "Radius Cutoff(a.u.)").split()[-1]) * au2angstrum
        Lmax = int(find_target_line(f, "Lmax").split()[-1])
        n = []
        for i in range(Lmax+1):
            n.append(int(find_target_line(f, "Number of {0}orbital".format(Lname[i])).split()[-1]))
        dr = float(find_target_line(f, "dr").split()[-1]) * au2angstrum
        r = torch.arange(0, radius+dr, dr)
        orbdict = {"r": r}
        # loop over all orbitals
        for iorb in range(sum(n)):
            find_target_line(f, "                Type                   L                   N")
            T, L, N = f.readline().split()[:3]
            fr = []
            while True:
                line = f.readline()
                if "-0.00000000000000e+00  \n" == line or "0.00000000000000e+00  \n" == line:
                    fr.append(0.0)
                    break
                else:
                    fr += map(float, line.split())
            orbdict[T+L+N] = torch.tensor(fr)
        
    return element, dr, radius, orbdict


def read_rescu_file(path):
    au2angstrum = 0.529177249
    orbdict = {}
    with h5py.File(path) as f:
        Nradial = len(f["data"]["OrbitalSet"]["Parameter"])
        for orbN in range(Nradial):
            rr = torch.tensor(f[f["data"]["OrbitalSet"]["rrData"][:][orbN][0]][:].reshape(-1) * au2angstrum)
            fr = f[f["data"]["OrbitalSet"]["frData"][:][orbN][0]][:].reshape(-1)
            tp = "".join([chr(i[0]) for i in f[f["data"]["OrbitalSet"]["Parameter"][orbN][0]]["type"][:]])
            T = int(tp.split("-")[0][-1]) - 1
            L = int(f[f["data"]["OrbitalSet"]["Parameter"][orbN][0]]["L"][:][0,0])
            N = int(f[f["data"]["OrbitalSet"]["Parameter"][orbN][0]]["N"][:][0,0])
            element = "".join([chr(i[0]) for i in f["data"]["atom"]["symbol"]])

            dr = (rr[1:] - rr[:-1]).mean()
            radius = rr.max()

            orbdict["0"+str(L)+str(N)+str(T)] = torch.tensor(fr)
    
    new_orbdict = {}
    # sort the orbdict
    orb_keys = list(orbdict.keys())
    orbls = [int(i[1]) for i in orb_keys]
    orbindex = sorted(range(len(orbls)), key=lambda k: orbls[k])
    count_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for i in orbindex:
        org_key = orb_keys[i]
        T_new = 0
        L_new = org_key[1]
        N_new = count_dict[int(L_new)]
        new_orbdict[str(T_new)+str(L_new)+str(N_new)] = orbdict[org_key]
        count_dict[int(L_new)] += 1
    
    new_orbdict["r"] = rr
    return element, dr, radius, new_orbdict

class AtomicBasis(object):
    def __init__(self, element, basis, rcut, radial_type="spline", dtype=torch.float32):
        self.element = element
        self.basis = basis
        self.rcut = rcut
        self.radial_type = radial_type
        self.dtype = dtype

        # build the mapping
        irreps = []
        self.mapping = []
        cc = 0 # count of the orbital
        symbol2l = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
        for i in range(0, len(basis), 2):
            bs = self.basis[i:i+2]
            n = int(bs[0])
            l = symbol2l[bs[1]]
            irreps.append((n, (l, (-1)**l)))
            
            for c in range(cc, cc+n):
                self.mapping += [c] * (2*l + 1)
            cc += n
        
        self.irreps = Irreps(irreps)
        self.mapping = torch.tensor(self.mapping)

    def __str__(self):
        return f"{self.element} {self.basis} {self.rcut} {self.radial_type}"

    def __repr__(self):
        return self.__str__()

    def __call__(self, r):
        if len(r.shape) == 1:
            assert r.shape[0] == 3
            r = r.unsqueeze(0)
        else:
            assert len(r.shape) == 2 and r.shape[1] == 3
        return self.radial(r.norm(dim=-1)).T[:,self.mapping] * self.spherical(r)
    
    @classmethod
    def from_orbfile(cls, filename):
        if filename.endswith(".mat"):
            element, dr, rcut, orbdict = read_rescu_file(filename)
        elif filename.endswith(".orb"):
            element, dr, rcut, orbdict = read_orbfile(filename)

        l2symbol = ["s", "p", "d", "f", "g"]
        symbol2l = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
        symbol_count = {}
        for k in orbdict.keys():
            if k != "r":
                symbol = l2symbol[int(k[1])]
                if symbol not in symbol_count:
                    symbol_count[symbol] = 0
                symbol_count[symbol] += 1
        count_list = [0] * 5
        for s in symbol_count:
            count_list[symbol2l[s]] = symbol_count[s]
        basis = "".join([str(c)+l2symbol[i] for i, c in enumerate(count_list) if c != 0])
        ab = cls(element=element, basis=basis, rcut=rcut, radial_type="spline", dtype=torch.float64)
        radial = []
        irreps = []

        for ik, k in enumerate(sorted(orbdict.keys())):
            if k != "r":
                radial.append(orbdict[k])
                irreps += [(1, (int(k[1]), (-1)**int(k[1])))]

        ab.radial = Interp1D(orbdict["r"].double(), torch.stack(radial, dim=0).double(), dim=0)
        ab.irreps = Irreps(irreps)
        ab.spherical = SphericalHarmonics(ab.irreps, True, "integral")

        return ab