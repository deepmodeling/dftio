from dftio.io.gaussian.gaussian_parser import GaussianParser
from dftio.io.gaussian.gaussian_tools import matrix_to_image, get_convention, check_transform


convention = get_convention(filename='gau.log')
a_gau_parser = GaussianParser()
a_ham_matrix, an_overlap_matrix, a_density_matrix = a_gau_parser.get_blocks(logname='gau.log', hamiltonian=True, overlap=True, density=True, convention=convention)
matrix_to_image(a_ham_matrix[:22, :22], filename='ham.png')
matrix_to_image(an_overlap_matrix[:22, :22], filename='overlap.png')
matrix_to_image(a_density_matrix[:22, :22], filename='density.png')

check_transform(filepath='gau.log')