from ase.io import read
from ase.neighborlist import neighbor_list
import numpy as np
from numba import jit
from scipy.special import spherical_in, sph_harm
import argparse
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--xyz', required=True, help='path to structures dataset')
parser.add_argument('--np', type=int, help='number of processes (default equals number of CPUs)')
parser.add_argument('--lmax', type=int, default=6, help='maximum l')
parser.add_argument('--alpha', type=int, default=.5, help='gaussian width')
parser.add_argument('--rc', type=float, default=2.5, help='radial cut-off in angstrom')
parser.add_argument('--nocenter', nargs="+", type=int, help='atomic numbers to skip as centers (e.g. --nocenter 1 2)')
opt = parser.parse_args()


structures = read(opt.xyz, format='extxyz', index=':')

l_max = opt.lmax+1
alpha = opt.alpha
repeated_l = np.repeat(np.arange(l_max), np.arange(1,l_max*2,2)).reshape(-1,1)
repeated_m = [[m for m in np.arange(-l, l+1)] for l in np.arange(l_max)]
repeated_m = np.array([item for sublist in repeated_m for item in sublist]).reshape(-1,1)


@jit(nopython=True)
def azimuthal_polar(r_vector):
    azimuth_vector = []
    polar_vector = []
    n = len(r_vector)
    for i in range(n):
        x,y,z = r_vector[i]
        azimuth = np.arctan2(y, x)
        if azimuth < 0:
            azimuth += 2*np.pi
        polar = np.arccos(z / np.linalg.norm(r_vector[i]))
        azimuth_vector.append(azimuth)
        polar_vector.append(polar)
    return (azimuth_vector,polar_vector)


def environment(neighbors, atom_i):
    n_i, n_j, n_D, n_d = neighbors
    selected = (n_i == atom_i)
    theta, phi = azimuthal_polar(n_D[selected])
    return (n_d[selected], theta, phi)


@jit(nopython=True)
def I_sum(part1, part2, part3, part4):
    result = 0
    n_i = part1.shape[1]
    n_j = part1.shape[0]
    for l in range(l_max):
        for m_i in range(-l, l+1):
            l_m_i = np.where((repeated_l == l) & (repeated_m == m_i))[0][0]
            for m_j in range(-l, l+1):
                l_m_j = np.where((repeated_l == l) & (repeated_m == m_j))[0][0]
                
                pair_result = np.complex(0,0)
                for r_i in range(n_i):
                    for r_j in range(n_j):
                        pair_result += part1[r_j,r_i] * part2[l, r_j, r_i] * part3[l_m_i,r_i] * part4[l_m_j,r_j]
                result += 8*np.power(np.pi,2)/(2*l+1)*((np.conj(pair_result) * pair_result).real)
    return result


prefactor = np.sqrt(2*np.power(np.pi,5)/np.power(alpha,3))
def environment_kernel(environment_Ai, environment_Bj):
    r_i, theta_i, phi_i = environment_Ai
    r_j, theta_j, phi_j = environment_Bj

    n_i = len(r_i)
    n_j = len(r_j)
    r_i = np.broadcast_to(r_i, (n_j, n_i))
    r_j = r_j.reshape(-1,1)
    
    part1 = prefactor*np.exp(-alpha*(np.power(r_i,2)+np.power(r_j,2))/2.)
    
    spherical_in_inside = 2*alpha*r_i*r_j
    l = np.arange(l_max).reshape(-1,1,1)
    part2 = spherical_in(l,spherical_in_inside)
    
    part3 = sph_harm(repeated_m, repeated_l, theta_i, phi_i)
    part4 = np.conj(sph_harm(repeated_m, repeated_l, theta_j, phi_j))
    
    return I_sum(part1, part2, part3, part4)


def covariance_matrix(A,B):
    neighbors_A = neighbor_list('ijDd', A, opt.rc)
    neighbors_B = neighbor_list('ijDd', B, opt.rc)

    i_list = set(neighbors_A[0])
    if opt.nocenter is not None:
        i_list = [i for i in i_list if A[i].number not in opt.nocenter]
    j_list = set(neighbors_B[0])
    if opt.nocenter is not None:
        j_list = [j for j in j_list if B[j].number not in opt.nocenter]
    size = (len(i_list),len(j_list))

    covariance_matrix = np.zeros(size)
    for ei, i in enumerate(i_list):
        environment_i = environment(neighbors_A, i)
        
        for ej, j in enumerate(j_list):
            environment_j = environment(neighbors_B, j)
            covariance_matrix[ei,ej] = environment_kernel(environment_i, environment_j)

    return covariance_matrix


def average_kernel(A, B):
    C = covariance_matrix(A,B)
    return C.mean()


def calculate(combination_i):
    i, j = combination[combination_i]
    return average_kernel(structures[i], structures[j])


combination = []
for i in range(len(structures)):
    for j in range(len(structures)):
        if j < i:
            combination.append((i,j))

with Pool(opt.np) as p:
    kernels = list(tqdm(p.imap(calculate, range(len(combination))), total=len(combination)))


np.save('k', np.array(kernels))
