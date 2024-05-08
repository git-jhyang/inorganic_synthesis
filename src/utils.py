from pymatgen.core import Element
import numpy as np
import torch

NEAR_ZERO = 1e-5

NonMetals = 'H C N O F P S Cl Se Br I'.split()
AlkaliMetals = 'Li Na K Rb Cs Fr'.split()
AlkaliEarthMetals = 'Be Mg Ca Sr Ba Ra'.split()
TransitionMetals = 'Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg'.split()
PostTransitionMetals = 'Al Ga In Sn Tl Pb Bi Po'.split()
Metalloids = 'B Si Ge As Sb Te At'.split()
Lanthanoids = 'La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu'.split()
Actinoids = 'Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr'.split()
Halogens = 'He Ne Ar Kr Xe Rn'.split()
Unknown = 'Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split()

MetalElements = sorted(AlkaliMetals + AlkaliEarthMetals + TransitionMetals + PostTransitionMetals + Metalloids + Lanthanoids + Actinoids, key=lambda x: Element(x).number)
LigandElements = sorted(NonMetals, key=lambda x: Element(x).number)
ActiveElements = sorted(MetalElements + NonMetals, key=lambda x: Element(x).number)
AllElements = sorted(ActiveElements + Halogens + Unknown, key=lambda x: Element(x).number)

def to_numpy(vector):
    if isinstance(vector, torch.Tensor):
        return vector.cpu().numpy()
    else:
        return np.array(vector)

def squared_error(mat1, mat2, average=True):
    x = to_numpy(mat1)
    y = to_numpy(mat2)
    sq_err = np.sum(np.square(x - y), -1)
    if average:
        return sq_err.mean()
    else:
        return sq_err

def cosin_similarity(mat1, mat2, average=True):
    x = to_numpy(mat1)
    y = to_numpy(mat2)
    l = np.sqrt(np.sum(np.square(x), -1, keepdims=True)) * np.sqrt(np.sum(np.square(y), -1, keepdims=True))
    m = (l == 0).squeeze()
    cos_sim = np.zeros(m.shape, dtype=float)
    cos_sim[~m] = np.sum((x * y)[~m] / l[~m], -1)
    cos_sim[m] = 1 - np.sqrt(np.square(x - y)[m].sum(-1))
    if average:
        return cos_sim.mean()
    else:
        return cos_sim

# def find_nearest(vectors, reference):
#     out = []
#     for vec in vectors.reshape(-1, reference.shape[-1]):
#         sser = squared_error(vec, reference, average=False)
#         csim = cosin_similarity(vec, reference, average=False)
#         i = np.argmin(sser - csim)
#         out.append([i, sser[i], csim[i]])
#     return np.array(out).T

def linear_kld_annealing(epochs, start=0, stop=1, period=500, ratio=0.5):
    '''
    Code from paper 'Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing'
    arXiv: https://arxiv.org/abs/1903.10145
    github: https://github.com/haofuml/cyclical_annealing

    Scheduling KLD for better training of VAE.
    '''
    beta = np.ones(epochs)
    step = (stop - start) / (period * np.clip(ratio, 0, 1))
    for i in range(int(period)):
        beta[i::int(period)] = start + step * i
    return np.clip(beta, start, stop)

def exponential_kld_annealing(epochs, start=-35, stop=0, period=500, ratio=0.5):
    '''
    Code from paper 'Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing'
    arXiv: https://arxiv.org/abs/1903.10145
    github: https://github.com/haofuml/cyclical_annealing

    Scheduling KLD for better training of VAE.
    '''
    beta = np.ones(epochs)
    step = (stop - start) / (period * ratio)
    for i in range(int(period)):
        beta[i::int(period)] = np.power(10.0, start + step * i)
    return np.clip(beta, np.power(10.0, start), np.power(10.0, stop))


#class StratifiedRandomSampler:
#    def __init__(self, dataset, class_attr)