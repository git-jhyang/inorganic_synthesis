from pymatgen.core import Element
import numpy as np
import torch, os, json

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

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/ligand_freq.json'),'r') as f:
    ligand_frequency = json.load(f)

ligand_label = {l:i for i,l in enumerate(ligand_frequency.keys())}
ligand_index = {i:l for i,l in enumerate(ligand_frequency.keys())}
ligand_index.update({-1:'Unk'})

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
    x = x / np.sqrt(np.sum(np.square(x), -1, keepdims=True))
    y = y / np.sqrt(np.sum(np.square(y), -1, keepdims=True))
    cos_sim = np.sum(x * y, -1)
    if average:
        return cos_sim.mean()
    else:
        return cos_sim

def find_nearest(vectors, reference, alpha=1):
    out = []
    for vec in vectors:
        sser = squared_error(vec.reshape(1,-1), reference, average=False)
        csim = cosin_similarity(vec.reshape(1,-1), reference, average=False)
        i = np.argmin(sser - alpha * csim)