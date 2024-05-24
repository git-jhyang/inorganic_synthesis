from .utils import ActiveElements, NEAR_ZERO, composit_parser
import numpy as np
import json, os, gzip, pickle

ATOM_EXCEPTION_WARNING = 'Warning: element [{}] is not included in feature type "{}"\n'
FEAT_EXCEPTION_WARNING = 'Warning: feature type [{}] is not supported. '

elmd = {}
for k in ['cgcnn','elemnet','magpie','magpie_sc','mat2vec','matscholar','megnet16','oliynyk','oliynyk_sc']:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elmd', f'{k}.json')) as f:
        elmd_data = json.load(f)
    elmd[k] = elmd_data

with gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/screened_unique_precursor.pkl.gz'),'rb') as f:
    _screened_precursor = pickle.load(f)
num_labels = len([k for k in _screened_precursor.keys() if isinstance(k, int)])

def composition_to_feature(composit_dict, 
                           feature_type='composit', 
                           dtype=np.float32, 
                           by_fraction=True,
                           norm=True,
                           *args, **kwargs):
    '''
    composit_dict = {
        element_0 (str): fraction_0 (float),
        element_1 (str): fraction_1 (float),
        ...
    }
    '''
    global ATOM_EXCEPTION_WARNING, FEAT_EXCEPTION_WARNING
    div = 1
    if norm and len(composit_dict) != 0:
        by_fraction = True
        div = 1.0 / np.sum(list(composit_dict.values()))

    if feature_type in elmd.keys():
        vec = [np.zeros_like(elmd[feature_type]['Li'])]
        for ele, frac in composit_dict.items():
            if ele in elmd[feature_type].keys():
                vec.append(np.array(elmd[feature_type][ele]) * (frac if by_fraction else 1) * div)
            elif ATOM_EXCEPTION_WARNING is not None:
                print(ATOM_EXCEPTION_WARNING.format(ele, feature_type))
        ATOM_EXCEPTION_WARNING = None
        return np.sum(vec, 0).astype(dtype).reshape(1,-1)
    else:
        if not feature_type.startswith('compo') and FEAT_EXCEPTION_WARNING is not None:
            print(FEAT_EXCEPTION_WARNING.format(feature_type))
            print('Possible feature types:', ['composit'] + list(elmd.keys()))
            print('Feature type is set to composit')
            FEAT_EXCEPTION_WARNING = None
        return active_composit_feature(composit_dict, dtype, by_fraction,).reshape(1,-1) * div

def active_composit_feature(composit_dict, dtype=float, by_fraction=True, *args, **kwargs):
    feat_vec = np.zeros(len(ActiveElements), dtype=dtype)
    for ele, frac in composit_dict.items():
        if frac <= NEAR_ZERO: continue
        feat_vec[ActiveElements.index(ele)] = frac if by_fraction else 1
    return feat_vec

# def metal_composit_feature(composit_dict, dtype=float, by_fraction=True, *args, **kwargs):
#     feat_vec = np.zeros(len(MetalElements) + 1, dtype=dtype)
#     for ele, frac in composit_dict.items():
#         feat_vec[MetalElements.index(ele) + 1] = frac if by_fraction else 1
#     if feat_vec.sum() == 0:
#         feat_vec[0] = 1
#     return feat_vec

# def ligand_composit_feature(composit_dict, dtype=float, by_fraction=True, *args, **kwargs):
#     feat_vec = np.zeros(len(LigandElements) + 1, dtype=dtype)
#     for ele, frac in composit_dict.items():
#         if ele in MetalElements:
#             feat_vec[0] += frac if by_fraction else 1
#         else:
#             feat_vec[LigandElements.index(ele) + 1] = frac if by_fraction else 1
#     return feat_vec

# def feature_to_composit(feat_vec, tol=1e-5):
#     n_feat = feat_vec.shape[-1]
#     feat_vec = feat_vec.reshape(-1, n_feat)
#     if n_feat not in [97, 87, 12]:
#         raise TypeError(f'feature type is not supported', composit_fnc)
#     if n_feat == 97: # active_composit
#         ref = np.array(ActiveElements)
#     elif n_feat == 87: # metal_composit
#         ref = np.array(['None'] + MetalElements)
#     elif n_feat == 12: # ligand_composit
#         ref = np.array(['Metal'] + LigandElements)
#     out = []
#     for vec, mask in zip(feat_vec, feat_vec > tol):
#         out.append(
#             {e:f for e,f in zip(ref[mask], vec[mask])}
#         )
#     return out

# def parse_feature(feat_vec, csim_cut = 0.5, sser_cut = 1.0, to_string=False):
#     out = []
#     if feat_vec.shape[-1] == 12:
#         ref = ligand_vector
#         chrs = np.array(['-'.join(k) for k in ligand_list[:-1]] + ['Unknown'])
#     elif feat_vec.shape[-1] == 97:
#         ref = precursor_vector
#         chrs = np.array(['-'.join(k) for k in precursor_list[:-1]] + ['Unknown'])
#     else:
#         raise ValueError('Not supported feature type')

#     for idx, sser, csim in zip(*find_nearest(feat_vec, ref)):
#         if (sser > sser_cut) or (csim < csim_cut):
#             out.append(-1)
#         else:
#             out.append(int(idx))
#     if to_string:
#         return chrs[out]
#     else:
#         return out

# def check_blacklist(comp):
#     key = tuple(k for k in sorted(comp.keys(), key=lambda x: Element(x).number))
#     if key in blacklist:
#         return True
#     else:
#         return False

def get_precursor_info(inp):
    if isinstance(inp, int):
        return _screened_precursor[inp]
    elif isinstance(inp, dict):
        pstr = composit_parser(inp)
    elif isinstance(inp, str) and inp in _screened_precursor.keys():
        pstr = inp
    else:
        raise ValueError('Invalid input', inp)
    return _screened_precursor[pstr]
