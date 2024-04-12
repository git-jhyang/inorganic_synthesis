from .utils import ActiveElements, MetalElements, LigandElements, NEAR_ZERO
import numpy as np
import json, os

PRINT_EXCEPTION_WARNING = 'Warning: element "{}" is not included in feature type "{}"\n'

elmd = {}
exception = {}
for k in ['cgcnn','elemnet','magpie','magpie_sc','mat2vec','matscholar','megnet16','oliynyk','oliynyk_sc']:
    elmd_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elmd', f'{k}.json')))
    elmd[k] = elmd_data
    exception[k] = [k for k in ActiveElements if k not in elmd_data.keys()]

composit_fnc = ['active_composit','metal_composit','ligand_composit']

def composition_to_feature(composit_dict, 
                           feature_type='active_composit', 
                           dtype=np.float32, 
                           by_fraction=True):
    '''
    composit_dict = {
        element_0 (str): fraction_0 (float),
        element_1 (str): fraction_1 (float),
        ...
    }
    '''
    if feature_type in composit_fnc:
        if feature_type == 'active_composit':
            return active_composit_feature(composit_dict, dtype, by_fraction)
        elif feature_type == 'metal_composit':
            return metal_composit_feature(composit_dict, dtype, by_fraction)
        elif feature_type == 'ligand_composit':
            return ligand_composit_feature(composit_dict, dtype, by_fraction)
    elif feature_type in elmd.keys():
        vec = []
        norm = 1.0 / np.sum(list(composit_dict.values()))
        for ele, frac in composit_dict.items():
            if ele in exception[feature_type]:
                if PRINT_EXCEPTION_WARNING is not None:
                    print(PRINT_EXCEPTION_WARNING.format(ele, feature_type))
                continue
            vec.append(np.array(elmd[feature_type][ele]) * (frac if by_fraction else 1))
        PRINT_EXCEPTION_WARNING = None
        return (np.sum(vec, 0) * norm).astype(dtype)
    else:
        raise TypeError(f'feature type [{feature_type}] is not supported', composit_fnc + list(elmd.keys()))

def active_composit_feature(composit_dict, dtype, by_fraction=True, **kwargs):
    feat_vec = np.zeros(len(ActiveElements), dtype=dtype)
    for ele, frac in composit_dict.items():
        if frac < NEAR_ZERO: continue
        feat_vec[ActiveElements.index(ele)] = frac if by_fraction else 1
    feat_vec /= feat_vec.sum()
    return feat_vec

def metal_composit_feature(composit_dict, dtype, by_fraction=True, **kwargs):
    feat_vec = np.zeros(len(MetalElements) + 1, dtype=dtype)
    for ele, frac in composit_dict.items():
        feat_vec[MetalElements.index(ele) + 1] = frac if by_fraction else 1
    if feat_vec.sum() == 0:
        feat_vec[0] = 1
    return feat_vec

def ligand_composit_feature(composit_dict, dtype, by_fraction=True, **kwargs):
    feat_vec = np.zeros(len(LigandElements) + 1, dtype=dtype)
    feat_vec[0] = 1
    for ele, frac in composit_dict.items():
        if ele in MetalElements:
            feat_vec[0] = 0
            continue
        feat_vec[LigandElements.index(ele) + 1] = frac if by_fraction else 1
    return feat_vec
