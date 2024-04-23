from .utils import ActiveElements, MetalElements, LigandElements, NEAR_ZERO, find_nearest
import numpy as np
import json, os

PRINT_EXCEPTION_WARNING = 'Warning: element "{}" is not included in feature type "{}"\n'

elmd = {}
for k in ['cgcnn','elemnet','magpie','magpie_sc','mat2vec','matscholar','megnet16','oliynyk','oliynyk_sc']:
    elmd_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elmd', f'{k}.json')))
    elmd[k] = elmd_data

composit_fnc = ['active_composit','metal_composit','ligand_composit']

def get_ligand_info():
    global ligand_label, ligand_index, ligand_vector
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/ligand_freq.json'),'r') as f:
        ligand_frequency = json.load(f)

    ligand_label = {l:i for i,l in enumerate(ligand_frequency.keys())}
    ligand_index = {i:l for i,l in enumerate(ligand_frequency.keys())}
    ligand_index.update({-1:'Unk'})
    ligand_vector = np.vstack([ligand_composit_feature({e:1 for e in l.replace('Metal','Li').split('-')}, float) for l in ligand_label.keys()])

def composition_to_feature(composit_dict, 
                           feature_type='active_composit', 
                           dtype=np.float32, 
                           by_fraction=True,
                           norm=True,
                           ):
    '''
    composit_dict = {
        element_0 (str): fraction_0 (float),
        element_1 (str): fraction_1 (float),
        ...
    }
    '''
    global PRINT_EXCEPTION_WARNING
    if feature_type == 'active_composit':
        return active_composit_feature(composit_dict, dtype, by_fraction, norm).reshape(1,-1)
    elif feature_type == 'metal_composit':
        return metal_composit_feature(composit_dict, dtype, by_fraction, norm).reshape(1,-1)
    elif feature_type == 'ligand_composit':
        return ligand_composit_feature(composit_dict, dtype, by_fraction, norm).reshape(1,-1)
    elif feature_type in elmd.keys():
        vec = [np.zeros_like(elmd[feature_type]['Li'])]
        div = 1
        if norm and len(composit_dict) != 0:
            div = 1.0 / np.sum(list(composit_dict.values()))
        for ele, frac in composit_dict.items():
            if ele in elmd[feature_type].keys():
                vec.append(np.array(elmd[feature_type][ele]) * (frac if by_fraction else 1))
            elif PRINT_EXCEPTION_WARNING is not None:
                print(PRINT_EXCEPTION_WARNING.format(ele, feature_type))
        PRINT_EXCEPTION_WARNING = None
        return (np.sum(vec, 0) * div).astype(dtype).reshape(1,-1)
    else:
        raise TypeError(f'feature type [{feature_type}] is not supported', composit_fnc + list(elmd.keys()))

def active_composit_feature(composit_dict, dtype, by_fraction=True, norm=True, *args, **kwargs):
    feat_vec = np.zeros(len(ActiveElements), dtype=dtype)
    for ele, frac in composit_dict.items():
        if frac < NEAR_ZERO: continue
        feat_vec[ActiveElements.index(ele)] = frac if by_fraction else 1
    if norm and feat_vec.sum() > 0:
        feat_vec /= feat_vec.sum()
    return feat_vec

def metal_composit_feature(composit_dict, dtype, by_fraction=True, *args, **kwargs):
    feat_vec = np.zeros(len(MetalElements) + 1, dtype=dtype)
    for ele, frac in composit_dict.items():
        feat_vec[MetalElements.index(ele) + 1] = frac if by_fraction else 1
    if feat_vec.sum() == 0:
        feat_vec[0] = 1
    return feat_vec

def ligand_composit_feature(composit_dict, dtype, by_fraction=True, *args, **kwargs):
    feat_vec = np.zeros(len(LigandElements) + 1, dtype=dtype)
    for ele, frac in composit_dict.items():
        if ele in MetalElements:
            feat_vec[0] += frac if by_fraction else 1
        else:
            feat_vec[LigandElements.index(ele) + 1] = frac if by_fraction else 1
    return feat_vec

def feature_to_composit(feat_vec, tol=0.01):
    n_feat = feat_vec.shape[-1]
    if n_feat not in [97, 87, 12]:
        raise TypeError(f'feature type is not supported', composit_fnc)
    if n_feat == 97: # active_composit
        ref = np.array(ActiveElements)
    elif n_feat == 87: # metal_composit
        ref = np.array(['None'] + MetalElements)
    elif n_feat == 12: # ligand_composit
        ref = np.array(['Metal'] + LigandElements)
    out = []
    for vec, mask in zip(feat_vec, feat_vec > tol):
        out.append(
            {e:f for e,f in zip(ref[mask], vec[mask])}
        )
    return out

def feature_to_ligand_index(feat_vec, csim_cut = 0.8, sser_cut = 0.3):
    out = []
    for idx, sser, csim in zip(*find_nearest(feat_vec, ligand_vector)):
        if (sser > sser_cut) or (csim < csim_cut):
            out.append(-1)
        else:
            out.append(idx)
    return out

get_ligand_info()