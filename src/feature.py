from .utils import ActiveElements, MetalElements, LigandElements, NEAR_ZERO, find_nearest, Element
import numpy as np
import json, os, gzip, pickle

PRINT_EXCEPTION_WARNING = 'Warning: element "{}" is not included in feature type "{}"\n'

elmd = {}
for k in ['cgcnn','elemnet','magpie','magpie_sc','mat2vec','matscholar','megnet16','oliynyk','oliynyk_sc']:
    elmd_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elmd', f'{k}.json')))
    elmd[k] = elmd_data

composit_fnc = ['active_composit','metal_composit','ligand_composit']

def init_info(min_count=0):
    global blacklist, ligand_count, ligand_vector, ligand_list, precursor_count, precursor_vector, precursor_list
    with gzip.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/unique_ligand.pkl.gz'),'rb') as f:
        unique_ligand = pickle.load(f)

    ligand_count = {}
    precursor_count = {}
    for ligand, _ligand_data in unique_ligand.items():
        for metal, _metal_data in _ligand_data['metals'].items():
            precursor = tuple(sorted(metal + ligand, key=lambda x: Element(x).number))
            c = _metal_data['count']
            if len(metal) == 0:
                if precursor not in ligand_count.keys():
                    ligand_count[ligand] = 0
                    precursor_count[ligand] = 0
                ligand_count[ligand] += c
                precursor_count[ligand] += c
            else:
                if precursor not in precursor_count.keys():
                    precursor_count[precursor] = 0
                if (('Metal',) + ligand) not in ligand_count.keys():
                    ligand_count[(('Metal',) + ligand)] = 0
                ligand_count[(('Metal',) + ligand)] += c
                precursor_count[precursor] += c
    blacklist = [k for k,v in ligand_count.items() if v < min_count]
    blacklist += [k for k,v in precursor_count.items() if v < min_count]
    ligand_count = {k:v for k,v in sorted(ligand_count.items(), key=lambda x: x[1], reverse=True) if k not in blacklist}
    ligand_count.update({():0})
    ligand_list = list(ligand_count.keys())
    precursor_count = {k:v for k,v in sorted(precursor_count.items(), key=lambda x: x[1], reverse=True) if k not in blacklist}
    precursor_count.update({():0})
    precursor_list = list(precursor_count.keys())
    ligand_vector = np.vstack([ligand_composit_feature({e if e != 'Metal' else 'Li':1 for e in k}) for k in ligand_list])
    precursor_vector = np.vstack([active_composit_feature({e:1 for e in k}) for k in precursor_list])
#    ligand_label = {l:i for i,l in enumerate(ligand_frequency.keys())}
#    ligand_index = {i:l for i,l in enumerate(ligand_frequency.keys())}
#    ligand_index.update({-1:'Unk'})
#    ligand_vector = np.vstack([ligand_composit_feature({e:1 for e in l.replace('Metal','Li').split('-')}, float) for l in ligand_label.keys()])

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

def active_composit_feature(composit_dict, dtype=float, by_fraction=True, norm=True, *args, **kwargs):
    feat_vec = np.zeros(len(ActiveElements), dtype=dtype)
    for ele, frac in composit_dict.items():
        if frac < NEAR_ZERO: continue
        feat_vec[ActiveElements.index(ele)] = frac if by_fraction else 1
    if norm and feat_vec.sum() > 0:
        feat_vec /= feat_vec.sum()
    return feat_vec

def metal_composit_feature(composit_dict, dtype=float, by_fraction=True, *args, **kwargs):
    feat_vec = np.zeros(len(MetalElements) + 1, dtype=dtype)
    for ele, frac in composit_dict.items():
        feat_vec[MetalElements.index(ele) + 1] = frac if by_fraction else 1
    if feat_vec.sum() == 0:
        feat_vec[0] = 1
    return feat_vec

def ligand_composit_feature(composit_dict, dtype=float, by_fraction=True, *args, **kwargs):
    feat_vec = np.zeros(len(LigandElements) + 1, dtype=dtype)
    for ele, frac in composit_dict.items():
        if ele in MetalElements:
            feat_vec[0] += frac if by_fraction else 1
        else:
            feat_vec[LigandElements.index(ele) + 1] = frac if by_fraction else 1
    return feat_vec

def feature_to_composit(feat_vec, tol=1e-5):
    n_feat = feat_vec.shape[-1]
    feat_vec = feat_vec.reshape(-1, n_feat)
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

def parse_feature(feat_vec, csim_cut = 0.5, sser_cut = 1.0, to_string=False):
    out = []
    if feat_vec.shape[-1] == 12:
        ref = ligand_vector
        chrs = np.array(['-'.join(k) for k in ligand_list[:-1]] + ['Unknown'])
    elif feat_vec.shape[-1] == 97:
        ref = precursor_vector
        chrs = np.array(['-'.join(k) for k in precursor_list[:-1]] + ['Unknown'])
    else:
        raise ValueError('Not supported feature type')

    for idx, sser, csim in zip(*find_nearest(feat_vec, ref)):
        if (sser > sser_cut) or (csim < csim_cut):
            out.append(-1)
        else:
            out.append(int(idx))
    if to_string:
        return chrs[out]
    else:
        return out

def parse_composit(comp, feature_type='active_composit'):
    if feature_type.startswith('active'):
        ref = precursor_list
    elif feature_type.startswith('ligand'):
        ref = ligand_list
    key = tuple(sorted(comp.keys(), key=lambda x: Element(x).number))
    return ref.index(key)

def check_blacklist(comp):
    key = tuple(k for k in sorted(comp.keys(), key=lambda x: Element(x).number))
    if key in blacklist:
        return True
    else:
        return False
    
init_info(20)