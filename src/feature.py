from pymatgen.core import Element
from .utils import SortedAllElements, MetalElements, NEAR_ZERO
import numpy as np
import torch, json, os

elmd = {}
exception = {}
for k in ['cgcnn','elemnet','magpie','magpie_sc','mat2vec','matscholar','megnet16','oliynyk','oliynyk_sc']:
    elmd_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elmd', f'{k}.json')))
    elmd[k] = elmd_data
    exception[k] = [k for k in SortedAllElements[1:] if k not in elmd_data.keys()]

PRINT_EXCEPTION_WARNING = 'Warning: element "{}" is not included in feature type "{}"\n'

def composition_to_feature(composit_dict, feature_type='basic', dtype=np.float32):
    '''
    composit_dict = {
        element_0 (str): fraction_0 (float),
        element_1 (str): fraction_1 (float),
        ...
    }
    '''
    vec = []
    if feature_type == 'basic':
        return basic_composit_feature(composit_dict=composit_dict, dtype=dtype)
    elif feature_type in elmd.keys():
        norm = 1.0 / np.sum(list(composit_dict.values()))
        for ele, frac in composit_dict.items():
            if ele in exception[feature_type]:
                if PRINT_EXCEPTION_WARNING is not None:
                    print(PRINT_EXCEPTION_WARNING.format(ele, feature_type))
                continue
            vec.append(np.array(elmd[feature_type][ele]) * frac)
        PRINT_EXCEPTION_WARNING = None
        return np.sum(vec, 0) * norm
    else:
        raise TypeError(f'feature type [{feature_type}] is not supported', list(elmd.keys()))

def basic_composit_feature(composit_dict, dtype):
    feat_vec = np.zeros(104, dtype=dtype)
    for ele, frac in composit_dict.items():
        feat_vec[Element(ele).number] = frac
    feat_vec /= feat_vec.sum()
    return feat_vec