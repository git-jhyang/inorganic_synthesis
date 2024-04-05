from pymatgen.core import Element
from .utils import SortedAllElements, MetalElements, NEAR_ZERO
import numpy as np
import torch

def composition_to_feature(composit_dict, feature_type='basic', dtype=np.float32):
    '''
    composit_dict = {
        element_0 (str): fraction_0 (float),
        element_1 (str): fraction_1 (float),
        ...
    }
    '''
    elmd_type = ['atomic','cgcnn','element','']
    if feature_type == 'basic':
        return basic_composit_feature(composit_dict=composit_dict, dtype=dtype)
    elif feature_type in elmd_type:
        pass
    else:
        raise TypeError(f'feature type [{feature_type}] is not supported',
                        
                        )
    


def basic_composit_feature(composit_dict, dtype):
    feat_vec = np.zeros(104, dtype=dtype)
    for ele, frac in composit_dict.items():
        feat_vec[Element(ele).number] = frac
    feat_vec /= feat_vec.sum()
    return feat_vec

#def 
