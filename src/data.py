from pymatgen.core import Element
from utils import MetalElements, FunctionalGroupElements
import numpy as np

metal_mapper = {
    
}

def composit_feature(composit_dict):
    '''
    composit_dict = {
        element_0 (str): fraction_0 (float),
        element_1 (str): fraction_1 (float),
        ...
    }
    '''
    feat_vec = np.zeros(104, dtype=np.float32)
    for ele, frac in composit_dict.items():
        feat_vec[Element(ele).number] = frac
    return feat_vec

def 