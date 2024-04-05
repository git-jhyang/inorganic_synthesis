import torch, gzip, pickle
import numpy as np
from pymatgen.core import Element
from .utils import SortedAllElements, MetalElements, NEAR_ZERO
from typing import Dict

def composition_to_feature(composit_dict, dtype=np.float32):
    '''
    composit_dict = {
        element_0 (str): fraction_0 (float),
        element_1 (str): fraction_1 (float),
        ...
    }
    1-dim vector of length 104
    maximum atomic number: 103 (Lr, Lawrencium)
    elements from Rf (104) to Og (118) are not included
    '''
    feat_vec = np.zeros(104, dtype=dtype)
    have_metal = False
    for ele, frac in composit_dict.items():
        if ele in MetalElements: have_metal = True
        feat_vec[Element(ele).number] = frac
    feat_vec /= feat_vec.sum()
    if not have_metal:
        feat_vec *= 0.5
        feat_vec[0] = 1 - feat_vec.sum()
    return feat_vec

def feature_to_composition(feature_vector, tol=NEAR_ZERO):
    '''
    1-dim vector of length 104
    maximum atomic number: 103 (Lr, Actinoids)
    elements from Rf (104) to Og (118) are not included
    '''
    if isinstance(feature_vector, torch.Tensor):
        feature_vector = feature_vector.cpu().numpy()
    feature_vector[feature_vector <= tol] = 0
    if feature_vector[0] != 0:
        feature_vector[0] = 0
        feature_vector *= 2
    idxs = np.where(feature_vector)[0]
    return {SortedAllElements[i]: feature_vector[i] for i in idxs} 

class BaseData:
    def __init__(self, data:Dict, info_attrs=['id','year','count_weight']):
        self._info_attrs = []
        self._feature_attrs = []
        self.device = None
        for attr in info_attrs:
            setattr(self, attr, data.get(attr))
            if data.get(attr) is not None:
                self._info_attrs.append(attr)

    def to_numpy(self):
        for attr in self._feature_attrs:
            data = getattr(self, attr)
            if isinstance(data, torch.Tensor):
                setattr(self, attr, data.cpu().numpy())
        self.device = None
    
    def to_torch(self):
        for attr in self._feature_attrs:
            data = getattr(self, attr)
            if isinstance(data, np.ndarray):
                setattr(self, attr, torch.from_numpy(data))
        self.device = 'cpu'
    
    def to(self, device='cpu'):
        if len(self._feature_attrs) == 0:
            return
        data = getattr(self, self._feature_attrs[0])
        if isinstance(data, np.ndarray):
            self.to_torch()
        for attr in self._feature_attrs:
            data = getattr(self, attr)
            setattr(self, attr, data.to(device))
        self.device = device

    def to_dict(self):
        output_dict = {attr:getattr(self, attr) for attr in self._info_attrs}
        for attr in self._feature_attrs:
            if 'comp' in attr:
                output_dict.update({attr:feature_to_composition(getattr(self, attr))})
            else:
                output_dict.update({attr:getattr(self, attr)})
        return output_dict

    def __dict__(self):
        return self.to_dict()

class CompositionData(BaseData):
    def __init__(self, data:Dict, composition:Dict, **kwargs):
        super(CompositionData, self).__init__(data, **kwargs)
        self.comp_feat = composition_to_feature(composition)
        self._feature_attrs.append('comp_feat')
        self.to_torch()

class ConditionalData(BaseData):
    def __init__(self, data:Dict, composition:Dict=None, element:str = None, **kwargs):
        super(ConditionalData, self).__init__(data, **kwargs)
        self.target_comp_feat = composition_to_feature(data['target'])
        
        if element is None:
            self.element_feat = composition_to_feature({})
        elif isinstance(element, str):
            self.element_feat = composition_to_feature({element:1})

        if composition is not None:
            self.precursor_comp_feat = composition_to_feature(composition)
            elements = [ele for ele in composition.keys() if ele in MetalElements]
            if len(elements) != 0:
                self.element_feat = composition_to_feature({elements[0]:1})
                if len(elements) != 1:
                    print('There is multiple metal elements in precursor. ID:', data['id'], composition)
            self._feature_attrs.append(['precursor_comp_feat'])

        self._feature_attrs.extend(['target_comp_feat','element_feat'])

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self._data = []
        self._year = []
        self._element = []
    
    def to_numpy(self):
        for data in self._data:
            data.to_numpy()
    
    def to_torch(self):
        for data in self._data:
            data.to_torch()
    
    def to(self, device):
        for data in self._data:
            data.to(device)

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, i):
        return self._data[i]
    

#    def cfn(self, dataset):
#        feat = []
#        info = []
#        for data in dataset:
#            feat.append(data.comp_feat)
#            info.append(data.to_dict())
#        return torch.vstack(feat), info
