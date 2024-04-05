import torch, gzip, pickle
import numpy as np
from .utils import MetalElements
from .feature import composition_to_feature
from typing import Dict

class BaseData:
    def __init__(self, data:Dict, info_attrs=['id','year','feature_type']):
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
        return output_dict

    def __dict__(self):
        return self.to_dict()

class CompositionData(BaseData):
    def __init__(self, data:Dict, composition:Dict, feature_type='basic', **kwargs):
        super(CompositionData, self).__init__(data, **kwargs)
        self.feature_type = feature_type
        self.composition = composition
        self.comp_feat = composition_to_feature(composition, feature_type=feature_type, **kwargs)
        self._info_attrs.append('composition')
        self._feature_attrs.append('comp_feat')
        self.to_torch()

class ConditionalData1(BaseData):
    def __init__(self, data:Dict, feature_type:str='basic',
                  prec_composition:Dict=None, core_element:str = None, **kwargs):
        super(ConditionalData1, self).__init__(data, **kwargs)
        self.target_comp_feat = composition_to_feature(data['target'], feature_type=feature_type, **kwargs)
        
        if core_element is None:
            self.element_feat = composition_to_feature({})
        elif isinstance(core_element, str):
            self.element_feat = composition_to_feature({core_element:1})

        if prec_composition is not None:
            self.precursor_comp_feat = composition_to_feature(prec_composition, feature_type=feature_type, **kwargs)
            elements = [ele for ele in prec_composition.keys() if ele in MetalElements]
            if len(elements) != 0:
                self.element_feat = composition_to_feature({elements[0]:1})
                if len(elements) != 1:
                    print('There is multiple metal elements in precursor. ID:', data['id'], prec_composition)
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
    
class CompositionDataset(BaseDataset):
    def __init__(self):
        super(CompositionDataset, self).__init__()

    def from_file(self, data_path='../data/unique_data.pkl.gz'):
        super(CompositionDataset, self).__init__()
        with gzip.open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        
        for data in dataset:
            comp_data = CompositionData(data, data['target_comp'])
            self._data.append(comp_data)
            self._element.append(comp_data.comp_feat != 0)
        self._data = np.array(self._data)
        self._year = np.array([d.year for d in self._data])
        self._element = np.vstack(self._element)

    def __getitem__(self, i):
        data = super().__getitem__(i)
        info = 
        return data.comp_feat, 

#    def cfn(self, dataset):
#        feat = []
#        info = []
#        for data in dataset:
#            feat.append(data.comp_feat)
#            info.append(data.to_dict())
#        return torch.vstack(feat), info
