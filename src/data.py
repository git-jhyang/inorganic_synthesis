import torch, gzip, pickle
import numpy as np
from .utils import MetalElements
from .feature import composition_to_feature
from typing import Dict, List

class BaseData:
    def __init__(self, data:Dict, info_attrs:List[str] = ['id','raw_id','year','feature_type']):
        self._info_attrs = info_attrs
        self._feature_attrs = []
        self.device = None
        for attr in self._info_attrs:
            setattr(self, attr, data.get(attr))

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
        if self.device is None:
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
    def __init__(self, data:Dict, 
                 composition:Dict=None, 
                 metal_composition:Dict={}, 
                 target_feature_type:str='basic', 
                 composition_feature_type:str='basic',
                 metal_feature_type:str='metal_frac',
                 **kwargs):
        super(ConditionalData1, self).__init__(data, **kwargs)
        
        target_feat = composition_to_feature(data['target'], feature_type=target_feature_type, **kwargs)

        if composition is not None:
            self.composition = composition
            self.comp_feat = composition_to_feature(composition, feature_type=composition_feature_type, **kwargs)
            metal_composition = {e:f for e,f in composition.items() if e in MetalElements}
            self._feature_attrs.append('comp_feat')
        element_feat = composition_to_feature(metal_composition, feature_type=metal_feature_type)

        self.metal_composition = metal_composition
        self.cond_feat = np.hstack([element_feat, target_feat])
        self._info_attrs.append('metal_composition')
        self._feature_attrs.append('cond_feat')

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

    def cfn(self, dataset):
        feat = []
        info = []
        for data in dataset:
            feat.append(data.comp_feat)
            info.append(data.to_dict())
        return torch.vstack(feat), info
