from re import L
import torch, gzip, pickle
import numpy as np
from .utils import MetalElements
from .feature import composition_to_feature
from typing import Dict, List

class BaseData:
    def __init__(self, data:Dict={}, info_attrs:List[str] = ['id','raw_id','year']):
        self._info_attrs = info_attrs.copy()
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

class CompositData(BaseData):
    def __init__(self, comp:Dict, data:Dict={}, comp_feat_type:str='active_composit', by_fraction=True, **kwargs):
        super().__init__(data, **kwargs)
        self.comp_feat_type = [comp_feat_type, by_fraction]
        self.comp = comp
        self.comp_feat = composition_to_feature(comp, feature_type=comp_feat_type, by_fraction=by_fraction)
        self._info_attrs.extend(['comp', 'comp_feat_type'])
        self._feature_attrs.append('comp_feat')
        self.to_torch()

class PrecursorData(BaseData):
    def __init__(self, 
                 target_comp:Dict,
                 data:Dict={},
                 target_feat_type:str='active_composit',
                 target_feat_by_fraction:bool=True,
                 metal_comp:Dict={}, 
                 metal_feat_type:str='metal_composit',
                 metal_feat_by_fraction:str=False,
                 precursor_comp:Dict=None,
                 precursor_feat_type:str='ligand_composit',
                 precursor_feat_by_fraction:bool=False,
                 **kwargs):
        super().__init__(data, **kwargs)
        
        if isinstance(precursor_comp, Dict):
            self.precursor_comp = precursor_comp
            self.precursor_feat = composition_to_feature(precursor_comp, feature_type=precursor_feat_type, by_fraction=precursor_feat_by_fraction)
            self.precursor_feat_type = [precursor_feat_type, precursor_feat_by_fraction]
            metal_comp = {e:f for e,f in precursor_comp.items() if e in MetalElements}
            self._info_attrs.extend(['precursor_comp', 'precursor_feat_type'])
            self._feature_attrs.append('precursor_feat')
        
        self.target_feat = composition_to_feature(target_comp, feature_type=target_feat_type, by_fraction=target_feat_by_fraction)
        self.metal_feat = composition_to_feature(metal_comp, feature_type=metal_feat_type, by_fraction=metal_feat_by_fraction)

        self.target_comp = target_comp
        self.target_feat_type = [target_feat_type, target_feat_by_fraction]
        self.metal_comp = metal_comp
        self.metal_feat_type = [metal_feat_type, metal_feat_by_fraction]

        self._info_attrs.extend(['target_comp', 'target_feat_type', 'metal_comp', 'metal_feat_type'])
        self._feature_attrs.extend(['target_feat','metal_feat'])
        self.to_torch()

class ReactionData(BaseData):
    def __init__(self, 
                 data:Dict,
                 target_comp_key='target_comp', 
                 target_feat_type:str='active_composit',
                 target_feat_by_fraction:bool=True,
                 metal_feat_type:str='metal_composit',
                 metal_feat_by_fraction:bool=False,
                 precursor_comp_key:str='precursor_comp',
                 precursor_feat_type:str='ligand_composit',
                 precursor_feat_by_fraction:bool=False,
                 **kwargs):
        super().__init__(data, **kwargs)
        self.graph_idx = 0
        self.target_comp = data[target_comp_key]
        self.target_comp_feat = composition_to_feature(composit_dict = self.target_comp, 
                                                       feature_type = target_feat_type, 
                                                       by_fraction = target_feat_by_fraction)
        
        self.metal_comp = []
        if precursor_comp_key:
            precursor_comp_feat = []
            self.precursor_comp = data[precursor_comp_key]
            for precursor in self.precursor_comp:
                precursor_comp_feat.append(
                    composition_to_feature(
                        composit_dict = precursor, 
                        feature_type = precursor_feat_type, 
                        by_fraction = precursor_feat_by_fraction
                    )
                )
                self.metal_comp.append({e:f for e,f in precursor.items() if e in MetalElements})
        else:
            for ele, frac in self.target_comp.items():
                if ele in MetalElements:
                    self.metal_comp.append({ele:frac})
            self.metal_comp.append({})
        self.metal_comp_feat = [
            composition_to_feature(
                composit_dict = metal_comp,
                feature_type = metal_feat_type, 
                by_fraction = metal_feat_by_fraction
            ) for metal_comp in self.metal_comp
        ]
    
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.init_dataset()

    def init_dataset(self, extend_dataset=False):
        if not extend_dataset:
            self._data = []

    def read_file(self, data_path):
        if ('.gz' in data_path) or ('.gzip' in data_path):
            f = gzip.open(data_path, 'rb')
        else:
            f = open(data_path, 'rb')
        dataset = pickle.load(f)
        f.close()
        return dataset

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
    def __init__(self, comp_feat_type='active_composit', by_fraction=True):
        super().__init__()
        self._comp_feat_type = comp_feat_type
        self._by_fraction = by_fraction

    def from_file(self, data_path='../data/unique_target.pkl.gz', extend_dataset=False):
        self.init_dataset(extend_dataset)
        dataset = self.read_file(data_path)

        for data in dataset:
            comp_data = CompositData(data=data, 
                                 comp=data['target_comp'], 
                                 comp_feat_type=self._comp_feat_type, 
                                 by_fraction=self._by_fraction)
            self._data.append(comp_data)
        self._year = np.array([d.year for d in self._data])
        self.to_torch()
        self.to('cpu')

    def cfn(self, dataset):
        feat = []
        info = []
        for data in dataset:
            feat.append(data.comp_feat)
            info.append(data.to_dict())
        return torch.vstack(feat), info

class ConditionDataset(BaseDataset):
    def __init__(self, 
                 precursor_feat_by_fraction:bool=False,
                 target_feat_type:str='active_composit',
                 target_feat_by_fraction:bool=True,
                 metal_feat_type:str='metal_composit',
                 metal_feat_by_fraction:bool=False,
                 train_data:bool=True,
                 ):
        
        super().__init__()

        self._precursor_feat_by_fraction = precursor_feat_by_fraction
        self._target_feat_type = target_feat_type
        self._target_feat_by_fraction = target_feat_by_fraction
        self._metal_feat_type = metal_feat_type
        self._metal_feat_by_fraction = metal_feat_by_fraction
        
        self._train_data = train_data

        dump = self.from_precursor({}, {})
        self.num_metal_feat = dump.metal_feat.shape[-1]
        self.num_target_feat = dump.target_feat.shape[-1]
        self.num_precursor_feat = dump.precursor_feat.shape[-1]

    def from_file(self, data_path='../data/unique_reaction.pkl.gz', extend_dataset=False, 
                  target_comp_key='target_comp', precursor_comp_key='precursor_comp', **kwargs):
        
        dataset = self.read_file(data_path)
        self.from_dataset(dataset, extend_dataset=extend_dataset, 
                          target_comp_key=target_comp_key, 
                          precursor_comp_key=precursor_comp_key,
                          **kwargs)
    
    def from_dataset(self, dataset:List[Dict], extend_dataset=False, 
                     target_comp_key='target_comp', precursor_comp_key='precursor_comp', 
                     info_attrs=['id','raw_id','year'], **kwargs):
        self.init_dataset(extend_dataset)
        for data in dataset:
            self.from_reaction(
                target_comp = data[target_comp_key], 
                precursor_comps= None if precursor_comp_key is None else data[precursor_comp_key],
                extend_dataset = True, data = data, info_attrs=info_attrs, **kwargs
            )

    def from_reaction(self, target_comp:Dict, precursor_comps:List[Dict]=None, extend_dataset=False, data={}, info_attrs=[], **kwargs):
        self.init_dataset(extend_dataset)
        if precursor_comps is None:
            for ele, frac in target_comp.items():
                if ele not in MetalElements:
                    continue
                self._data.append(self.from_metal(metal_comp={ele:frac}, target_comp=target_comp, data=data, info_attrs=info_attrs, **kwargs))
            self._data.append(self.from_metal(metal_comp={}, target_comp=target_comp, data=data, info_attrs=info_attrs, **kwargs))
        else:
            for precursor_comp in precursor_comps:
                self._data.append(self.from_precursor(precursor_comp=precursor_comp, target_comp=target_comp, data=data, info_attrs=info_attrs, **kwargs))

    def from_precursor(self, precursor_comp:Dict, target_comp:Dict, data={}, info_attrs=[], **kwargs):
        return PrecursorData(
                data=data,
                target_comp = target_comp,
                precursor_comp = precursor_comp, 
                precursor_feat_type = 'ligand_composit',
                precursor_feat_by_fraction = self._precursor_feat_by_fraction,
                target_feat_type = self._target_feat_type,
                target_feat_by_fraction = self._target_feat_by_fraction,
                metal_feat_type = self._metal_feat_type,
                metal_feat_by_fraction = self._metal_feat_by_fraction,
                info_attrs=info_attrs,
                **kwargs
            )

    def from_metal(self, metal_comp:Dict, target_comp:Dict, data={}, info_attrs=[], **kwargs):
        return PrecursorData(
                data=data,
                target_comp = target_comp,
                metal_comp = metal_comp, 
                target_feat_type = self._target_feat_type,
                target_feat_by_fraction = self._target_feat_by_fraction,
                metal_feat_type = self._metal_feat_type,
                metal_feat_by_fraction = self._metal_feat_by_fraction,
                info_attrs=info_attrs,
                **kwargs
            )

    def cfn(self, dataset):
        info = []
        metal_feat = []
        target_feat = []
        for data in dataset:
            metal_feat.append(data.metal_feat)
            target_feat.append(data.target_feat)
            info.append(data.to_dict())

        metal_feat = torch.concat(metal_feat)
        target_feat = torch.concat(target_feat)
        precursor_feat = None
        if self._train_data:
            precursor_feat = torch.concat([data.precursor_feat for data in dataset])
        feat = {
            'x':precursor_feat, 
            'condition':torch.concat([metal_feat, target_feat], -1)
        }
        return feat, info