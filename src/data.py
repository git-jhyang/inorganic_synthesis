import torch, gzip, pickle
import numpy as np
from .utils import MetalElements
from .feature import composition_to_feature
from typing import Dict, List

class BaseData:
    def __init__(self, data:Dict={}, info_attrs:List[str] = ['id','raw_id','year']):
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

class CompData(BaseData):
    def __init__(self, comp:Dict, data:Dict={}, comp_feat_type:str='active_composit', by_fraction=True, **kwargs):
        super(CompData, self).__init__(data, **kwargs)
        self.comp_feat_type = [comp_feat_type, by_fraction]
        self.comp = comp
        self.comp_feat = composition_to_feature(comp, feature_type=comp_feat_type, by_fraction=by_fraction)
        self._info_attrs.extend(['comp', 'comp_feat_type'])
        self._feature_attrs.append('comp_feat')
        self.to_torch()

class MetalCondLigandData(BaseData):
    def __init__(self, 
                 target_comp:Dict,                
                 data:Dict={},
                 ligand_comp:Dict=None,
                 metal_comp:Dict={}, 
                 ligand_feat_type:str='ligand_composit',
                 ligand_feat_by_fraction:bool=False,
                 target_feat_type:str='active_composit',
                 target_feat_by_fraction:bool=True,
                 metal_feat_type:str='metal_composit',
                 metal_feat_by_fraction:bool=False,
                 **kwargs):
        super(MetalCondLigandData, self).__init__(data, **kwargs)
        
        if ligand_comp is not None:
            self.ligand_comp = ligand_comp
            self.ligand_feat = composition_to_feature(ligand_comp, feature_type=ligand_feat_type, by_fraction=ligand_feat_by_fraction)
            self.ligand_feat_type = [ligand_feat_type, ligand_feat_by_fraction]
            metal_comp = {e:f for e,f in ligand_comp.items() if e in MetalElements}
            self._info_attrs.extend(['ligand_comp', 'ligand_feat_type'])
            self._feature_attrs.append('ligand_feat')
        
        target_feat = composition_to_feature(target_comp, feature_type=target_feat_type, by_fraction=target_feat_by_fraction)
        element_feat = composition_to_feature(metal_comp, feature_type=metal_feat_type, by_fraction=metal_feat_by_fraction)

        self.target_comp = target_comp
        self.target_feat_type = [target_feat_type, target_feat_by_fraction]
        self.metal_comp = metal_comp
        self.metal_feat_type = [metal_feat_type, metal_feat_by_fraction]

        self.cond_feat = np.hstack([element_feat, target_feat])
        self._info_attrs.extend(['target_comp','metal_comp'])
        self._feature_attrs.append(['cond_feat'])

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.init_data()

    def init_data(self):
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
        super(CompositionDataset, self).__init__()
        self._comp_feat_type = comp_feat_type
        self._by_fraction = by_fraction

    def from_file(self, data_path='../data/unique_target.pkl.gz', extend_dataset=False):
        if not extend_dataset:
            self.init_data()
        dataset = self.read_file(data_path)

        for data in dataset:
            comp_data = CompData(data=data, comp=data['target_comp'], comp_feat_type=self._comp_feat_type, by_fraction=self._by_fraction)
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
                 ligand_feat_type:str='ligand_composit',
                 target_feat_type:str='active_composit',
                 metal_feat_type:str='metal_composit',
    ):
        super(ConditionDataset, self).__init__()
        self._ligand_feat_type = ligand_feat_type
        self._target_feat_type = target_feat_type
        self._metal_feat_type = metal_feat_type

    def from_file(self, data_path='../data/unique_target.pkl.gz', extend_dataset=False):
        if not extend_dataset:
            self.init_data()
        dataset = self.read_file(data_path)

        for data in dataset:
            for prec_data in data['precursors']:
                MetalCondLigandData(data=data, 
                                    prec_data['precursor_comp'],
                )

    def from_targets(self, target_comps:List[Dict], extend_dataset=False):
        if not extend_dataset:
            self.init_data()
        for target_comp in target_comps:
            self.from_target(target_comp, True)

    def from_target(self, target_comp:Dict, extend_dataset=True):
        if not extend_dataset:
            self.init_data()
        for ele, frac in target_comp.items():
            if ele not in MetalElements:
                continue
            MetalCondLigandData(info_attrs=[],
                                data=  {ele:frac}

