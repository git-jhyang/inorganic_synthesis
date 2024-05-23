from re import L
import torch, gzip, pickle
import numpy as np
from .utils import MetalElements, composit_parser
from .feature import composition_to_feature
from typing import Dict, List

class BaseData:
    def __init__(self, data:Dict={}, info_attrs:List[str] = ['id','year']):
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
    def __init__(self, 
                 comp:Dict, 
                 data:Dict={}, 
                 feat_type:str='composit', 
                 by_fraction=True, 
                 **kwargs):
        super().__init__(data, **kwargs)
        self._info_attrs.extend(['comp', 'feat_type'])
        self._feature_attrs.append('feat')

        self.feat_type = [feat_type, by_fraction]
        self.comp = comp
        self.feat = composition_to_feature(comp, feature_type=feat_type, by_fraction=by_fraction)
        self.to_torch()

class ReactionData(BaseData):
    def __init__(self, 
                 data:Dict={},
                 target_comp:Dict={},
                 precursor_comps:List[Dict]=[],
                 feat_type:str='composit',
                 target_feat_by_fraction:bool=True,
                 metal_feat_by_fraction:str=False,
                 precursor_feat_by_fraction:bool=True,
                 conditions:List[str]=[],
                 condition_values:List[float]=[],
                 labels:int=None,
                 weights:float=None,
                 **kwargs):
        super().__init__(data, **kwargs)

        # meta info
        self._info_attrs.extend(['feat_type', 'target_comp', 'metal_comp'])
        self._feature_attrs.extend(['target_feat','metal_feat','edge_feat','edge_index'])
        self.feat_type = [feat_type, target_feat_by_fraction, metal_feat_by_fraction, precursor_feat_by_fraction]

        # target
        self.target_comp = target_comp
        self.target_feat = composition_to_feature(composit_dict=target_comp, 
                                                  feature_type=feat_type, 
                                                  by_fraction=target_feat_by_fraction)

        # metal and precursor
        self.metal_comp = []
        if isinstance(precursor_comps, List) and len(precursor_comps) != 0:
            self._info_attrs.extend(['precursor_comp'])
            self._feature_attrs.extend(['precursor_feat', 'label', 'weight'])
            
            self.precursor_comps = precursor_comps
            precursor_feat = []
            for precursor_comp in precursor_comps:
                precursor_feat.append(composition_to_feature(composit_dict=precursor_comp, 
                                                             feature_type=feat_type,
                                                             by_fraction=precursor_feat_by_fraction))
                self.metal_comp.append({e:f for e,f in precursor_comp.items() if e in MetalElements})
            self.precursor_feat = np.vstack(precursor_feat)
        else:
            for ele, frac in target_comp.items():
                if ele in MetalElements:
                    self.metal_comp.append({ele:frac})
            self.metal_comp.append({})

        self.metal_feat = np.vstack([
            composition_to_feature(composit_dict=metal_comp, 
                                   feature_type=feat_type,
                                   by_fraction=metal_feat_by_fraction) 
            for metal_comp in self.metal_comp])

        # graph 
        self._feature_attrs.extend(['edge_feat','edge_index'])
        edge_index = []
        edge_feat = []
        for i, metal_i in enumerate(self.metal_comp):
            for j, metal_j in enumerate(self.metal_comp):
                edge_index.append([i,j])
                if i == j:
                    edge_feat.append(self.target_feat)
                else:
                    edge_comp = {}
                    for e, f in self.target_comp.items():
                        if e in metal_i.keys() or e in metal_j.keys():
                            edge_comp.update({e:f})
                        elif e in MetalElements:
                            continue
                        elif len(metal_i) == 0 or len(metal_j) == 0:
                            edge_comp.update({e:f})
                    edge_feat.append(
                        composition_to_feature(composit_dict = edge_comp, 
                                               feature_type = feat_type, 
                                               by_fraction = target_feat_by_fraction)
                    )
        self.edge_index = np.array(edge_index, dtype=int).T
        self.edge_feat = np.vstack(edge_feat, dtype=np.float32)

        # conditions
        for attr, value in zip(conditions, condition_values):
            setattr(self, attr, np.array(value, dtype=np.float32).reshape(1,-1))
            self._feature_attrs.append(attr)

        # labels and weights
        if labels is not None:
            self.labels = np.array(labels, dtype=int).reshape(-1,1)
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32).reshape(-1,1)

        self.to_torch()

################################################################################################
# Dataset class
################################################################################################

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

    def from_file(self, 
                  data_path='../data/unique_target.pkl.gz', 
                  compsition_key='target_comp',
                  extend_dataset=False):

        self.init_dataset(extend_dataset)
        dataset = self.read_file(data_path)

        for data in dataset:
            comp_data = CompositData(data=data, 
                                     comp=data[compsition_key], 
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

class NNN(BaseDataset):
    def __init__(self, 
                 feat_type:str='active_composit',
                 target_feat_by_fraction:bool=True,
                 metal_feat_by_fraction:bool=False,
                 precursor_feat_by_fraction:bool=False,
                 train:bool=True,
                 ):
        
        super().__init__()

        self._feat_type = feat_type
        self._target_feat_by_fraction = target_feat_by_fraction
        self._metal_feat_by_fraction = metal_feat_by_fraction
        self._precursor_feat_by_fraction = precursor_feat_by_fraction
        
        self._train = train

        dump = self.from_precursor({'Li':1}, {'Li':1})
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
            _data = []
            for precursor_comp in precursor_comps:
                _data.append(self.from_precursor(precursor_comp=precursor_comp, target_comp=target_comp, data=data, info_attrs=info_attrs, **kwargs))
                if _data[-1].skip:
                    return
            self._data.extend(_data)

    def from_precursor(self, precursor_comp:Dict, target_comp:Dict, data={}, info_attrs=[], **kwargs):
        return PrecursorData(
                data=data,
                target_comp = target_comp,
                precursor_comp = precursor_comp, 
                precursor_feat_type = self._precursor_feat_type,
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
        
        precursor_feat = []
        precursor_index = []
        if self._train_data:
            for data in dataset:
                precursor_feat.append(data.precursor_feat)
                precursor_index.append(data.precursor_index)
            precursor_feat = torch.concat(precursor_feat)
            precursor_index = torch.concat(precursor_index)

        feat = {
            'x':precursor_feat, 
            'label':precursor_index,
            'condition':torch.concat([metal_feat, target_feat], -1)
        }
        return feat, info
    
class ReactionDataset(BaseDataset):
    def __init__(self, 
                 feat_type:str='active_composit',
                 target_feat_by_fraction:bool=True,
                 metal_feat_by_fraction:bool=False,
                 precursor_feat_by_fraction:bool=False,
                 train:bool=True,
                 ):
        
        super().__init__()
        
        self._feat_type = feat_type
        self._target_feat_by_fraction = target_feat_by_fraction
        self._metal_feat_by_fraction = metal_feat_by_fraction
        self._precursor_feat_by_fraction = precursor_feat_by_fraction
        
        self._train_data = train

        dump = self.from_data(data={'dump_tgt':{'Li':1}, 'dump_prec':[{'Li':1}]},
                              target_comp_key = 'dump_tgt', 
                              precursor_comp_key = 'dump_prec',
                              info_attrs = [])
        self.num_metal_feat = dump.metal_feat.shape[-1]
        self.num_target_feat = dump.edge_feat.shape[-1]
        self.num_precursor_feat = dump.precursor_feat.shape[-1]
        self.num_class = None

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
            _data = self.from_data(data = data,
                                target_comp_key = target_comp_key, 
                                precursor_comp_key = precursor_comp_key,
                                info_attrs = info_attrs,
                                **kwargs)
            if _data.skip:
                continue
            self._data.append(_data)

    def from_data(self, data, target_comp_key, precursor_comp_key=None, info_attrs=[], *args, **kwargs):
        return ReactionData(data = data,
                            target_comp_key = target_comp_key, 
                            target_feat_type = self._target_feat_type,
                            target_feat_by_fraction = self._target_feat_by_fraction,
                            metal_feat_type = self._metal_feat_type,
                            metal_feat_by_fraction = self._metal_feat_by_fraction,
                            precursor_comp_key = precursor_comp_key,
                            precursor_feat_type = self._precursor_feat_type,
                            precursor_feat_by_fraction = self._precursor_feat_by_fraction,
                            info_attrs = info_attrs,
                            **kwargs)

    def cfn(self, dataset):
        info = []
        metal_feat = []
        edge_feat = []
        edge_index = []
        n = 0
        for data in dataset:
            n_node = data.metal_feat.shape[0]
            metal_feat.append(data.metal_feat)
            edge_feat.append(data.edge_feat)
            edge_index.append(data.edge_index + n)
            info.append(data.to_dict())
            n += n_node
        
        precursor_index = []
        precursor_feat = []
        if self._train_data:
            for data in dataset:
                precursor_feat.append(data.precursor_feat)
                precursor_index.append(data.precursor_index)
            precursor_feat = torch.concat(precursor_feat)
            precursor_index = torch.concat(precursor_index)

        feat = {
            'x':precursor_feat,
            'label':precursor_index,
            'edge_attr':torch.concat(edge_feat),
            'edge_index':torch.concat(edge_index, -1),
            'condition':torch.concat(metal_feat),
        }
        return feat, info