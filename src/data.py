from re import L
import torch, gzip, pickle
import numpy as np
from .utils import MetalElements
from .feature import composition_to_feature, get_precursor_label, NUM_LABEL, UNK_LABEL, EOS_LABEL
from typing import Dict, List

class BaseData:
    def __init__(self, 
                 data : Dict = {}, 
                 info_attrs : List[str] = ['id','year'],
                 *args, **kwargs):
        
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

class CompositionData(BaseData):
    def __init__(self, 
                 comp : Dict, 
                 data : Dict = {}, 
                 feat_type : str = 'composit', 
                 label : int = None,
                 weight : float = None,
                 *args, **kwargs):
        super().__init__(data, *args, **kwargs)

        self._info_attrs.extend(['comp', 'feat_type'])
        self._feature_attrs.append('feat')

        self.feat_type = feat_type
        self.comp = comp
        self.feat = composition_to_feature(comp, feature_type=feat_type, by_fraction=True)

        if label is not None:
            self.label = np.array(label, dtype=int).reshape(-1,1)
            self._feature_attrs.append('label')
        if weight is not None:
            self.weight = np.array(weight, dtype=np.float32).reshape(-1,1)
            self._feature_attrs.append('weight')

        self.to_torch()

class ReactionData(BaseData):
    def __init__(self, 
                 data : Dict = {},
                 feat_type : str = 'composit',
                 target_comp : Dict = {},
                 precursor_comps : List[Dict] = [],
                 conditions : List[str] = [],
                 condition_values : List[float] = [],
                 labels : int = None,
                 weights : float = None,
                 *args, **kwargs):
        super().__init__(data, *args, **kwargs)

        # meta info
        self._info_attrs.append('feat_type')
        self.feat_type = feat_type

        # labels and weights
        if labels is not None:
            self.labels = np.array(labels, dtype=int).reshape(-1,1)
            self._feature_attrs.append('labels')
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32).reshape(-1,1)
            self._feature_attrs.append('weights')

        # target
        self.target_comp = target_comp
        self._info_attrs.append('target_comp')
        self._feature_attrs.append('target_feat')
        self.target_feat = composition_to_feature(composit_dict=target_comp, 
                                                  feature_type=feat_type, 
                                                  by_fraction=True)

        # conditions
        for attr, value in zip(conditions, condition_values):
            setattr(self, attr, np.array(value, dtype=np.float32).reshape(1,-1))
            self._feature_attrs.append(attr)

        # metal and precursor
        metal_comp = []
        if isinstance(precursor_comps, List) and len(precursor_comps) != 0:
            self._info_attrs.append('precursor_comps')
            self._feature_attrs.append('precursor_feat')
            
            self.precursor_comps = precursor_comps
            precursor_feat = []
            for precursor_comp in precursor_comps:
                precursor_feat.append(composition_to_feature(composit_dict=precursor_comp, 
                                                             feature_type=feat_type,
                                                             by_fraction=True))
                metal_comp.append({e:f for e,f in precursor_comp.items() if e in MetalElements})
            self.precursor_feat = np.vstack(precursor_feat)
        else:
            for ele, frac in target_comp.items():
                if ele in MetalElements:
                    metal_comp.append({ele:frac})
            metal_comp.append({})

        self._feature_attrs.append('metal_feat')
        self.metal_feat = np.vstack([
            composition_to_feature(composit_dict=metal_comp, 
                                   feature_type=feat_type,
                                   by_fraction=False) 
            for metal_comp in metal_comp])
        self.to_torch()

class GraphData(ReactionData):
    def __init__(self, 
                 data : Dict = {},
                 feat_type : str = 'composit',
                 target_comp : Dict = {},
                 precursor_comps : List[Dict] = [],
                 conditions : List[str] = [],
                 condition_values : List[float] = [],
                 labels : int = None,
                 weights : float = None,
                 *args, **kwargs):
        
        super().__init__(data = data,
                         feat_type = feat_type,
                         target_comp = target_comp,
                         precursor_comps = precursor_comps,
                         conditions = conditions,
                         condition_values = condition_values,
                         labels = labels,
                         weights = weights,
                         *args, **kwargs)

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
                                               by_fraction = True,
                                               norm = True)
                    )
        self.edge_index = np.array(edge_index, dtype=int).T
        self.edge_feat = np.vstack(edge_feat, dtype=np.float32)

class SequenceData(ReactionData):
    def __init__(self, 
                 data : Dict = {},
                 feat_type : str = 'composit',
                 target_comp : Dict = {},
                 precursor_comps : List[Dict] = [],
                 conditions : List[str] = [],
                 condition_values : List[float] = [],
                 max_length : int = 8,
                 labels : int = None,
                 weights : float = None,
                 *args, **kwargs):
        super().__init__(data = data,
                         feat_type = feat_type,
                         target_comp = target_comp,
                         precursor_comps = precursor_comps,
                         conditions = conditions,
                         condition_values = condition_values,
                         labels = None,
                         weights = weights,
                         *args, **kwargs)
        
        self._feature_attrs.pop('metal_feat')
        delattr(self, 'metal_feat')

        # precursor feat
        if hasattr(self, 'precursor_feat'):
            pad = np.zeros((self.precursor_feat.shape[0], max_length), dtype=np.float32)
            self.precursor_feat = np.vstack([
                pad[0].reshape(1,-1), self.precursor_feat, pad
            ])[:max_length]

        # labels
        self._feature_attrs.append('labels')
        if labels is None:
            self.labels = np.array([EOS_LABEL]).astype(int)
        else:
            padded = [EOS_LABEL] + [l for l in np.array(labels).reshape(-1)] + [EOS_LABEL] * max_length
            self.labels = np.array(padded[:max_length]).astype(int)

        self.to_torch()

################################################################################################
#
# Dataset class
#
################################################################################################

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.init_dataset()
        self._train = False
        self.num_labels = NUM_LABEL

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

    def update_info(self):
        if len(self._data) == 0:
            return
        for attr in self._data[0]._feature_attrs:
            if 'feat' not in attr:
                continue
            vec = getattr(self._data[0], attr)
            setattr(self, f'num_{attr}', vec.shape[1])

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
    def __init__(self, comp_feat_type='composit', by_fraction=True):
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
            comp_data = CompositionData(data=data, 
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
    
class ReactionDataset(BaseDataset):
    def __init__(self, feat_type:str='composit', data_type='sequence'):
        super().__init__()
        self.has_temp_info = False
        self.has_time_info = False

        self._feat_type = feat_type
        if 'graph' in data_type.lower():
            self._data_type = 'graph'
        elif 'seq' in data_type.lower():
            self._data_type = 'sequence'
        else:
            self._data_type = 'reaction'
        
        self._train = False
        self._data = []

    def from_file(self, data_path, extend_dataset=False, 
                  target_comp_key='target_comp', precursor_comp_key='precursor_comp', 
                  heat_temp_key=None, heat_time_key=None, *args, **kwargs):
        dataset = self.read_file(data_path)
        self.from_dataset(dataset, extend_dataset = extend_dataset, 
                          target_comp_key = target_comp_key, 
                          precursor_comp_key = precursor_comp_key,
                          heat_temp_key = heat_temp_key, 
                          heat_time_key = heat_time_key, 
                          *args, **kwargs)
    
    def from_dataset(self, dataset:List[Dict], extend_dataset=False, 
                     target_comp_key='target_comp', precursor_comp_key='precursor_comp', 
                     heat_temp_key=None, heat_time_key=None, info_attrs=['id','year'], 
                     *args, **kwargs):
        self.init_dataset(extend_dataset)
        for data in dataset:
            self.from_data(data = data,
                           target_comp_key = target_comp_key, 
                           precursor_comp_key = precursor_comp_key,
                           heat_temp_key = heat_temp_key, 
                           heat_time_key = heat_time_key, 
                           info_attrs = info_attrs,
                           **kwargs)

    def from_data(self, data, target_comp_key, precursor_comp_key=None, 
                  heat_temp_key=None, heat_time_key=None, info_attrs=[], 
                  *args, **kwargs):
        precursor_comps = []
        precursor_labels = None
        if precursor_comp_key is not None and precursor_comp_key in data.keys():
            precursor_comps = data[precursor_comp_key]
            precursor_labels = [get_precursor_label(p) for p in precursor_comps]
            precursor_weights = None
            self._train = True
        conditions = []
        condition_values = []
        if heat_temp_key is not None:
            self.has_temp_info = True
            key, val = heat_temp_key
            if data[key][val] is None:
                return
            conditions.append(['heat_temp'])
            condition_values.append(data[key][val])
        if heat_time_key is not None:
            self.has_time_info = True
            key, val = heat_time_key
            if data[key][val] is None:
                return
            conditions.append(['heat_time'])
            condition_values.append(data[key][val])

        if self._data_type == 'reaction':
            self._data.append(
                ReactionData(data = data,
                            feat_type = self._feat_type,
                            target_comp = data[target_comp_key], 
                            precursor_comps = precursor_comps,
                            conditions = conditions,
                            condition_values = condition_values,
                            labels = precursor_labels,
                            weights = precursor_weights,
                            info_attrs = info_attrs,
                            *args, **kwargs)
            )
        elif self._data_type == 'graph':
            self._data.append(
                GraphData(data = data,
                          feat_type = self._feat_type,
                          target_comp = data[target_comp_key], 
                          precursor_comps = precursor_comps,
                          conditions = conditions,
                          condition_values = condition_values,
                          labels = precursor_labels,
                          weights = precursor_weights,
                          info_attrs = info_attrs,
                          *args, **kwargs)
            )
        elif self._data_type == 'sequence':
            self._data.append(
                SequenceData(data = data,
                             feat_type = self._feat_type,
                             target_comp = data[target_comp_key], 
                             precursor_comps = precursor_comps,
                             conditions = conditions,
                             condition_values = condition_values,
                             labels = precursor_labels,
                             weights = precursor_weights,
                             info_attrs = info_attrs,
                             *args, **kwargs)
            )

    def cfn(self, dataset):
        info = []
        x = []
        rxn_index = []
        n = 0
        for i, data in enumerate(dataset):
            nx = data.metal_feat.shape[0]

        

    def fcfn(self, dataset):
        info = []
        node_feat = []
        edge_attr = []
        edge_index = []
        rxn_index = []
        n = 0
        for i, data in enumerate(dataset):
            nx = data.metal_feat.shape[0]
            node = getattr(data, 'metal_feat')
            if self.has_temp_info:
                node = torch.concat([
                    node, data.heat_temp.repeat((nx, 1))
                ], dim=-1)
            if self.has_time_info:
                node = torch.concat([
                    node, data.heat_time.repeat((nx, 1))
                ], dim=-1)
            node_feat.append(node)
            edge_attr.append(data.edge_feat)
            edge_index.append(data.edge_index + n)
            rxn_index.append([i] * nx)
            info.append(data.to_dict())
            n += nx
        
        x = []
        labels = []
#        weights = []
        if self._train:
            for data in dataset:
                x.append(data.precursor_feat)
                labels.append(data.labels)
#                weights.append(data.weights)
            x = torch.concat(x)
            labels = torch.concat(labels)
#            weights = torch.concat(weights)

        feat = {
            'x':x,
            'label':labels,
#            'weights':weights,
            'node_feat':torch.concat(node_feat),
            'edge_attr':torch.concat(edge_attr),
            'edge_index':torch.concat(edge_index, -1),
            'rxn_index':np.hstack(rxn_index)
        }
        return feat, info

    def gcfn(self, dataset):
        feat, info = self.fcfn(dataset)
        for data in dataset:


    def rscfn(self, dataset):
        info = []       
        rxn_index = []
        for i, data in enumerate(dataset):
            nx = data.metal_feat.shape[0]

            rxn_index.append([i] * nx)
            info.append(data.to_dict())
