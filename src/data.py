from re import L
import torch, gzip, pickle
import numpy as np
from zmq import has
from .utils import MetalElements
from .feature import composition_to_feature, get_precursor_label, NUM_LABEL, SOS_LABEL, EOS_LABEL
from typing import Dict, List

class BaseData:
    def __init__(self, 
                 data : Dict = {}, 
                 base_info_attrs : List[str] = ['id','id_urxn','count','doi','year','year_doc'],
                 info_attrs : List[str] = [],
                 *args, **kwargs):        
        self._info_attrs = []
        self._feature_attrs = []
        self.device = None
        for attr in base_info_attrs + info_attrs:
            if attr not in data.keys():
                continue
            if attr in self._info_attrs:
                continue            
            self._info_attrs.append(attr)
            setattr(self, attr, data[attr])

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
                 heat_temp : float = None,
                 heat_time : float = None,
                 weights : float = None,
                 *args, **kwargs):
        super().__init__(data, *args, **kwargs)

        # meta info
        self._info_attrs.append('feat_type')
        self.feat_type = feat_type

        # weights
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float32).reshape(-1,1)
            self._feature_attrs.append('weights')

        # label, precursor and metal
        metal_comp = []
        if isinstance(precursor_comps, List) and len(precursor_comps) != 0:
            self._info_attrs.append('precursor_comps')
            self._feature_attrs.extend(['labels','precursor_feat'])
            
            self.precursor_comps = precursor_comps
            precursor_feat = []
            labels = []
            for precursor_comp in precursor_comps:
                precursor_feat.append(composition_to_feature(composit_dict=precursor_comp, 
                                                             feature_type=feat_type,
                                                             by_fraction=True))
                labels.append(get_precursor_label(precursor_comp))
                metal_comp.append({e:f for e,f in precursor_comp.items() if e in MetalElements})
            self.precursor_feat = np.vstack(precursor_feat)
            self.labels = np.array(labels, dtype=int).reshape(-1,1)
        else:
            for ele, frac in target_comp.items():
                if ele in MetalElements:
                    metal_comp.append({ele:frac})
            metal_comp.append({})

        # metal
        self._feature_attrs.append('metal_feat')
        self.metal_feat = np.vstack([
            composition_to_feature(composit_dict=metal_comp, 
                                   feature_type=feat_type,
                                   by_fraction=False) 
            for metal_comp in metal_comp])
        self.n = self.metal_feat.shape[0] # number of precursors

        # target
        self.target_comp = target_comp
        self._info_attrs.append('target_comp')
        self._feature_attrs.append('condition_feat')
        self.condition_feat = composition_to_feature(composit_dict=target_comp, 
                                                     feature_type=feat_type, 
                                                     by_fraction=True)

        # conditions
        if heat_temp is not None:
            self._info_attrs.append('heat_temp')
            self.heat_temp = heat_temp
            self.condition_feat = np.hstack([self.condition_feat, [[heat_temp * 0.001 - 1]]]).astype(np.float32)
        if heat_time is not None:
            self._info_attrs.append('heat_time')
            self.heat_time = heat_time
            self.condition_feat = np.hstack([self.condition_feat, [[np.log10(heat_time) - 1]]]).astype(np.float32)

class GraphData(ReactionData):
    def __init__(self, 
                 data : Dict = {},
                 feat_type : str = 'composit',
                 target_comp : Dict = {},
                 precursor_comps : List[Dict] = [],
                 heat_temp : float = None,
                 heat_time : float = None,
                 weights : float = None,
                 *args, **kwargs):
        
        super().__init__(data = data,
                         feat_type = feat_type,
                         target_comp = target_comp,
                         precursor_comps = precursor_comps,
                         heat_temp = heat_temp,
                         heat_time = heat_time,
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
                 heat_temp : float = None,
                 heat_time : float = None,
                 max_length : int = 8,
                 weights : float = None,
                 *args, **kwargs):
        super().__init__(data = data,
                         feat_type = feat_type,
                         target_comp = target_comp,
                         precursor_comps = precursor_comps,
                         heat_temp = heat_temp,
                         heat_time = heat_time,
                         weights = weights,
                         *args, **kwargs)
        
        self._feature_attrs.pop(self._feature_attrs.index('metal_feat'))
        delattr(self, 'metal_feat')
        self.n = max_length

        # labels & precursor feat
        pad = composition_to_feature({}, feature_type=feat_type)
        if hasattr(self, 'precursor_feat'):
            self.m = self.labels.shape[0]
            self.precursor_feat = np.vstack([
                pad.reshape(1,-1), self.precursor_feat, *[pad] * max_length
            ])[:max_length].astype(np.float32)[np.newaxis, ...]
            self.labels = np.hstack([
                [SOS_LABEL], self.labels.reshape(-1), [EOS_LABEL] * max_length
            ])[:max_length].astype(int).reshape(1,-1)
        else:
            self._feature_attrs.append('precursor_feat')
            self._feature_attrs.append('labels')
            self.precursor_feat = pad.reshape(1, 1, -1)
            self.labels = np.array([SOS_LABEL]).reshape(1,-1)

    def shuffle(self):
        if not hasattr(self, 'm'): 
            return
        j = np.random.permutation(self.m)
        i = np.arange(self.n)
        i[1:self.m + 1] = j + 1
        return self.precursor_feat[:, i], self.labels[:, i]

################################################################################################
#
# Dataset class
#
################################################################################################

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self._data = []
        self._train = False
        self.num_labels = NUM_LABEL

    def init_dataset(self, init=False):
        if not init:
            return
        if init and len(self._data) != 0:
            for attr in self._data[0]._feature_attrs:
                delattr(self, f'num_{attr}')
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
            if hasattr(self, f'num_{attr}'):
                n1 = getattr(self._data[0], attr).shape[-1]
                n2 = getattr(self._data[-1], attr.shape[-1])
                if n1 != n2:
                    raise ValueError('Inconsistent data dimension detected', f'num_{attr}')
            else:
                n = getattr(self._data[0], attr).shape[-1]
                setattr(self, f'num_{attr}', n)

    def get_info(self):
        for k, v in self.__dict__.items():
            if k == '_data':
                print(f'{k} : {len(self._data)}' )
            else:
                print(f'{k} : {v}')

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

        self.init_dataset(not extend_dataset)
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
#        if 'graph' in data_type.lower():
#            self._data_type = 'graph'
#        elif 'seq' in data_type.lower():
#            self._data_type = 'sequence'
#        else:
#            self._data_type = 'reaction'
        self._data_type = 'sequence'

        self._train = False
        self._data = []

    def from_file(self, data_path, extend_dataset=False, 
                  target_comp_key='target_comp', precursor_comp_key='precursor_comp', 
                  heat_temp_key=None, heat_time_key=None, info_attrs=[], 
                  *args, **kwargs):
        dataset = self.read_file(data_path)
        self.from_dataset(dataset, extend_dataset = extend_dataset, 
                          target_comp_key = target_comp_key, 
                          precursor_comp_key = precursor_comp_key,
                          heat_temp_key = heat_temp_key, 
                          heat_time_key = heat_time_key, 
                          info_attrs = info_attrs,
                          *args, **kwargs)
    
    def from_dataset(self, dataset:List[Dict], extend_dataset=False, 
                     target_comp_key='target_comp', precursor_comp_key='precursor_comp', 
                     heat_temp_key=None, heat_time_key=None, info_attrs=[], 
                     *args, **kwargs):
        self.init_dataset(not extend_dataset)
        for data in dataset:
            self.from_data(data = data,
                           target_comp_key = target_comp_key, 
                           precursor_comp_key = precursor_comp_key,
                           heat_temp_key = heat_temp_key, 
                           heat_time_key = heat_time_key, 
                           info_attrs = info_attrs,
                           *args, **kwargs)
        self.update_info()
        self.to_torch()

    def from_data(self, data, target_comp_key, precursor_comp_key=None, 
                  heat_temp_key=None, heat_time_key=None, info_attrs=[], 
                  *args, **kwargs):
        precursor_comps = []
        weights = None
        if precursor_comp_key is not None and precursor_comp_key in data.keys():
            precursor_comps = data[precursor_comp_key]
            self._train = True
        heat_temp = None
        if heat_temp_key is not None:
            self.heat_temp_info = True
            key, val = heat_temp_key
            heat_temp = data[key][val]
            if heat_temp is None:
                return
        heat_time = None
        if heat_time_key is not None:
            self.heat_time_info = True
            key, val = heat_time_key
            heat_time = data[key][val]
            if heat_time is None:
                return
#        if self._data_type == 'reaction':
#            self._data.append(
#                ReactionData(data = data,
#                            feat_type = self._feat_type,
#                            target_comp = data[target_comp_key], 
#                            precursor_comps = precursor_comps,
#                            conditions = conditions,
#                            condition_values = condition_values,
#                            weights = weights,
#                            info_attrs = info_attrs,
#                            *args, **kwargs)
#            )
#        elif self._data_type == 'graph':
#            self._data.append(
#                GraphData(data = data,
#                          feat_type = self._feat_type,
#                          target_comp = data[target_comp_key], 
#                          precursor_comps = precursor_comps,
#                          conditions = conditions,
#                          condition_values = condition_values,
#                          weights = weights,
#                          info_attrs = info_attrs,
#                          *args, **kwargs)
#            )
#        elif self._data_type == 'sequence':
        if self._train:
            weights = 1.0 / data['count']
        self._data.append(
            SequenceData(data = data,
                        feat_type = self._feat_type,
                        target_comp = data[target_comp_key], 
                        precursor_comps = precursor_comps,
                        heat_temp = heat_temp,
                        heat_time = heat_time,
                        weights = weights,
                        info_attrs = info_attrs,
                        *args, **kwargs)
        )

    def cfn(self, dataset):
        info = []
#        rxn_index = []
#        inp = []
        conditions = []
        label = []
#        for i, data in enumerate(dataset):
        for data in dataset:
            info.append(data.to_dict())
#            rxn_index.append([i] * data.n)
            _prec_feat, _labels = data.shuffle()
#                inp.append(_prec_feat)
            label.append(_labels)
            conditions.append(data.condition_feat)
#        rxn_index = np.hstack(rxn_index)
#        inps = torch.concat(inp).float()
        label = torch.concat(label).long()
        context = torch.concat(conditions).float()

        if self._train:
            weight = torch.concat([d.weights for d in dataset]).float()

#        if hasattr(self, 'metal_feat'):
#            metal_info = torch.concat([data.metal_feat for data in dataset])

        # final output
#        if self._data_type == 'reaction':
#            return {
#                'inp':inp,
#                'label':label,
#                'condition':condition,
#                'rxn_index':rxn_index,
#            }, info
#        elif self._data_type == 'graph':
#            ns = 0
#            edge_attr = []
#            edge_index = []
#            for data in dataset:
#                edge_attr.append(data.edge_attr)
#                edge_index.append(data.edge_index + ns)
#                ns += data.n
#            edge_attr = torch.concat(edge_attr).float()
#            edge_index = torch.concat(edge_index, dim=-1).long()
#            return {
#                'inp':inp,
#                'label':label,
#                'x':condition,
#                'rxn_index':rxn_index,
#                'edge_attr':edge_attr,
#                'edge_index':edge_index,
#            }, info
#        elif self._data_type == 'sequence':
        l_seq = dataset[0].n
        return {
#            'inps': inps if isinstance(inps, torch.Tensor) else inps,
            'target': label[:, :l_seq-1],
            'label': label[:, 1:],
            'context': context,
            'weight': weight.repeat(1, l_seq - 1).view(-1) if self._train else None,
            'mask': (label[:, 1:] != EOS_LABEL).view(-1) if self._train else None,
        }, info