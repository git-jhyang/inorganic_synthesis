import torch, gzip, pickle, json
import numpy as np
from .utils import MetalElements, composit_parser
from .feature import composition_to_feature
from typing import Dict, List

EOS_LABEL = None
SOS_LABEL = None

class BaseData:
    def __init__(self, 
                 data : Dict = {}, 
                 base_info_attrs : List[str] = ['id','id_urxn','count','doi','year','year_doc'],
                 info_attrs : List[str] = [],
                 *args, **kwargs):        
        self._info_attrs = []
        self._feature_attrs = []
#        self.device = None
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
#        self.device = None
    
    def to_torch(self):
        for attr in self._feature_attrs:
            data = getattr(self, attr)
            if isinstance(data, np.ndarray):
                setattr(self, attr, torch.from_numpy(data))            
#        self.device = 'cpu'
    
#    def to(self, device='cpu'):
#        if len(self._feature_attrs) == 0:
#            return
#        data = getattr(self, self._feature_attrs[0])
#        if self.device is None:
#            self.to_torch()
#        for attr in self._feature_attrs:
#            data = getattr(self, attr)
#            setattr(self, attr, data.to(device))
#        self.device = device

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
                 precursor_refs = None,
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
                label, feat = precursor_refs.get_precursor_data(precursor_comp)
                precursor_feat.append(feat)
                labels.append(label)
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
                 precursor_refs = None,
                 heat_temp : float = None,
                 heat_time : float = None,
                 weights : float = None,
                 *args, **kwargs):
        
        super().__init__(data = data,
                         feat_type = feat_type,
                         target_comp = target_comp,
                         precursor_comps = precursor_comps,
                         precursor_refs = precursor_refs,
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
                 precursor_refs = None,
                 heat_temp : float = None,
                 heat_time : float = None,
                 max_length : int = 8,
                 weights : float = None,
                 *args, **kwargs):
        super().__init__(data = data,
                         feat_type = feat_type,
                         target_comp = target_comp,
                         precursor_comps = precursor_comps,
                         precursor_refs = precursor_refs,
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
            EOS, EOS_VEC = precursor_refs.get_precursor_data('EOS')
            SOS, SOS_VEC = precursor_refs.get_precursor_data('SOS')
                
            self.m = self.labels.shape[0]
            self.precursor_feat = np.vstack([
                SOS_VEC, self.precursor_feat, np.repeat(EOS_VEC, max_length, axis=0)
            ])[:max_length].astype(np.float32)[np.newaxis, ...]
            self.labels = np.hstack([
                [SOS], self.labels.reshape(-1), [EOS] * max_length
            ])[:max_length].astype(int).reshape(1,-1)
        else:
            self._feature_attrs.append('precursor_feat')
            self._feature_attrs.append('labels')
            self.precursor_feat = pad.reshape(1, 1, -1)
            self.labels = np.array([precursor_refs.SOS_LABEL]).reshape(1,-1)

    def shuffle(self):
        if not hasattr(self, 'm'): 
            return
        j = np.random.permutation(self.m)
        i = np.arange(self.n)
        i[1:self.m + 1] = j + 1
        return self.precursor_feat[:, i], self.labels[:, i]



################################################################################################
################################################################################################
#
#   Precursor dataset class
#
################################################################################################
################################################################################################



class PrecursorDataset:
    def __init__(self, 
                 feat_type:str = 'composit',
                 by_fraction:bool = True,
                 norm:bool = True
                 ):
        self._feat_type = feat_type
        self._by_fraction = by_fraction
        self._norm = norm

#        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/screened_precursor.pkl.gz')
        path = '../data/screened_precursor.pkl.gz'

        with gzip.open(path, 'rb') as f:
            self.precursor_source = pickle.load(f)
        self.active_precursors = [p['precursor_str'] for p in self.precursor_source]
        self.parse_labels()

    def parse_labels(self, active_precursors=None):
        global EOS_LABEL, SOS_LABEL
        if isinstance(active_precursors, (list, np.ndarray)):
            self.active_precursors = active_precursors

        self.label_to_precursor = []
        self.label_to_source = []
        self.precursor_to_label = {}
        self.source_to_label = -np.ones(len(self.precursor_source) + 2, dtype=int)
        self.embedding = []
        for i, prec in enumerate(self.precursor_source):
            pstr = prec['precursor_str']
            if pstr not in self.active_precursors:
                continue
            j = len(self.label_to_precursor)
            self.label_to_precursor.append(pstr)
            self.label_to_source.append(i)
            self.precursor_to_label.update({pstr:j})
            self.source_to_label[i] = j
            self.embedding.append(composition_to_feature(composit_dict = prec['precursor_comp'], 
                                                         feature_type = self._feat_type,
                                                         by_fraction = self._by_fraction,
                                                         norm = self._norm))

        self.NUM_LABEL = len(self.label_to_precursor) + 2
        i = len(self.precursor_source)
        j = len(self.label_to_precursor)

        EOS_LABEL = j
        self.label_to_precursor.append('EOS')
        self.label_to_source.append(i)
        self.precursor_to_label.update({'EOS':j})
        self.source_to_label[i] = j
        self.embedding.append(np.zeros_like(self.embedding[0]))

        SOS_LABEL = j + 1
        self.label_to_precursor.append('SOS')
        self.label_to_source.append(i + 1)
        self.precursor_to_label.update({'SOS':j + 1})
        self.source_to_label[i + 1] = j + 1
        self.embedding.append(np.ones_like(self.embedding[0]))

        self.label_to_precursor = np.array(self.label_to_precursor)
        self.label_to_source = np.array(self.label_to_source)
        self.embedding = np.vstack(self.embedding)
        
    def update_labels(self, dataset, prec_comp_attr='precursor_comps'):
        prec_comps = []
        for data in dataset:
            if not hasattr(data, prec_comp_attr):
                continue
            for p in getattr(data, prec_comp_attr):
                prec_comp = composit_parser(p)
                if prec_comp in prec_comps:
                    continue
                else:
                    prec_comps.append(prec_comp)
        self.parse_labels(prec_comps)

    def save(self, path):
        if isinstance(self.active_precursors, np.ndarray):
            self.active_precursors = self.active_precursors.tolist()
        info = {
            'feat_type': self._feat_type,
            'by_fraction': self._by_fraction,
            'norm': self._norm,
            'active_precursors': self.active_precursors,
        }
        with open(path, 'w') as f:
            json.dump(info, f, indent=4)
    
    def load(self, path):
        with open(path, 'r') as f:
            info = json.load(f)
        self.__init__(feat_type = info['feat_type'],
                      by_fraction = info['by_fraction'],
                      norm = info['norm'])
        self.parse_labels(info['active_precursors'])

    def _check_valid_precursor(self, precursor):
        if isinstance(precursor, int) and precursor < self.NUM_LABEL:
            i = precursor
        elif isinstance(precursor, str) and precursor in self.active_precursors:
            i = self.precursor_to_label[precursor]
        elif isinstance(precursor, dict) and composit_parser(precursor) in self.active_precursors:
            i = self.precursor_to_label[composit_parser(precursor)]
        else:
            print("Unknown precursor: ", precursor)
            return None
        return i

    def get_precursor_data(self, precursor):
        i = self._check_valid_precursor(precursor)
        if i is None:
            return None, None
        return i, self.embedding[i]

    def get_precursor_info(self, precursor):
        i = self._check_valid_precursor(precursor)
        if i is None:
            return None, None
        j = self.label_to_source[i]
        return i, self.precursor_source[j]

    def to_dict(self):
        return {
            'feature_type': self._feat_type, 
            'EOS_LABEL': EOS_LABEL, 
            'SOS_LABEL': SOS_LABEL, 
            'NUM_LABEL': self.NUM_LABEL,
        }



################################################################################################
################################################################################################
#
#   Reaction dataset class
#
################################################################################################
################################################################################################



class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self._data = []
        self._train = False

    def init_dataset(self):
        if len(self._data) != 0:
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
    
#    def to(self, device):
#        for data in self._data:
#            data.to(device)

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
        if not extend_dataset:
            self.init_dataset()
        dataset = self.read_file(data_path)

        for data in dataset:
            comp_data = CompositionData(data=data, 
                                        comp=data[compsition_key], 
                                        comp_feat_type=self._comp_feat_type, 
                                        by_fraction=self._by_fraction)
            self._data.append(comp_data)
        self._year = np.array([d.year for d in self._data])
        self.to_torch()

    def cfn(self, dataset):
        feat = []
        info = []
        for data in dataset:
            feat.append(data.comp_feat)
            info.append(data.to_dict())
        return torch.vstack(feat), info
    
class ReactionDataset(BaseDataset):
    def __init__(self, 
                 feat_type:str = 'composit', 
#                 data_type:str = 'sequence', 
                 shuffle_sequence:bool = True, 
                 sequence_length:int = 8,
                 include_eos:int = 0,
                 weights:bool = True):
        super().__init__()
        self.precursor_dataset = PrecursorDataset(feat_type=feat_type,
                                                  by_fraction = True,
                                                  norm = True)
        self.has_temp_info = False
        self.has_time_info = False

        self._feat_type = feat_type
        self._include_eos = include_eos if include_eos in [0,1] else -1
        self._sequence_length = sequence_length
        self._shuffle_sequence = shuffle_sequence
        self._weights = weights
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
        if not extend_dataset:
            self.init_dataset()
            active_precs = []
            for data in dataset:
                for prec in data['precursor_comp']:
                    if prec in active_precs:
                        continue
                    active_precs.append(prec)
            self.precursor_dataset.update(active_precs)
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
        weights = 1
        if self._train and self._weights:
            weights = 1.0 / data['count']
        self._data.append(
            SequenceData(data = data,
                         feat_type = self._feat_type,
                         target_comp = data[target_comp_key], 
                         precursor_comps = precursor_comps,
                         precursor_refs = self.precursor_dataset,
                         heat_temp = heat_temp,
                         heat_time = heat_time,
                         weights = weights,
                         info_attrs = info_attrs,
                         max_length = self._sequence_length,
                         *args, **kwargs)
        )

    def cfn(self, dataset):
        info = []
        conditions = []
        l_seq = dataset[0].n
        for data in dataset:
            info.append(data.to_dict())
            conditions.append(data.condition_feat)
        
        if self._shuffle_sequence:
            labels = torch.concat([d.shuffle()[1] for d in dataset]).long()
        else:
            labels = torch.concat([d.labels for d in dataset]).long()

        target = labels[:, :l_seq-1]
        label = labels[:, 1:].reshape(-1)

        if self._include_eos == 0:
            mask = (label != EOS_LABEL)
        elif self._include_eos == 1:
            mask = target.reshape(-1) != EOS_LABEL
        else:
            mask = torch.ones_like(label).bool()

        if self._train:
            weights = torch.concat([d.weights for d in dataset]).float().repeat(1, l_seq - 1).view(-1)
        else:
            weights = None

        return {
            'target': target,
            'label': label,
            'context': torch.concat(conditions).float(),
            'weight': weights,
            'mask': mask,
        }, info
    
#    def _cfn(self, dataset):
#        rxn_index = []
#        inp = []
#        for i, data in enumerate(dataset):
#            rxn_index.append([i] * data.n)
#            inp.append(_prec_feat)
#        rxn_index = np.hstack(rxn_index)
#        inps = torch.concat(inp).float()
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
#        return
    