import torch, gzip, pickle, json, abc
import numpy as np
from .utils import MetalElements, composit_parser
from .feature import (PrecursorReference,
                      PrecursorSequenceReference, 
                      LigandTemplateReference, 
                      composition_to_feature)
from typing import Dict, List

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

# class CompositionData(BaseData):
#     def __init__(self, 
#                  comp : Dict, 
#                  data : Dict = {}, 
#                  feat_type : str = 'composit', 
#                  label : int = None,
#                  weight : float = 1.0,
#                  *args, **kwargs):
#         super().__init__(data, *args, **kwargs)

#         self._info_attrs.extend(['comp', 'feat_type'])
#         self._feature_attrs.extend(['feat', 'weight'])

#         self.feat_type = feat_type
#         self.comp = comp
#         self.feat = composition_to_feature(comp, feature_type=feat_type, by_fraction=True)
#         self.weight = np.array(weight, dtype=np.float32).reshape(-1,1)

#         if label is not None:
#             self.label = np.array(label, dtype=int).reshape(-1,1)
#             self._feature_attrs.append('label')

#         self.to_torch()

# changed for multilabels
class ReactionData(BaseData):
    def __init__(self, 
                 data : Dict = {},
                 feat_type : str = 'composit',
                 target_comp : Dict = {},
                 precursor_comps : List[Dict] = [],
                 precursor_ref = None,
                 heat_temp : float = None,
                 heat_time : float = None,
                 weights : float = 1.0,
                 *args, **kwargs):
        super().__init__(data, *args, **kwargs)

        # meta info
        self.data_type = 'basic'
        self.feat_type = feat_type
        self._info_attrs.extend(['data_type', 'feat_type'])

        # weights
        self.weight = np.array(weights, dtype=np.float32).reshape(-1,1)
        self._feature_attrs.append('weight')

        # target
        self.target_comp = target_comp
        self.target_feat = composition_to_feature(composit_dict=target_comp, 
                                                  feature_type=feat_type, 
                                                  by_fraction=True)
        self._info_attrs.append('target_comp')
        self._feature_attrs.append('target_feat')

        # metal
        metals = []
        label_mask = []
        meta_feat = []
        non_metals = {}
        for ele in self.target_comp.keys():
            if ele in MetalElements:
                metals.append(ele)
                label_mask.append(precursor_ref.get_mask(ele))
                meta_feat.append(composition_to_feature({ele:1}, feat_type, by_fraction=False))
            else:
                non_metals.update({ele:1})
        label_mask.append(precursor_ref.get_mask('none'))
        meta_feat.append(composition_to_feature(non_metals, feat_type, by_fraction=False))
        self.meta_feat = np.vstack(meta_feat)
        self.label_mask = np.vstack(label_mask)
        self.n = self.meta_feat.shape[0]
        self._feature_attrs.extend(['meta_feat','label_mask'])

        # label and precursor
        if isinstance(precursor_comps, List) and len(precursor_comps) != 0:
            self.precursor_comps = precursor_comps
            self.label = np.zeros((len(meta_feat), precursor_ref.NUM_LABEL), dtype=np.float32)
            precursor_feat = np.repeat(precursor_ref.get_embedding({}), len(meta_feat), axis=0)
            for precursor_comp in precursor_comps:
                metal = [e for e in precursor_comp.keys() if e in MetalElements]
                if len(metal) != 0:
                    i = metals.index(metal[0])
                else:
                    i = len(metals)
                j = precursor_ref.to_label(precursor_comp)
                self.label[i, j] = 1
                precursor_feat[i] += composition_to_feature(precursor_comp, feat_type, by_fraction=True).reshape(-1)
            self.precursor_feat = np.vstack(precursor_feat)
            self._info_attrs.append('precursor_comps')
            self._feature_attrs.extend(['label','precursor_feat'])

        # conditions
        if (heat_temp or heat_time) is not None:
            self.condition_feat = np.zeros((1,0))
            self._feature_attrs.append('condition_feat')
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
                 precursor_ref = None,
                 heat_temp : float = None,
                 heat_time : float = None,
                 weights : float = 1.0,
                 *args, **kwargs):
        
        super().__init__(data = data,
                         feat_type = feat_type,
                         target_comp = target_comp,
                         precursor_comps = precursor_comps,
                         precursor_ref = precursor_ref,
                         heat_temp = heat_temp,
                         heat_time = heat_time,
                         weights = weights,
                         *args, **kwargs)

        # graph 
        self._feature_attrs.extend(['edge_feat','edge_index'])
        edge_index = []
        edge_feat = []
        for i, meta_i in enumerate(self.meta_feat):
            for j in range(self.n):
                edge_index.append([i,j])
                if i == j:
                    edge_feat.append(self.target_feat)
                else:
                    edge_feat.append(meta_i)
        self.edge_index = np.array(edge_index, dtype=int).T
        self.edge_feat = np.vstack(edge_feat, dtype=np.float32)
        
        delattr(self, 'target_feat')
        self._feature_attrs.pop(self._feature_attrs.index('target_feat'))

class SequenceData(ReactionData):
    def __init__(self, 
                 data : Dict = {},
                 feat_type : str = 'composit',
                 target_comp : Dict = {},
                 precursor_comps : List[Dict] = [],
                 precursor_ref = None,
                 heat_temp : float = None,
                 heat_time : float = None,
                 max_length : int = 8,
                 weights : float = 1.0,
                 *args, **kwargs):
        super().__init__(data = data,
                         feat_type = feat_type,
                         target_comp = target_comp,
                         precursor_comps = precursor_comps,
                         precursor_ref = precursor_ref,
                         heat_temp = heat_temp,
                         heat_time = heat_time,
                         weights = weights,
                         *args, **kwargs)
        
        self._feature_attrs.pop(self._feature_attrs.index('metal_feat'))
        delattr(self, 'metal_feat')
        self.n = max_length

        # labels & precursor feat
        EOS, EOS_VEC = precursor_ref.get_embedding('EOS')
        SOS, SOS_VEC = precursor_ref.get_embedding('SOS')

        self._feature_attrs.append('sequence_mask')
        sequence_mask = np.zeros((max_length), dtype=bool)
        if hasattr(self, 'precursor_feat'): 
            self.m = self.labels.shape[0]
            self.precursor_feat = np.vstack([
                SOS_VEC.reshape(1,-1), self.precursor_feat, np.repeat(EOS_VEC.reshape(1,-1), max_length, axis=0)
            ])[:max_length].astype(np.float32)[np.newaxis, ...]
            self.labels = np.hstack([
                [SOS], self.labels.reshape(-1), [EOS] * max_length
            ])[:max_length].astype(int).reshape(1,-1)
            sequence_mask[:self.m+1] = True
        else:
            self._feature_attrs.append('precursor_feat')
            self._feature_attrs.append('labels')
            self.precursor_feat = SOS_VEC.reshape(1, 1, -1)
            self.labels = np.array([SOS]).reshape(1,-1)
        self.sequence_mask = sequence_mask.reshape(1,-1)

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
#   Reaction dataset class
#
################################################################################################
################################################################################################

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self._data = []
        self._train = False
        self.has_temp_info = False
        self.has_time_info = False

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

    @property
    def NUM_LABEL(self):
        return self.precursor_dataset.NUM_LABEL

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
                for prec in data[precursor_comp_key]:
                    pstr = composit_parser(prec)
                    if pstr in active_precs:
                        continue
                    active_precs.append(pstr)
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

    def parsing_data(self, data, precursor_comp_key=None, heat_temp_key=None, heat_time_key=None):
        precursor_comps = []
        if (precursor_comp_key is not None) and (precursor_comp_key in data.keys()):
            precursor_comps = data[precursor_comp_key]
            self._train = True
        heat_temp = None
        if heat_temp_key is not None:
            self.has_temp_info = True
            key, val = heat_temp_key
            heat_temp = data[key][val]
        heat_time = None
        if heat_time_key is not None:
            self.has_time_info = True
            key, val = heat_time_key
            heat_time = data[key][val]
        return precursor_comps, heat_temp, heat_time

    @abc.abstractmethod
    def from_data(self, data, target_comp_key, precursor_comp_key=None, 
                  heat_temp_key=None, heat_time_key=None, info_attrs=[], 
                  *args, **kwargs):
        '''
        data object
        '''
        pass

    @abc.abstractmethod
    def cfn(self, data):
        '''
        collate function
        '''
        pass

# class CompositionDataset(BaseDataset):
#     def __init__(self, comp_feat_type='composit', by_fraction=True):
#         super().__init__()
#         self._comp_feat_type = comp_feat_type
#         self._by_fraction = by_fraction

#     def from_file(self, 
#                   data_path='../data/unique_target.pkl.gz', 
#                   compsition_key='target_comp',
#                   extend_dataset=False):
#         if not extend_dataset:
#             self.init_dataset()
#         dataset = self.read_file(data_path)

#         for data in dataset:
#             comp_data = CompositionData(data=data, 
#                                         comp=data[compsition_key], 
#                                         comp_feat_type=self._comp_feat_type, 
#                                         by_fraction=self._by_fraction)
#             self._data.append(comp_data)
#         self._year = np.array([d.year for d in self._data])
#         self.to_torch()

#     def cfn(self, dataset):
#         feat = []
#         info = []
#         for data in dataset:
#             feat.append(data.comp_feat)
#             info.append(data.to_dict())
#         return torch.vstack(feat), info
    
class ReactionDataset(BaseDataset):
    def __init__(self, feat_type:str = 'composit',
                 *args, **kwargs):
        super().__init__()
        self._feat_type = feat_type
        self.precursor_dataset = PrecursorReference(feat_type=feat_type, 
                                                    by_fraction = True, 
                                                    *args, **kwargs)

    def from_data(self, data, target_comp_key, precursor_comp_key=None, 
                  heat_temp_key=None, heat_time_key=None, info_attrs=[], 
                  *args, **kwargs):
        weights = 1.0
        precursor_comps, heat_temp, heat_time = self.parsing_data(data, precursor_comp_key, heat_temp_key, heat_time_key)
        if len(precursor_comps) != 0:
            self._train = True
        if (heat_temp_key is not None) and (heat_temp is None):
            return
        if (heat_time_key is not None) and (heat_time is None):
            return
        self._data.append(
            ReactionData(data = data,
                        feat_type = self._feat_type,
                        target_comp = data[target_comp_key], 
                        precursor_comps = precursor_comps,
                        precursor_ref = self.precursor_dataset,
                        heat_temp = heat_temp,
                        heat_time = heat_time,
                        weights = weights,
                        info_attrs = info_attrs,
                        *args, **kwargs)
        )

    def cfn(self, dataset):
        info = []
        rxn_id = []
        meta_feat = []

        for i, data in enumerate(dataset):
            info.append(data.to_dict())
            rxn_id.append([i] * data.n)
            meta_feat.append(data.target_feat)
        
        rxn_id = np.hstack(rxn_id).astype(int)
        meta_feat = torch.vstack(meta_feat).float()

        if self.has_temp_info or self.has_time_info:
            condition_feat = [data.condition_feat for data in dataset]
            meta_feat = torch.hstack([torch.vstack(condition_feat), meta_feat]).float()

        prec_feat = []
        label = []
        label_mask = []
        label_mask_ = []
        if self._train:
            for data in dataset:
                prec_feat.append(data.precursor_feat.sum(0))
                label.append(data.label.sum(0))
                label_mask.append(data.label_mask.sum(0).bool())
                label_mask_.append(data.label_mask)
            prec_feat = torch.vstack(prec_feat)
            label = torch.vstack(label)
            label_mask = torch.vstack(label_mask)
            label_mask_ = torch.vstack(label_mask_).numpy()

        return {
            'rxn_id' : rxn_id,
            'condition' : meta_feat,
            'x' : prec_feat,
            'label' : label,
            'label_mask' : label_mask,
            'label_mask_orig' : label_mask_
        }, info

#########################################################################################################

class ReactionGraphDataset(BaseDataset):
    def __init__(self, feat_type:str = 'composit',
                 *args, **kwargs):
        super().__init__()
        self._feat_type = feat_type
        self.precursor_dataset = LigandTemplateReference(feat_type=feat_type, 
                                                         by_fraction = True, 
                                                         *args, **kwargs)

    def from_data(self, data, target_comp_key, precursor_comp_key=None, 
                  heat_temp_key=None, heat_time_key=None, info_attrs=[], 
                  *args, **kwargs):
        weights = 1.0
        precursor_comps, heat_temp, heat_time = self.parsing_data(data, precursor_comp_key, heat_temp_key, heat_time_key)
        if len(precursor_comps) != 0:
            self._train = True
        if (heat_temp_key is not None) and (heat_temp is None):
            return
        if (heat_time_key is not None) and (heat_time is None):
            return
        self._data.append(
            GraphData(data = data,
                        feat_type = self._feat_type,
                        target_comp = data[target_comp_key], 
                        precursor_comps = precursor_comps,
                        precursor_ref = self.precursor_dataset,
                        heat_temp = heat_temp,
                        heat_time = heat_time,
                        weights = weights,
                        info_attrs = info_attrs,
                        *args, **kwargs)
        )

    def cfn(self, dataset):
        info = []
        rxn_id = []
        meta_feat = []
        edge_feat = []
        edge_index = []
        n = 0
        for i, data in enumerate(dataset):
            info.append(data.to_dict())
            rxn_id.append([i] * data.n)
            meta_feat.append(data.meta_feat)
            edge_feat.append(data.edge_feat)
            edge_index.append(data.edge_index + n)
            n += data.n
        rxn_id = np.hstack(rxn_id).astype(int)
        meta_feat = torch.vstack(meta_feat).float()
        edge_feat = torch.vstack(edge_feat).float()
        edge_index = torch.hstack(edge_index).long()

        if self.has_temp_info or self.has_time_info:
            condition_feat = [data.condition_feat.repeat(data.n, 1) for data in dataset]
            meta_feat = torch.hstack([torch.vstack(condition_feat), meta_feat]).float()

        prec_feat = []
        label = []
        label_mask = []
        weight = []
        if self._train:
            for data in dataset:
                prec_feat.append(data.precursor_feat)
                label.append(data.label)
                label_mask.append(data.label_mask)
                weight.append(data.weight.repeat(data.label.shape))
            prec_feat = torch.vstack(prec_feat)
            label = torch.vstack(label)
            label_mask = torch.vstack(label_mask)
            weight = torch.vstack(weight)

        return {
            'rxn_id' : rxn_id,
            'condition' : meta_feat,
            'edge_attr' : edge_feat,
            'edge_index' : edge_index,
#            'condition_feat' : condition_feat,
            'x' : prec_feat,
            'label' : label,
            'label_mask' : label_mask,
            'weight' : weight,
        }, info

#########################################################################################################

class SequenceDataset(BaseDataset):
    def __init__(self, 
                 feat_type:str = 'composit', 
                 shuffle_sequence:bool = True, 
                 sequence_length:int = 8,
                 include_eos:int = 0,
                 *args, **kwargs):
        super().__init__()
        self.precursor_dataset = PrecursorSequenceReference(feat_type=feat_type,
                                                            by_fraction = True,
                                                            norm = True,
                                                            *args, **kwargs)

        self._feat_type = feat_type
        self._include_eos = include_eos if include_eos in [0,1] else -1
        self._sequence_length = sequence_length
        self._shuffle_sequence = shuffle_sequence

    @property
    def EOS_LABEL(self):
        return self.precursor_dataset.EOS_LABEL

    @property
    def SOS_LABEL(self):
        return self.precursor_dataset.SOS_LABEL

    def get_embedding(self, x):
        embed = self.precursor_dataset.get_embedding(x)
        return torch.from_numpy(embed).float()

    def from_data(self, data, target_comp_key, precursor_comp_key=None, 
                  heat_temp_key=None, heat_time_key=None, info_attrs=[], 
                  *args, **kwargs):
        precursor_comps, heat_temp, heat_time = self.parsing_data(data, precursor_comp_key, heat_temp_key, heat_time_key)
        if len(precursor_comps) != 0:
            self._train = True
        if (heat_temp_key is not None) and (heat_temp is None):
            return
        if (heat_time_key is not None) and (heat_time is None):
            return
        weights = 1.0 / data['count']
        self._data.append(
            SequenceData(data = data,
                         feat_type = self._feat_type,
                         target_comp = data[target_comp_key], 
                         precursor_comps = precursor_comps,
                         precursor_ref = self.precursor_dataset,
                         heat_temp = heat_temp,
                         heat_time = heat_time,
                         weights = weights,
                         info_attrs = info_attrs,
                         max_length = self._sequence_length,
                         *args, **kwargs)
        )

    def cfn(self, dataset):
        info = []
        labels = []
        prec_feats = []
        conditions = []
        weights = []
        sequence_mask = []
        precursor_mask = []

        N = dataset[0].n
        for data in dataset:
            info.append(data.to_dict())
            conditions.append(data.condition_feat)
            sequence_mask.append(data.sequence_mask)
            precursor_mask.append(data.precursor_mask)
            weights.append(data.weights)

        if self._shuffle_sequence:
            for data in dataset:
                prec_feat, label = data.shuffle()
                labels.append(label)
                prec_feats.append(prec_feat)
        else:
            for data in dataset:
                labels.append(data.labels)
                prec_feats.append(data.precursor_feat)

        labels = torch.concat(labels).long()
        prec_feats = torch.concat(prec_feats).float()[:, :N-1]
        conditions = torch.concat(conditions).float()
        weights = torch.concat(weights).reshape(-1,1).repeat(1, N-1).reshape(-1).float()
        # evade overfitting to EOS
        sequence_mask = torch.concat(sequence_mask).bool()
        # for fast convergence
        precursor_mask = torch.concat(precursor_mask).bool().unsqueeze(1)

#        target = labels[:, :N-1]
        label = labels[:, 1:].reshape(-1)
        neg_label = torch.concat([labels[:, -1:], labels[:, 1:-1]], -1).reshape(-1)

        if self._include_eos == 0:
            sequence_mask = sequence_mask[:, 1:].reshape(-1)
        elif self._include_eos == 1:
            sequence_mask = sequence_mask[:, :N-1].reshape(-1)
        else:
            sequence_mask = torch.ones_like(label).bool()

        return {
#            'target': target,
            'x': prec_feats,
            'label': label,
            'neg_label': neg_label,
            'context': conditions,
            'weight': weights,
            'sequence_mask': sequence_mask,
            'precursor_mask': precursor_mask,
        }, info