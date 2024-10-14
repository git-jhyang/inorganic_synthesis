from .utils import ActiveElements, AllElements, NEAR_ZERO, MetalElements, composit_parser
import numpy as np
import json, os, gzip, pickle, numbers, abc, dill

ATOM_EXCEPTION_WARNING = 'Warning: element [{}] is not included in feature type "{}"\n'
FEAT_EXCEPTION_WARNING = 'feature type [{}] is not supported. '
EXCEPTION_ELEMENTS = []

elmd = {}
for k in ['cgcnn','elemnet','magpie_sc','mat2vec','matscholar','megnet16','oliynyk_sc']:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elmd', f'{k}.json')) as f:
        elmd_data = json.load(f)
    elmd[k] = elmd_data

def composition_to_feature(composit_dict, 
                           feature_type='composit', 
                           dtype=np.float32, 
                           by_fraction=True,
                           norm=True,
                           *args, **kwargs):
    '''
    composit_dict = {
        element_0 (str): fraction_0 (float),
        element_1 (str): fraction_1 (float),
        ...
    }
    '''
    global ATOM_EXCEPTION_WARNING, FEAT_EXCEPTION_WARNING
    div = 1
    if norm and len(composit_dict) != 0:
        by_fraction = True
        div = 1.0 / np.sum(list(composit_dict.values()))

    if feature_type in elmd.keys():
        vec = [np.zeros_like(elmd[feature_type]['Li'])]
        for ele, frac in composit_dict.items():
            if ele in elmd[feature_type].keys():
                vec.append(np.array(elmd[feature_type][ele]) * (frac if by_fraction else 1) * div)
            elif ele not in EXCEPTION_ELEMENTS:
                print(ATOM_EXCEPTION_WARNING.format(ele, feature_type))
                EXCEPTION_ELEMENTS.append(ele)
        return np.sum(vec, 0).astype(dtype).reshape(1,-1)
    elif feature_type.startswith('comp'):
        if 'ext' in feature_type:
            return active_composit_feature(composit_dict, ref=AllElements, dtype=dtype, by_fraction=by_fraction).reshape(1,-1) * div
        return active_composit_feature(composit_dict, dtype=dtype, by_fraction=by_fraction).reshape(1,-1) * div
    else:
        raise ValueError(FEAT_EXCEPTION_WARNING.format(feature_type), 
                         'Possible feature types:', ['composit','composit_ext'] + list(elmd.keys()),
                         'Feature type is set to composit')

def active_composit_feature(composit_dict, ref=ActiveElements, dtype=float, by_fraction=True, *args, **kwargs):
    feat_vec = np.zeros(len(ref), dtype=dtype)
    for ele, frac in composit_dict.items():
        if frac <= NEAR_ZERO: continue
        if ele in ref:
            feat_vec[ref.index(ele)] = frac if by_fraction else 1
        elif ele not in EXCEPTION_ELEMENTS:
            print(ATOM_EXCEPTION_WARNING.format(ele, 'composit'))
            EXCEPTION_ELEMENTS.append(ele)
    return feat_vec

# def metal_composit_feature(composit_dict, dtype=float, by_fraction=True, *args, **kwargs):
#     feat_vec = np.zeros(len(MetalElements) + 1, dtype=dtype)
#     for ele, frac in composit_dict.items():
#         feat_vec[MetalElements.index(ele) + 1] = frac if by_fraction else 1
#     if feat_vec.sum() == 0:
#         feat_vec[0] = 1
#     return feat_vec

# def ligand_composit_feature(composit_dict, dtype=float, by_fraction=True, *args, **kwargs):
#     feat_vec = np.zeros(len(LigandElements) + 1, dtype=dtype)
#     for ele, frac in composit_dict.items():
#         if ele in MetalElements:
#             feat_vec[0] += frac if by_fraction else 1
#         else:
#             feat_vec[LigandElements.index(ele) + 1] = frac if by_fraction else 1
#     return feat_vec

# def feature_to_composit(feat_vec, tol=1e-5):
#     n_feat = feat_vec.shape[-1]
#     feat_vec = feat_vec.reshape(-1, n_feat)
#     if n_feat not in [97, 87, 12]:
#         raise TypeError(f'feature type is not supported', composit_fnc)
#     if n_feat == 97: # active_composit
#         ref = np.array(ActiveElements)
#     elif n_feat == 87: # metal_composit
#         ref = np.array(['None'] + MetalElements)
#     elif n_feat == 12: # ligand_composit
#         ref = np.array(['Metal'] + LigandElements)
#     out = []
#     for vec, mask in zip(feat_vec, feat_vec > tol):
#         out.append(
#             {e:f for e,f in zip(ref[mask], vec[mask])}
#         )
#     return out

# def parse_feature(feat_vec, csim_cut = 0.5, sser_cut = 1.0, to_string=False):
#     out = []
#     if feat_vec.shape[-1] == 12:
#         ref = ligand_vector
#         chrs = np.array(['-'.join(k) for k in ligand_list[:-1]] + ['Unknown'])
#     elif feat_vec.shape[-1] == 97:
#         ref = precursor_vector
#         chrs = np.array(['-'.join(k) for k in precursor_list[:-1]] + ['Unknown'])
#     else:
#         raise ValueError('Not supported feature type')

#     for idx, sser, csim in zip(*find_nearest(feat_vec, ref)):
#         if (sser > sser_cut) or (csim < csim_cut):
#             out.append(-1)
#         else:
#             out.append(int(idx))
#     if to_string:
#         return chrs[out]
#     else:
#         return out

# def check_blacklist(comp):
#     key = tuple(k for k in sorted(comp.keys(), key=lambda x: Element(x).number))
#     if key in blacklist:
#         return True
#     else:
#         return False

################################################################################################
################################################################################################
#
#   Precursor dataset class
#
################################################################################################
################################################################################################

class BaseReference:
    def __init__(self, feat_type, by_fraction, norm=False, 
                 label_weight_fnc = None,
                 ref_fn='unique_precursor.pkl.gz',
                 *args, **kwargs):
        self._feat_type = feat_type
        self._by_fraction = by_fraction
        self._norm = norm
        if label_weight_fnc is None:
            self._label_weight_fnc = lambda x: 1
        else:
            self._label_weight_fnc = label_weight_fnc
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', ref_fn)
#        path = f'../data/{ref_fn}'

        with gzip.open(path, 'rb') as f:
            self._precursor_source = pickle.load(f)
#        self._precursor_source.append({'precursor_str':''})
        self._active_precursors = [p['precursor_str'] for p in self._precursor_source]
        self._precursor_to_source = {c:i for i,c in enumerate(self._active_precursors)}

    def save(self, path, fn='precursor_data.json'):
        info = {
            'feat_type': self._feat_type,
            'by_fraction': self._by_fraction,
            'norm': self._norm,
            'active_precursors': self._active_precursors,
            'label_weight_fnc': dill.dumps(self._label_weight_fnc),
        }
        with open(os.path.join(path, fn), 'w') as f:
            json.dump(info, f, indent=4)
    
    def load(self, path, fn='precursor_data.json'):
        with open(os.path.join(path, fn), 'r') as f:
            info = json.load(f)
        self.__init__(feat_type = info['feat_type'],
                      by_fraction = info['by_fraction'],
                      norm = info['norm'],
                      label_weight_fnc = dill.loads(info['label_weight_fnc']))
        self.update(info['active_precursors'])
        return self

    @abc.abstractmethod
    def update(self):
        pass
    
    @abc.abstractmethod
    def _check_valid_precursor(self):
        pass

    @abc.abstractmethod
    def to_label(self):
        pass

    @abc.abstractmethod
    def get_info(self):
        pass

    @abc.abstractmethod
    def get_weight(self):
        pass

    @abc.abstractmethod
    def to_dict(self):
        pass

#####################################################################################

class PrecursorReference(BaseReference):
    def __init__(self, 
                 feat_type:str = 'composit',
                 by_fraction:bool = True,
                 norm:bool = True,
                 label_weight_fnc = None, 
                 *args, **kwargs):
        super().__init__(feat_type=feat_type, by_fraction=by_fraction, norm=norm, 
                         label_weight_fnc=label_weight_fnc, *args, **kwargs)
        self.update()

    def update(self, active_precursors=None):
        if not isinstance(active_precursors, (list, np.ndarray, tuple, set)):
            active_precursors = self._active_precursors
        _active_precursors = []
        self._source_to_label = - np.ones(len(self._precursor_source), dtype=int)
        self._label_to_source = []
        for i, precursor in enumerate(self._precursor_source):
            precursor_str = precursor['precursor_str']
            if precursor_str not in active_precursors:
                continue
            _active_precursors.append(precursor_str)
            j = len(self._label_to_source)
            self._label_to_source.append(i)
            self._source_to_label[i] = j
        
        self.NUM_LABEL = len(self._label_to_source)
        self._active_precursors = _active_precursors.copy()

        self._weight = np.zeros((len(MetalElements) + 1, self.NUM_LABEL), dtype=float)
        for i,j in enumerate(self._label_to_source):
            for ele in self._precursor_source[j]['precursor_comp'].keys():
                if ele in MetalElements:
                    k = MetalElements.index(ele) + 1
                    self._weight[k, i] = self._label_weight_fnc(self._precursor_source[j])
            if self._weight[:, i].sum() == 0:
                self._weight[0, i] = self._label_weight_fnc(self._precursor_source[j])
        self._label_to_source = np.array(self._label_to_source)

    def _check_valid_precursor(self, precursor, exit=True):
        i_src = None
        if isinstance(precursor, numbers.Integral) and (precursor < self.NUM_LABEL):
            i_src = self._label_to_source[precursor]
        elif isinstance(precursor, str) and (precursor in self._precursor_to_source.keys()):
            i_src = self._precursor_to_source[precursor]
        elif isinstance(precursor, dict):
            precursor_str = composit_parser(precursor)
            if precursor_str in self._precursor_to_source.keys():
                i_src = self._precursor_to_source[precursor_str]
        if (i_src is None) and exit:
            raise ValueError('Invalid precursor', precursor)
        return i_src

    def to_label(self, precursor):
        i_src = self._check_valid_precursor(precursor)
        return self._source_to_label[i_src]

    def get_info(self, precursor):
        i_src = self._check_valid_precursor(precursor)
        return self._precursor_source[i_src]

    def get_weight(self, metal):
        if metal in MetalElements:
            return self._weight[MetalElements.index(metal)+1].reshape(1,-1)
        else:
            return self._weight[0].reshape(1,-1)

    def to_dict(self):
        return {
            'feature_type': self._feat_type, 
            'NUM_LABEL': self.NUM_LABEL,
        }

#####################################################################################

class PrecursorSequenceReference(PrecursorReference):
    def __init__(self, 
                 feat_type:str = 'composit',
                 by_fraction:bool = True,
                 norm:bool = True,
                 label_weight_fnc = None,
                 *args, **kwargs):
        super().__init__(feat_type=feat_type, by_fraction=by_fraction, norm=norm, 
                         label_weight_fnc=label_weight_fnc, *args, **kwargs)
        self._precursor_to_source.update({
            'EOS':len(self._precursor_source),
            'SOS':len(self._precursor_source) + 1,
        })
        self.update()

    def update(self, active_precursors=None):
        super().update(active_precursors=active_precursors)
        self._source_to_label = np.hstack([self._source_to_label, [-1, -1]])

        self.NUM_LABEL = len(self._label_to_source) + 2

        # set EOS
        self.EOS_LABEL = len(self._label_to_source)
        self._source_to_label[self._precursor_to_source['EOS']] = self.EOS_LABEL
        self._weight = np.hstack([self._weight,
                                  np.zeros(len(MetalElements)+1, 1)])
        self._weight[0, -1] = True

        # set SOS
        self.SOS_LABEL = len(self._label_to_source) + 1
        self._source_to_label[self._precursor_to_source['SOS']] = self.SOS_LABEL

        self._label_to_source = np.hstack([self._label_to_source, 
                                           [self._precursor_to_source['EOS'],
                                            self._precursor_to_source['SOS']]])
        self._embedding = np.vstack([self._embedding, 
                                     np.zeros_like(self._embedding[0]).reshape(1,-1),
                                     np.ones_like(self._embedding[0]).reshape(1,-1)])

    def get_embedding(self, precursor):
        try:
            label = self.to_label(precursor)
            return self._embedding[label]
        except:
            return np.zeros_like(self._embedding[0])

    def to_dict(self):
        return {
            'feature_type': self._feat_type, 
            'EOS_LABEL': self.EOS_LABEL, 
            'SOS_LABEL': self.SOS_LABEL, 
            'NUM_LABEL': self.NUM_LABEL,
        }

#####################################################################################

class LigandTemplateReference(BaseReference):
    def __init__(self, 
                 feat_type:str = 'composit',
                 by_fraction:bool = True,
                 label_weight_fnc = lambda x: 1/np.power(x['count'], 1.2),
                 *args, **kwargs):
        super().__init__(feat_type=feat_type, by_fraction=by_fraction,
                         label_weight_fnc=label_weight_fnc, *args, **kwargs)
#        self._source_vecs = np.vstack([composition_to_feature(c) for c in self._active_precursors])
        self.update()

    def update(self, active_precursors=None):
        if not isinstance(active_precursors, (list, np.ndarray, tuple, set)):
            active_precursors = self._active_precursors
        _active_precursors = []
        self._ligand_dict = {}
        for i_src, precursor in enumerate(self._precursor_source):
            if precursor['precursor_str'] not in active_precursors:
                continue
            _active_precursors.append(precursor['precursor_str'])
            non_metal = {}
            i_metal = 0
            for ele, n in precursor['precursor_comp'].items():
                if ele in MetalElements:
                    i_metal = MetalElements.index(ele) + 1
                else:
                    non_metal[ele] = n
            ligand_str = composit_parser(non_metal, norm=False)
            if ligand_str not in self._ligand_dict.keys():
                self._ligand_dict[ligand_str] = {
                    'label':None, 
                    'composition':non_metal,
                    'metals':[]}
            self._ligand_dict[ligand_str]['metals'].append((i_metal, i_src))
        self.NUM_LABEL = len(self._ligand_dict)
        self._active_precursors = _active_precursors.copy()

        self._source_to_label = np.zeros((len(self._precursor_source), 2), dtype=int)
        self._label_to_source = - np.ones((len(MetalElements)+1, self.NUM_LABEL), dtype=int)
        self._weight = np.zeros_like(self._label_to_source).astype(float)
        self._ligand_str = list(self._ligand_dict.keys())
        for j, ligand_info in enumerate(self._ligand_dict.values()):
            ligand_info['label'] = j
            for i, i_source in ligand_info['metals']:
                self._label_to_source[i, j] = i_source
                self._weight[i, j] = self._label_weight_fnc(self._precursor_source[i_source])
                self._source_to_label[i_source] = i, j

    def _check_valid_precursor(self, *args):
        i_src = None
        if len(args) == 1:
            if isinstance(args[0], dict):
                precursor = composit_parser(args[0])
            elif isinstance(args[0], str):
                precursor = args[0]
            else:
                raise ValueError('Precursor must be either `dict` or `str`, got', type(args[0]))
            try:
                i_src = self._precursor_to_source[precursor]
            except:
                i_src = None
        elif len(args) > 1:
            metal, ligand = args[:2]
            i_metal, i_ligand = 0, None
            if isinstance(metal, str) and metal in MetalElements:
                i_metal = MetalElements.index(metal) + 1
            elif isinstance(metal, numbers.Integral) and metal < len(MetalElements) + 1:
                i_metal = metal
            if isinstance(ligand, str) and (ligand in self._ligand_dict.keys()):
                i_ligand = self._ligand_dict[ligand]['label']
            elif isinstance(ligand, numbers.Integral) and ligand < len(self._ligand_dict):
                i_ligand = ligand
            if (i_metal is None) or (i_ligand is None):
                i_src = None
            else:
                i_src = self._label_to_source[i_metal, i_ligand]
            if i_src == -1:
                i_src = None
        if i_src is None:
            raise ValueError('Invalid precursor', args)
        return i_src        

    def to_label(self, *args):
        i_source = self._check_valid_precursor(*args)
        _, label = self._source_to_label[i_source]
        return label
    
    def get_info(self, *args):
        i_source = self._check_valid_precursor(*args)
        return self._precursor_source[i_source]
    
    def get_weight(self, metal):
        if metal in MetalElements:
            return self._weight[MetalElements.index(metal)+1].reshape(1,-1)
        else:
            return self._weight[0].reshape(1,-1)

    def to_dict(self):
        return {
            'feature_type': self._feat_type, 
            'NUM_LABEL': self.NUM_LABEL,
        }