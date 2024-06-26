from .utils import ActiveElements, AllElements, NEAR_ZERO, MetalElements, composit_parser
import numpy as np
import json, os, gzip, pickle, numbers

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
                         'Possible feature types:', ['composit','composit_extended'] + list(elmd.keys()),
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

class PrecursorDataset:
    def __init__(self, 
                 feat_type:str = 'composit',
                 by_fraction:bool = True,
                 norm:bool = True
                 ):
        self._feat_type = feat_type
        self._by_fraction = by_fraction
        self._norm = norm

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/screened_precursor.pkl.gz')
#        path = '../data/screened_precursor.pkl.gz'

        with gzip.open(path, 'rb') as f:
            self.precursor_source = pickle.load(f)
        self.active_precursors = [p['precursor_str'] for p in self.precursor_source]
        self.update()

    def update(self, active_precursors=None):
        if isinstance(active_precursors, (list, np.ndarray, tuple, set)):
            self.active_precursors = active_precursors

        self.label_to_precursor = []
        self.label_to_source = []
        self.precursor_to_label = {}
        self.source_to_label = -np.ones(len(self.precursor_source) + 2, dtype=int)
        self.embedding = []

        # parsing data based on new precursor set
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

        # make metal-precursor mask
        precursor_mask = np.zeros((len(MetalElements) + 1, self.NUM_LABEL), dtype=int)
        for i,j in enumerate(self.label_to_source):
            for ele in self.precursor_source[j]['precursor_comp'].keys():
                if ele not in MetalElements:
                    continue
                k = MetalElements.index(ele)
                precursor_mask[k, i] = 1
            if precursor_mask[:, i].sum() == 0:
                precursor_mask[-1, i] = 1
        self.precursor_mask = precursor_mask.astype(bool)

        # set EOS
        i = len(self.precursor_source)
        j = len(self.label_to_precursor)

        self.EOS_LABEL = j
        self.label_to_precursor.append('EOS')
        self.label_to_source.append(i)
        self.precursor_to_label.update({'EOS':j})
        self.source_to_label[i] = j
        self.embedding.append(np.zeros_like(self.embedding[0]))
        self.precursor_mask[-1, j] = True

        # set SOS
        self.SOS_LABEL = j + 1
        self.label_to_precursor.append('SOS')
        self.label_to_source.append(i + 1)
        self.precursor_to_label.update({'SOS':j + 1})
        self.source_to_label[i + 1] = j + 1
        self.embedding.append(np.ones_like(self.embedding[0]))

        self.label_to_precursor = np.array(self.label_to_precursor)
        self.label_to_source = np.array(self.label_to_source)
        self.embedding = np.vstack(self.embedding)
        
    def save(self, path, fn='precursor_data.json'):
        self.active_precursors = list(self.active_precursors)

        info = {
            'feat_type': self._feat_type,
            'by_fraction': self._by_fraction,
            'norm': self._norm,
            'active_precursors': self.active_precursors,
        }
        with open(os.path.join(path, fn), 'w') as f:
            json.dump(info, f, indent=4)
    
    def load(self, path, fn='precursor_data.json'):
        with open(os.path.join(path, fn), 'r') as f:
            info = json.load(f)
        self.__init__(feat_type = info['feat_type'],
                      by_fraction = info['by_fraction'],
                      norm = info['norm'])
        self.update(info['active_precursors'])

    def _check_valid_precursor(self, precursor):
        if isinstance(precursor, numbers.Integral):
            if precursor < self.NUM_LABEL:
                i = precursor
            else:
                print('Exceeding maximum precursor index', self.NUM_LABEL-1)
                return None
        elif isinstance(precursor, str) and precursor in self.label_to_precursor:
            i = self.precursor_to_label[precursor]
        elif isinstance(precursor, dict) and composit_parser(precursor) in self.label_to_precursor:
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

    def get_precursor_mask_from_target(self, target):
        mask = self.precursor_mask[-1].copy()
        for ele in target.keys():
            if ele in MetalElements:
                mask = mask | self.precursor_mask[MetalElements.index(ele)]
        return mask.reshape(1,-1)

    def to_dict(self):
        return {
            'feature_type': self._feat_type, 
            'EOS_LABEL': self.EOS_LABEL, 
            'SOS_LABEL': self.SOS_LABEL, 
            'NUM_LABEL': self.NUM_LABEL,
        }
