from .utils import ActiveElements, NEAR_ZERO, composit_parser
import numpy as np
import json, os, gzip, pickle

ATOM_EXCEPTION_WARNING = 'Warning: element [{}] is not included in feature type "{}"\n'
FEAT_EXCEPTION_WARNING = 'Warning: feature type [{}] is not supported. '

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
            elif ATOM_EXCEPTION_WARNING is not None:
                print(ATOM_EXCEPTION_WARNING.format(ele, feature_type))
        ATOM_EXCEPTION_WARNING = None
        return np.sum(vec, 0).astype(dtype).reshape(1,-1)
    else:
        if not feature_type.startswith('compo') and FEAT_EXCEPTION_WARNING is not None:
            print(FEAT_EXCEPTION_WARNING.format(feature_type))
            print('Possible feature types:', ['composit'] + list(elmd.keys()))
            print('Feature type is set to composit')
            FEAT_EXCEPTION_WARNING = None
        return active_composit_feature(composit_dict, dtype, by_fraction,).reshape(1,-1) * div

def active_composit_feature(composit_dict, dtype=float, by_fraction=True, *args, **kwargs):
    feat_vec = np.zeros(len(ActiveElements), dtype=dtype)
    for ele, frac in composit_dict.items():
        if frac <= NEAR_ZERO: continue
        feat_vec[ActiveElements.index(ele)] = frac if by_fraction else 1
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

        self.EOS_LABEL = j
        self.label_to_precursor.append('EOS')
        self.label_to_source.append(i)
        self.precursor_to_label.update({'EOS':j})
        self.source_to_label[i] = j
        self.embedding.append(np.zeros_like(self.embedding[0]))

        self.SOS_LABEL = j + 1
        self.label_to_precursor.append('SOS')
        self.label_to_source.append(i + 1)
        self.precursor_to_label.update({'SOS':j + 1})
        self.source_to_label[i + 1] = j + 1
        self.embedding.append(np.ones_like(self.embedding[0]))

        self.label_to_precursor = np.array(self.label_to_precursor)
        self.label_to_source = np.array(self.label_to_source)
        self.embedding = np.vstack(self.embedding)
        
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
        self.update(info['active_precursors'])

    def _check_valid_precursor(self, precursor):
        if isinstance(precursor, int) and precursor < self.NUM_LABEL:
            i = precursor
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

    def to_dict(self):
        return {
            'feature_type': self._feat_type, 
            'EOS_LABEL': self.EOS_LABEL, 
            'SOS_LABEL': self.SOS_LABEL, 
            'NUM_LABEL': self.NUM_LABEL,
        }
