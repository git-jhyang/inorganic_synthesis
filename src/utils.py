from pymatgen.core import Element, Composition
from itertools import product
import numpy as np
import torch, copy

NEAR_ZERO = 1e-5

NonMetals = 'H C N O F P S Cl Se Br I'.split()
AlkaliMetals = 'Li Na K Rb Cs Fr'.split()
AlkaliEarthMetals = 'Be Mg Ca Sr Ba Ra'.split()
TransitionMetals = 'Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg'.split()
PostTransitionMetals = 'Al Ga In Sn Tl Pb Bi Po'.split()
Metalloids = 'B Si Ge As Sb Te At'.split()
Lanthanoids = 'La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu'.split()
Actinoids_1 = 'Ac Th Pa U Np Pu'.split()
Actinoids_2 = 'Am Cm Bk Cf Es Fm Md No Lr'.split() # uncommon actinoids
Halogens = 'He Ne Ar Kr Xe Rn'.split()
Unknown = 'Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split()

MetalElements = sorted(AlkaliMetals + AlkaliEarthMetals + TransitionMetals + PostTransitionMetals + Metalloids + Lanthanoids + Actinoids_1 + Actinoids_2, key=lambda x: Element(x).number)
LigandElements = sorted(NonMetals, key=lambda x: Element(x).number)
ActiveElements = sorted(
    AlkaliMetals + AlkaliEarthMetals + TransitionMetals + PostTransitionMetals + 
    Metalloids + Lanthanoids + Actinoids_1 + NonMetals, key=lambda x: Element(x).number)
AllElements = sorted(ActiveElements + Actinoids_2 + Halogens + Unknown, key=lambda x: Element(x).number)

def to_numpy(vector):
    if isinstance(vector, torch.Tensor):
        return vector.cpu().numpy()
    else:
        return np.array(vector)


################################################################################################


def squared_error(mat1, mat2, average=True):
    x = to_numpy(mat1)
    y = to_numpy(mat2)
    sq_err = np.sum(np.square(x - y), -1)
    if average:
        return sq_err.mean()
    else:
        return sq_err

def cosin_similarity(mat1, mat2, average=True):
    x = to_numpy(mat1)
    y = to_numpy(mat2)
    l = np.sqrt(np.sum(np.square(x), -1, keepdims=True)) * np.sqrt(np.sum(np.square(y), -1, keepdims=True))
    m = (l == 0).squeeze()
    cos_sim = np.zeros(m.shape, dtype=float)
    cos_sim[~m] = np.sum((x * y)[~m] / l[~m], -1)
    cos_sim[m] = 1 - np.sqrt(np.square(x - y)[m].sum(-1))
    if average:
        return cos_sim.mean()
    else:
        return cos_sim

# def find_nearest(vectors, reference):
#     out = []
#     for vec in vectors.reshape(-1, reference.shape[-1]):
#         sser = squared_error(vec, reference, average=False)
#         csim = cosin_similarity(vec, reference, average=False)
#         i = np.argmin(sser - csim)
#         out.append([i, sser[i], csim[i]])
#     return np.array(out).T


################################################################################################


def linear_kld_annealing(epochs, start=0, stop=1, period=500, ratio=0.5):
    '''
    Code from paper 'Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing'
    arXiv: https://arxiv.org/abs/1903.10145
    github: https://github.com/haofuml/cyclical_annealing

    Scheduling KLD for better training of VAE.
    '''
    beta = np.ones(epochs)
    step = (stop - start) / (period * np.clip(ratio, 0, 1))
    for i in range(int(period)):
        beta[i::int(period)] = start + step * i
    return np.clip(beta, start, stop)


################################################################################################


def exponential_kld_annealing(epochs, start=-35, stop=0, period=500, ratio=0.5):
    '''
    Code from paper 'Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing'
    arXiv: https://arxiv.org/abs/1903.10145
    github: https://github.com/haofuml/cyclical_annealing

    Scheduling KLD for better training of VAE.
    '''
    beta = np.ones(epochs)
    step = (stop - start) / (period * ratio)
    for i in range(int(period)):
        beta[i::int(period)] = np.power(10.0, start + step * i)
    return np.clip(beta, np.power(10.0, start), np.power(10.0, stop))


################################################################################################


def composit_parser(composit, fmt='{:.5f}', norm=True):
    if isinstance(composit, list):
        _comp = {}
        for comp in composit:
            for k, v in comp.items():
                if k in _comp:
                    _comp[k] += v
                else:
                    _comp[k] = v
    else:
        _comp = composit.copy()
    comp_str = []
    n = 1 if ((not norm) or (len(_comp) == 0)) else 1. / np.sum(list(_comp.values()))
    for k, v in sorted(_comp.items(), key=lambda x: Element(x[0]).number):
        comp_str.append(f'{k}_' + fmt.format(v * n))
    return ' '.join(comp_str)


################################################################################################


def check_precursor_frequency(reactions, comp_key='precursor_comp'):
    prec_idx = {}
    prec_data = []
    for rxn in reactions:
        for prec_comp in rxn[comp_key]:
            pstr = composit_parser(prec_comp)
            if pstr not in prec_idx.keys():
                prec_idx[pstr] = len(prec_data)
                prec_data.append({
                    'count_rxn': 1,
                    'count':rxn['count'],
                    'precursor_comp':prec_comp,
                    'precursor_str':pstr,
                })
            else:
                i = prec_idx[pstr]
                prec_data[i]['count_rxn'] += 1
                prec_data[i]['count'] += rxn['count']
    return sorted(prec_data, key=lambda x: x['count_rxn'], reverse=True)


################################################################################################


def screening_reactions_by_freq(reactions, precursors, minimum_frequency=5):
    freq = {d['precursor_str']:d['count_rxn'] for d in precursors if isinstance(d, dict)}
    screened_reaction = []
    for rxn in reactions:
        skip_rxn = False
        for prec_comp in rxn['precursor_comp']:
            pstr = composit_parser(prec_comp)
            if pstr not in freq.keys() or freq[pstr] < minimum_frequency:
                skip_rxn = True
                break
        if skip_rxn:
            continue
        screened_reaction.append(copy.deepcopy(rxn))
    screened_precursor = check_precursor_frequency(screened_reaction)
    min_count = screened_precursor[-1]['count_rxn']
    print(min_count, screened_precursor[-1]['count'], len(screened_reaction), len(screened_precursor))
    if min_count < minimum_frequency:
        return screening_reactions_by_freq(screened_reaction, screened_precursor, minimum_frequency)
    else:
        return screened_reaction, screened_precursor

# def sequence_output_metrics(pred, label):
#     if len(pred.shape) != len(label.shape):
#         pred = pred.argmax(-1)
#     N, S = pred.shape
#     sorted(np.unique(label.reshape(N, S)[:, -1], return_counts=True), key=lambda x: x[1])

#     mask = np.hstack([np.ones((N, 1), dtype=bool), (label != DS.EOS_LABEL)[..., :-1]]).reshape(-1)
#     acc = accuracy_score(label.reshape(-1)[mask], pred.reshape(-1)[mask])
#     f1_mi = f1_score(label.reshape(-1)[mask], pred.reshape(-1)[mask], average='micro')
#     f1_ma = f1_score(label.reshape(-1)[mask], pred.reshape(-1)[mask], average='macro')
#     hit_rxn = np.array([(p[m] != l[m]).sum() == 0 for p, l, m in zip(pred, label, mask)]).astype(float).mean()

def heat_tempearture_norm(x):
    return x * 0.001 - 1

def heat_tempearture_denorm(x):
    return x * 1000 + 1000

def heat_time_norm(x):
    return np.log10(x) - 1

def heat_time_denorm(x):
    return np.power(10, x + 1)


################################################################################################

def get_is_last(reaction_id):
    return np.hstack([reaction_id[1:] != reaction_id[:-1], [True]])

################################################################################################

def sort_precursor_by_target_element(target, precursor):
    j = []
    for ele1 in target.keys():
        for i, comp in enumerate(precursor):
            if ele1 not in comp.keys(): continue
            if i in j: continue
            j.append(i)
    target_str = Composition(target).get_integer_formula_and_factor()[0]
    precursor_str = [Composition(p).get_integer_formula_and_factor()[0] for p in precursor]
    return target_str, [precursor_str[_j] for _j in j]

################################################################################################

def compute_acc(output, th=None, ths=np.linspace(0.1, 0.9, 81)):
    is_last = get_is_last(output['rxn_id'])
    if th is None:
        accs = [np.mean(output['label'].sum(1)[is_last] == (output['pred_has'][is_last] > th).astype(float)) for th in ths]
        th = ths[np.argmax(accs)]
    acc = np.mean(output['label'].sum(1)[is_last] == (output['pred_has'][is_last] > th).astype(float))
    return acc, th

def compute_top_accuracy(output):
    out = []
    for p, l in zip(output['pred_label'], output['label']):
        if l.sum() == 0:
            continue
        idxs = np.argsort(p)[::-1].tolist()
        out.append(idxs.index(l.argmax()))
    out = np.array(out)
    return np.mean(out), np.mean(out < 1), np.mean(out < 3) # mean_label_rank, accuracy, top-3 recall

# def compute_metrics_from_cvae_output_v0(output, n_top=4, print_result=False):
#     rxn_id = output['rxn_id']
#     label = 

#     top_output = {f'top_{i+1}':{} for i in range(n_top)}
#     for i, l, p, m, h in zip(rxn_id, label, pred_lbl, label_mask, has_label):
#         if i not in top_output['top_1'].keys():
#             for j in range(n_top):
#                 top_output[f'top_{j+1}'][i] = [[], []]
#         if m.sum() == 1:
#             for j in range(n_top):
#                 top_output[f'top_{j+1}'][i][0] = np.hstack([top_output[f'top_{j+1}'][i][0], [True]])
#                 top_output[f'top_{j+1}'][i][1] = np.hstack([top_output[f'top_{j+1}'][i][1], [True]])
#             continue
#         check = h + l.sum()
#         if check == 0:
#             continue
#         elif check == 1:
#             for j in range(n_top):
#                 top_output[f'top_{j+1}'][i][0] = np.hstack([top_output[f'top_{j+1}'][i][0], [False]])
#                 top_output[f'top_{j+1}'][i][1] = np.hstack([top_output[f'top_{j+1}'][i][1], [False]])
#         else:
#             idxs = np.argsort(p)[::-1][:n_top]
#             p_ = np.zeros_like(m)
#             for j,k in enumerate(idxs):
#                 if m[k]:
#                     p_[k] = True
#                 top_output[f'top_{j+1}'][i][0] = np.hstack([top_output[f'top_{j+1}'][i][0], p_[l]])
#                 top_output[f'top_{j+1}'][i][1] = np.hstack([top_output[f'top_{j+1}'][i][1], l[p_]])
#     output_metric = {}
#     for k, vs in top_output.items():
#         r = np.hstack([v[0] for v in vs.values()]).mean()
#         p = np.hstack([v[1] for v in vs.values()]).mean()
#         f = 2 * r * p / (r + p)
#         r2 = np.mean([np.sum(v[0] == 0) == 0 for v in vs.values()])
#         p2 = np.mean([np.sum(v[1] == 0) == 0 for v in vs.values()])
#         f2 = 2 * r2 * p2 / (r2 + p2)
#         output_metric[k] = [f, p, r, f2, p2, r2]
    
#     if print_result:
#         print('Null label hit accuracy : {:.4f} (th: {:.3f})'.format(acc, th))
#         print('-' * 56)
#         print('{:7s}| {:23s}| {}'.format('', 'Precursor', 'Reaction'))
#         s = ' '.join([f'{s:7s}' for s in ['F1','Prec','Recall']])
#         print('{:7s}| {}| {}'.format('', s, s))
#         print('-' * 56)
#         for k, vs in output_metric.items():
#             print('{:6s} | {} | {}'.format(k, '  '.join([f'{v:.4f}' for v in vs[:3]]), '  '.join([f'{v:.4f}' for v in vs[3:]])))
#     return acc, th, output_metric, parsed_output

def get_likely_list(target, metals, preds, pred_has, precursor_reference):
    ele_mask = [precursor_reference.get_weight(e).reshape(-1) > 0 for e in target.keys() if e not in MetalElements]
    sorted_probs = []
    for metal, prob in zip(metals, preds):
        step_probs = []
        idxs = np.argsort(prob)[::-1]
        p_sum = 0
        for idx in idxs:
            p = prob[idx]
            if p_sum > 0.99 or p < 0.01:
                break
            p_sum += p
            step_probs.append([p, idx])
        if metal == 'none':
            step_probs = [[1 - pred_has, precursor_reference._ligand_str.index('')]] + [[_p * pred_has, _i] for _p,_i in step_probs]
        sorted_probs.append(step_probs)

    likelies = []
    l_sum = 0
    for comb in product(*sorted_probs):
        if np.sum([c[0] for c in comb]) < 0.1:
            continue
        labels = [c[1] for c in comb]
        skip = False
        if len(ele_mask) != 0:
            label_mask = np.zeros_like(ele_mask[0], dtype=bool)
            label_mask[labels] = True
            for _mask in ele_mask:
                if (_mask & label_mask).sum() == 0:
                    skip = True
                    break
        if skip:
            continue
        l = np.prod([c[0] for c in comb])
        p = []
        for metal, label in zip(metals, labels):
            if metal == 'none' and label == precursor_reference._ligand_str.index(''):
                continue
            p_comp = precursor_reference.get_info(metal, label)['precursor_comp']
            p.append(Composition(p_comp).get_integer_formula_and_factor()[0])
        likelies.append((tuple(p), l))
        l_sum += l
    return [(l[0], l[1]/l_sum) for l in sorted(likelies, key=lambda x: x[1], reverse=True)]


def train_test_split(n_data, valid_ratio=None, test_ratio=None, seed=None):
    if isinstance(seed, int):
        np.random.seed(seed)
    if valid_ratio is None:
        n_valid = 0
    else:
        n_valid = int(n_data * valid_ratio)
    if test_ratio is None:
        n_test = 0
    else:
        n_test = int(n_data * test_ratio)
    if n_valid + n_test == 0:
        raise ValueError("Neither `valid_ratio` nor `test_ratio` is given")
    n_train = n_data - n_valid - n_test
    i_valid = n_train + n_valid

    idxs = np.arange(n_data)
    np.random.shuffle(idxs)

    train_idx = idxs[:n_train]
    valid_idx = idxs[n_train:i_valid]
    test_idx  = idxs[i_valid:]

    return train_idx, valid_idx, test_idx

class CrossValidation:
    def __init__(self, n_fold:int, data=None, n_data=None, stratum=None, return_index=False, seed:int=None):
        if data is None and n_data is None:
            raise ValueError('Either `n_data` or `data` should be given')
        if data is None:
            return_index = True
        if isinstance(seed, int):
            np.random.seed(seed)

        self.return_index = return_index
        if data is not None:
            n_data = len(data)
            self._data = np.array(data)

        index = np.arange(n_data)
        if stratum is None:
            k = np.min([n_fold, n_data])
            if k < n_fold:
                print(f"Notice: Reduced number of folds ({n_fold} -> {k})")
            np.random.shuffle(index)
            self.train_index, self.valid_index = self._split_(index, k)
        else:
            if len(stratum) != n_data:
                raise ValueError(f"Dimension mismatch between `data` ({n_data}) and `stratum` ({len(stratum)})")
            stratum = np.array(stratum)
            cs = np.sort(np.unique(stratum))
            ks = np.array([np.sum(c == stratum) for c in cs])
            k  = np.min([n_fold, np.max(ks)])
            if k < n_fold:
                print(f"Notice: Reduced number of folds ({n_fold} -> {k})")
            train_index = [[] for _ in range(k)]
            valid_index = [[] for _ in range(k)]
            if k > np.min(ks):
                print(f"Warning: Number of folds is larger than number of data (got: {k} / min: {np.min(ks)}).\nMore than one fold contains full data of class: {cs[ks < n_fold]}")
            for c in cs:
                idx = index[c == stratum]
                np.random.shuffle(idx)
                tidxs, vidxs = self._split_(idx, k)
                for i, (tidx, vidx) in enumerate(zip(tidxs, vidxs)):
                    train_index[i].append(tidx)
                    valid_index[i].append(vidx)
            self.train_index = [np.hstack(idx) for idx in train_index]
            self.valid_index = [np.hstack(idx) for idx in valid_index]

    def __getitem__(self, i:int):
        if self.return_index:
            return self.train_index[i], self.valid_index[i]
        else:
            return self._data[self.train_index[i]], self._data[self.valid_index[i]]

    def __len__(self):
        return len(self.train_index)
    
    def _split_(self, index, k):
        n = len(index)
        c = int(n/k) + bool(n%k)
        l = n - k * (c - bool(n%k))
        train_index = []
        valid_index = []
        i1 = 0
        i2 = c
        for i in range(k):
            train_index.append(np.hstack([index[:i1], index[i2:]]))
            valid_index.append(index[i1:i2])

            i1 += c
            if i == (l-1): c -= 1
            i2 += c

        return train_index, valid_index
