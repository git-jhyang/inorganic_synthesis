from pymatgen.core import Element, Composition
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

def parse_rxn_ids(data):
    rxn_ids_ = []
    n = 0
    for rxn_id in data:
        rxn_ids_.append(rxn_id + n - rxn_id.min())
        n += rxn_id.max() + 1
    rxn_ids = np.hstack(rxn_ids_)
    is_last = np.hstack([rxn_ids[1:] != rxn_ids[:-1], [True]])
    return rxn_ids, is_last

def compute_metrics_from_cvae_output_v0(output, th=None, print_result=False):
    pred_has = 1 / (1 + np.exp(-np.hstack(output['pred_has'])))
    pred_lbl = np.vstack(output['pred_label'])
    label = np.vstack(output['label']).astype(bool)
    weight = np.vstack(output['weight'])
    label_mask = weight > 0
    has_label = label.sum(1)
    rxn_ids, is_last = parse_rxn_ids(output['rxn_id'])

    if th is None:
        ths = np.linspace(0.1, 0.9, 801)
        has_hit = [np.mean(has_label[is_last] == (pred_has[is_last] > th)) for th in ths]
        th = ths[np.argmax(has_hit)]
    acc = np.mean(has_label[is_last] == (pred_has[is_last] > th))
    has_label = np.ones_like(is_last)
    has_label[is_last] = pred_has[is_last] > th

    out = {f'top_{i+1}':{} for i in range(4)}
    for i, l, p, m, h in zip(rxn_ids, label, pred_lbl, label_mask, has_label):
        if i not in out['top_1'].keys():
            for j in range(4):
                out[f'top_{j+1}'][i] = [[], []]
        if m.sum() == 1:
            for j in range(4):
                out[f'top_{j+1}'][i][0] = np.hstack([out[f'top_{j+1}'][i][0], [True]])
                out[f'top_{j+1}'][i][1] = np.hstack([out[f'top_{j+1}'][i][1], [True]])
        check = h + l.sum()
        if check == 0:
            continue
        elif check == 1:
            for j in range(4):
                out[f'top_{j+1}'][i][0] = np.hstack([out[f'top_{j+1}'][i][0], [False]])
                out[f'top_{j+1}'][i][1] = np.hstack([out[f'top_{j+1}'][i][1], [False]])
        else:
            idxs = np.argsort(p)[::-1][:4]
            p_ = np.zeros_like(m)
            for j,k in enumerate(idxs):
                if m[k]:
                    p_[k] = True
                out[f'top_{j+1}'][i][0] = np.hstack([out[f'top_{j+1}'][i][0], p_[l]])
                out[f'top_{j+1}'][i][1] = np.hstack([out[f'top_{j+1}'][i][1], l[p_]])
    _out = {}
    for k,vs in out.items():
        r = np.hstack([v[0] for v in vs.values()]).mean()
        p = np.hstack([v[1] for v in vs.values()]).mean()
        f = 2 * r * p / (r + p)
        r2 = np.mean([np.sum(v[0] == 0) == 0 for v in vs.values()])
        p2 = np.mean([np.sum(v[1] == 0) == 0 for v in vs.values()])
        f2 = 2 * r2 * p2 / (r2 + p2)
        _out[k] = [f, p, r, f2, p2, r2]
    
    if print_result:
        print('Null label hit accuracy : {:.4f} (th: {:.3f})'.format(acc, th))
        print('-' * 56)
        print('{:7s}| {:23s}| {}'.format('', 'Precursor', 'Reaction'))
        s = ' '.join([f'{s:7s}' for s in ['F1','Prec','Recall']])
        print('{:7s}| {}| {}'.format('', s, s))
        print('-' * 56)
        for k, vs in _out.items():
            print('{:6s} | {} | {}'.format(k, '  '.join([f'{v:.4f}' for v in vs[:3]]), '  '.join([f'{v:.4f}' for v in vs[3:]])))
    return acc, th, _out

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

