import multiprocessing as mp
import seq_crxn as R
import numpy as np

param_dict = {
    'feature_type' : ['composit','cgcnn','elemnet','magpie_sc','mat2vec','matscholar','megnet16','oliynyk_sc',],
    'include_eos' : [1, 2],
    'shuffle_sequence' : [True, False],
    'weighted_loss': [True, False],
    'heads' : [2, 4, 8],
    'hidden_dims' : [32, 64, 128, 256],
    'hidden_layers' : [2, 4, 6, 8],
    'positional_encoding' : [True, False],
}

def exc(i):
    np.random.seed(i)
    for k, v in param_dict.items():
        setattr(R.args, k, v[np.random.randint(0, len(v))])
#    return R.args
    R.main(R.args)

with mp.Pool(2) as pool:
    pool.map(exc, [i for i in range(600)])
#for i in range(10):
#    print(exc(i))