import multiprocessing as mp
import classification as R
import numpy as np

hidden_dims = [64, 128, 256]
hidden_layers = [2, 4, 6]
latent_dims = [8, 16, 32]
feature_type = ['active_composit','cgcnn','elemnet','magpie','mat2vec','matscholar','megnet16','oliynyk',]
archi = ['fcnn','conv']

def exc(i):
    np.random.seed(i)
    i_hd = np.random.randint(0, len(hidden_dims))
    i_hl = np.random.randint(0, len(hidden_layers))
    i_ld = np.random.randint(0, len(latent_dims))
    i_te = np.random.randint(0, len(feature_type))
    i_ar = np.random.randint(0, len(archi))

    R.args.archi = archi[i_ar]
    R.args.hidden_dim = hidden_dims[i_hd]
    R.args.hidden_layers = hidden_layers[i_hl]
    R.args.latent_dim = latent_dims[i_ld]
    R.args.feature_type = feature_type[i_te]
    R.main(R.args)

#with mp.Pool(2) as pool:
#    pool.map(exc, [i for i in range(600)])
#
for i in range(600):
    exc(i)