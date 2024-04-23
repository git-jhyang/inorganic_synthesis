import multiprocessing as mp
import run_cvae as R
import numpy as np

hidden_dims = [32, 64, 128, 256]
hidden_layers = [2, 3, 4, 5]
latent_dims = [8, 16, 32]
graph_type = ['conv','attention']
target_feature_type = ['active_composit','cgcnn','elemnet','magpie','magpie_sc','mat2vec','matscholar','megnet16','oliynyk','oliynyk_sc']


def exc(i):
    np.random.seed(i)
    i_hd = np.random.randint(0, len(hidden_dims))
    i_hl = np.random.randint(0, len(hidden_layers))
    i_ld = np.random.randint(0, len(latent_dims))
    i_te = np.random.randint(0, len(target_feature_type))
    i_gt = np.random.randint(0, len(graph_type))

    R.args.hidden_dim = hidden_dims[i_hd]
    R.args.hidden_layers = hidden_layers[i_hl]
    R.args.latent_dim = latent_dims[i_ld]
    R.args.target_feature_type = target_feature_type[i_te]
    R.args.graph_type = graph_type[i_gt]
    R.main(R.args)

with mp.Pool(2) as pool:
    pool.map(exc, [i for i in range(200)])