import multiprocessing as mp
import train as T
import numpy as np


feature_type = ['composit','cgcnn','elemnet','magpie_sc','mat2vec','matscholar','megnet16','oliynyk_sc',]
batch_size = [64, 128, 256]
hidden_dims = [64, 128, 256]
hidden_layers = [2, 4, 6]
latent_dims = [8, 16, 32]

def exc(i, shared_list):
    np.random.seed(i)
    while True:
        i_ft = np.random.randint(0, len(feature_type))
        i_bs = np.random.randint(0, len(batch_size))
        i_hd = np.random.randint(0, len(hidden_dims))
        i_hl = np.random.randint(0, len(hidden_layers))
        i_ld = np.random.randint(0, len(latent_dims))
#        i_bn = int(np.random.rand() > 0.5)
        key = (i_ft, i_bs, i_hd, i_hl, i_ld)
        if key not in shared_list:
            shared_list.append(key)
            break

    T.args.data_feature_type = feature_type[i_ft]
    T.args.train_batch_size = batch_size[i_bs]
    T.args.model_hidden_dim = hidden_dims[i_hd]
    T.args.model_hidden_layers = hidden_layers[i_hl]
    T.args.model_latent_dim = latent_dims[i_ld]
#    T.args.model_batch_norm = bool(i_bn)
    T.main(T.args)

with mp.Pool(2) as pool:
    shared_list = mp.Manager().list()
    pool.starmap(exc, [(i, shared_list) for i in range(600)])
#
#for i in range(600):
#    exc(i)