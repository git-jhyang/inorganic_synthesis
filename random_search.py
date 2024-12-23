import multiprocessing as mp
import train as T
import numpy as np
from itertools import product


#feature_type = ['cgcnn','elemnet','magpie','oliynyk','elemnet+magnet','magpie+magnet','cgcnn+elemnet']
feature_type = ['magpie','oliynyk','elemnet+magnet']
batch_size = [32, 64]
hidden_dims = [64, 128]
hidden_layers = [2, 4]
latent_dims = [4, 8, 16]

hps = list(product(*[feature_type, batch_size, hidden_dims, hidden_layers, latent_dims]))


def exc(i, shared_list):
#     np.random.seed(i)
#     while True:
#         i_ft = np.random.randint(0, len(feature_type))
#         i_bs = np.random.randint(0, len(batch_size))
#         i_hd = np.random.randint(0, len(hidden_dims))
#         i_hl = np.random.randint(0, len(hidden_layers))
#         i_ld = np.random.randint(0, len(latent_dims))
# #        i_bn = int(np.random.rand() > 0.5)
#         key = (i_ft, i_bs, i_hd, i_hl, i_ld)
# #        key = (i_bs, i_hd, i_hl, i_ld)
#         if key not in shared_list:
#             shared_list.append(key)
#             break

#     T.args.data_feature_type = feature_type[i_ft]
#     T.args.train_batch_size = batch_size[i_bs]
#     T.args.model_hidden_dim = hidden_dims[i_hd]
#     T.args.model_hidden_layers = hidden_layers[i_hl]
#     T.args.model_latent_dim = latent_dims[i_ld]

    # from narrowed hp scope, iterate all conditions
    if i >= len(hps):
        return
    ft, bs, hd, hl, ld = hps[i]
    T.args.data_feature_type = ft
    T.args.train_batch_size = bs
    T.args.model_hidden_dim = hd
    T.args.model_hidden_layers = hl
    T.args.model_latent_dim = ld


    T.args.split_cross_valid = 5
#    T.args.model_batch_norm = bool(i_bn)
    T.args.output_path = '/home/jhyang/WORKSPACES/MODELS/isyn/GCVAE_CASE_1_condition'
    T.args.data_path = './data/screened_document_reaction_ss.pkl.gz'
    T.args.split_by_year = False
    T.args.split_cross_valid == 5
    T.main(T.args)

    T.args.split_by_year = True
    T.args.split_cross_valid == 0
    T.main(T.args)

with mp.Pool(2) as pool:
    shared_list = mp.Manager().list()
    pool.starmap(exc, [(i, shared_list) for i in range(200)])
