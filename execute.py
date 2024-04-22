import multiprocessing as mp
import run_cvae as R
import numpy as np

hidden_dims = [32, 64, 128, 256, 512]
hidden_layers = [2, 3, 4]
latent_dims = [8, 16, 32]
target_feature_type = ['active_composit','cgcnn','elemnet','magpie','magpie_sc','mat2vec','matscholar','megnet16','oliynyk','oliynyk_sc']


def exc(i):
    np.random.seed(i)
    i_ehd, i_dhd = np.random.randint(0, len(hidden_dims), 2)
    i_ehl, i_dhl = np.random.randint(0, len(hidden_layers), 2)
    i_ld = np.random.randint(0, len(latent_dims))
    i_te = np.random.randint(0, len(target_feature_type))

    R.args.encoder_hidden_dim = hidden_dims[i_ehd]
    R.args.encoder_hidden_layers = hidden_layers[i_ehl]
    R.args.decoder_hidden_dim = hidden_dims[i_dhd]
    R.args.decoder_hidden_layers = hidden_layers[i_dhl]
    R.args.latent_dim = latent_dims[i_ld]
    R.args.target_feature_type = target_feature_type[i_te]
    R.main(R.args)

with mp.Pool(3) as pool:
    pool.map(exc, [i for i in range(200)])