import os, pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.constants import cosin_similarity
from src.networks import CVAE, GraphCVAE
from src.data import ConditionDataset, ReactionDataset
from src.trainer import VAETrainer
from src.feature import feature_to_ligand_index
from src.utils import squared_error, cyclical_kld_annealing
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from argparse import ArgumentParser

parser = ArgumentParser()
#parser.add_argument('--encoder_hidden_dim', default=32)
#parser.add_argument('--encoder_hidden_layers', default=2)
#parser.add_argument('--decoder_hidden_dim', default=32)
#parser.add_argument('--decoder_hidden_layers', default=2)
parser.add_argument('--hidden_dim', default=32, type=int)
parser.add_argument('--hidden_layers', default=2, type=int)
parser.add_argument('--latent_dim', default=16, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--graph', default='conv', type=str)
parser.add_argument('--epochs', default=5000, type=int)
parser.add_argument('--early_stop', default=100, type=int)
parser.add_argument('--logging_interval', default=5, type=int)
parser.add_argument('--target_feature_type', default='active_composit', type=str)

args = parser.parse_args()

torch.set_float32_matmul_precision('high')

def main(args):
    identifier = '{:s}_{:s}_batch_{:04d}_mdim_{:02d}_{:03d}_{:1d}'.format(
        args.graph, args.target_feature_type, args.batch_size, 
        args.latent_dim, args.hidden_dim, args.hidden_layers, 
    )

    output_path = f'/home/jhyang/WORKSPACES/MODELS/isyn/VAE_graph/{identifier}'
    if os.path.isdir(output_path):
        for i in range(100):
            if not os.path.isdir(output_path + f'_case_{i:02d}'):
                break
        output_path += f'_case_{i:02d}'
    os.makedirs(output_path, exist_ok=True)
    writer = SummaryWriter(output_path)

    DS = ReactionDataset(target_feat_type=args.target_feature_type)
    DS.from_file(data_path='./data/unique_reaction.pkl.gz')
    DS.to('cuda')

    years = np.array([d.year for d in DS])
    test_mask = years > 2018
    train_mask = years < 2017
    valid_mask = ~train_mask & ~test_mask

    train_dl = DataLoader(DS, batch_size=args.batch_size, sampler=SubsetRandomSampler(np.where(train_mask)[0]), collate_fn=DS.cfn)
    valid_dl = DataLoader(DS, batch_size=4096, sampler=np.where(valid_mask)[0], collate_fn=DS.cfn)
    test_dl = DataLoader(DS, batch_size=4096, sampler=np.where(test_mask)[0], collate_fn=DS.cfn)

    model = GraphCVAE(input_dim = DS.num_precursor_feat, 
                      edge_dim = DS.num_target_feat,
                      latent_dim = args.latent_dim, 
                      condition_dim = DS.num_metal_feat,
                      graph = args.graph,
                      encoder_hidden_dim = args.hidden_dim, 
                      encoder_hidden_layers = args.hidden_layers, 
                      decoder_hidden_dim = args.hidden_dim, 
                      decoder_hidden_layers = args.hidden_layers,
    )

    trainer = VAETrainer(model, 1e-4, device='cuda')
    betaset = cyclical_kld_annealing(args.epochs, n_cycle=10)
    best_loss = 1e5
    count = 0
    
    for epoch, beta in zip(range(1, args.epochs+1), betaset):
        train_loss = trainer.train(train_dl, beta=beta)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        if epoch % args.logging_interval == 0:
            valid_loss, valid_output = trainer.test(valid_dl, beta=beta)
            v_t = feature_to_ligand_index(valid_output['input'])
            v_p = feature_to_ligand_index(valid_output['pred'])
            v_acc = accuracy_score(v_t, v_p)
            v_f1 = f1_score(v_t, v_p, average='micro')
            v_mse = squared_error(valid_output['input'], valid_output['pred'])
            v_csim = cosin_similarity(valid_output['input'], valid_output['pred'])
            writer.add_scalar('Loss/Valid', valid_loss, epoch)
            writer.add_scalar('KLD/Valid', valid_output['kld'].sum(), epoch)
            writer.add_scalar('ACC/Valid', v_acc, epoch)
            writer.add_scalar('F1/Valid', v_f1, epoch)
            writer.add_scalar('MSE/Valid', v_mse, epoch)
            writer.add_scalar('CSIM/Valid', v_csim, epoch)
    
            test_loss, test_output = trainer.test(test_dl, beta=beta)
            t_t = feature_to_ligand_index(test_output['input'])
            t_p = feature_to_ligand_index(test_output['pred'])
            t_acc = accuracy_score(t_t, t_p)
            t_f1 = f1_score(t_t, t_p, average='micro')
            t_mse = squared_error(test_output['input'], test_output['pred'])
            t_csim = cosin_similarity(test_output['input'], test_output['pred'])
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('KLD/Test', test_output['kld'].sum(), epoch)
            writer.add_scalar('ACC/Test', t_acc, epoch)
            writer.add_scalar('F1/Test', t_f1, epoch)
            writer.add_scalar('MSE/Test', t_mse, epoch)
            writer.add_scalar('CSIM/Test', t_csim, epoch)

            if epoch % (args.logging_interval * 20) == 0:
                writer.add_histogram('Z/Valid', valid_output['z'], epoch)
                writer.add_histogram('Mu/Valid', valid_output['latent'][:,0], epoch)
                writer.add_histogram('logVar/Valid', valid_output['latent'][:,1], epoch)
                writer.add_histogram('Z/Test', test_output['z'], epoch)
                writer.add_histogram('Mu/Test', test_output['latent'][:,0], epoch)
                writer.add_histogram('logVar/Test', test_output['latent'][:,1], epoch)

            if valid_loss < best_loss:
                count = 0
                best_loss = valid_loss
                writer.add_scalar('Loss/BestValid', valid_loss, epoch)
                trainer.model.save(output_path, 'model')
                with open(os.path.join(output_path, 'output.valid.pkl'),'wb') as f:
                    pickle.dump(valid_output, f)
                with open(os.path.join(output_path, 'output.test.pkl'),'wb') as f:
                    pickle.dump(test_output, f)
                with open(os.path.join(output_path, 'epoch.txt'),'a') as f: 
                    f.write('{:8d} {:15.5f} {:15.5f} {:15.5f}\n'.format(epoch, train_loss, valid_loss, test_loss))
            else:
                count += args.logging_interval
                if args.early_stop > 0 and count > args.early_stop:
                    return

if __name__ == '__main__':
    import time
    t1 = time.time()

    main(args)

    with open('logging.txt','a') as f:
        f.write('{:10.2f} min - {:s}_batch_{:04d}_mdim_{:02d}_{:03d}_{:1d}_{:03d}_{:1d}\n'.format(
            (time.time() - t1) / 60.0,
            args.target_feature_type, args.batch_size, args.latent_dim, 
            args.encoder_hidden_dim, args.encoder_hidden_layers, 
            args.decoder_hidden_dim, args.decoder_hidden_layers, 
        ))