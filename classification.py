import os, pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.networks import GraphCVAE, CVAE
from src.data import ReactionDataset, ConditionDataset
from src.trainer import VAEClassTrainer
from src.utils import linear_kld_annealing
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from argparse import ArgumentParser

parser = ArgumentParser()
#parser.add_argument('--encoder_hidden_dim', default=32)
#parser.add_argument('--encoder_hidden_layers', default=2)
#parser.add_argument('--decoder_hidden_dim', default=32)
#parser.add_argument('--decoder_hidden_layers', default=2)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--hidden_layers', default=4, type=int)
parser.add_argument('--latent_dim', default=32, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--archi', default='conv', type=str)
parser.add_argument('--epochs', default=5000, type=int)
parser.add_argument('--early_stop', default=200, type=int)
parser.add_argument('--logging_interval', default=5, type=int)
parser.add_argument('--feature_type', default='active_composit', type=str)
parser.add_argument('--logging', action='store_true')

args = parser.parse_args()

torch.set_float32_matmul_precision('high')

def main(args):
    if args.archi.startswith('fcnn') or args.archi.startswith('fully'):
        args.archi = 'fcnn'
    if args.archi.startswith('att'):
        args.archi = 'attention'
    elif args.archi.startswith('conv'):
        args.archi = 'convolution'

    identifier = 'cls/{:s}_{:s}_batch_{:04d}_mdim_{:02d}_{:03d}_{:1d}'.format(
        args.archi, args.target_feature_type, args.batch_size, 
        args.latent_dim, args.hidden_dim, args.hidden_layers, 
    )
    output_path = f'/home/jhyang/WORKSPACES/MODELS/isyn/CVAE/{identifier}'
    if os.path.isdir(output_path):
        for i in range(100):
            if not os.path.isdir(output_path + f'_case_{i:02d}'):
                break
        output_path += f'_case_{i:02d}'

    os.makedirs(output_path, exist_ok=True)
    writer = SummaryWriter(output_path)
    if args.logging:
        print(args)
        print(output_path)

    if args.archi == 'fcnn':
        DS = ConditionDataset(target_feat_type = args.feature_type,
                              metal_feat_type = args.feature_type,
                              precursor_feat_type = args.feature_type,
                              precursor_feat_by_fraction = False)
        DS.from_file(data_path='./data/unique_reaction.pkl.gz')

        model = CVAE(
            input_dim = DS.num_precursor_feat, 
            latent_dim = args.latent_dim, 
            output_dim = DS.num_class,
            condition_dim = DS.num_metal_feat + DS.num_target_feat,
            encoder_hidden_dim = args.hidden_dim, 
            encoder_hidden_layers = args.hidden_layers, 
            decoder_hidden_dim = args.hidden_dim, 
            decoder_hidden_layers = args.hidden_layers,
        )

    else:
        DS = ReactionDataset(target_feat_type = args.feature_type,
                             metal_feat_type = args.feature_type,
                             precursor_feat_type = args.feature_type,
                             precursor_feat_by_fraction = False)
        DS.from_file(data_path='./data/unique_reaction.pkl.gz')

        model = GraphCVAE(
            input_dim = DS.num_precursor_feat, 
            edge_dim = DS.num_target_feat,
            latent_dim = args.latent_dim, 
            output_dim = DS.num_class,
            condition_dim = DS.num_metal_feat,
            graph = args.archi,
            encoder_hidden_dim = args.hidden_dim, 
            encoder_hidden_layers = args.hidden_layers, 
            decoder_hidden_dim = args.hidden_dim, 
            decoder_hidden_layers = args.hidden_layers,
        )    

    DS.to('cuda')

    years = np.array([d.year for d in DS])
    test_mask = years > 2018
    train_mask = years < 2017
    valid_mask = ~train_mask & ~test_mask

    train_dl = DataLoader(DS, batch_size=args.batch_size, sampler=SubsetRandomSampler(np.where(train_mask)[0]), collate_fn=DS.cfn)
    valid_dl = DataLoader(DS, batch_size=4096, sampler=np.where(valid_mask)[0], collate_fn=DS.cfn)
    test_dl = DataLoader(DS, batch_size=4096, sampler=np.where(test_mask)[0], collate_fn=DS.cfn)

    trainer = VAEClassTrainer(model, 1e-3, device='cuda')

    beta_schedular = linear_kld_annealing(args.epochs, period=500)
    best_loss = 1e5
    count = 0
    
    for epoch, beta in zip(range(1, args.epochs+1), beta_schedular):
        train_loss = trainer.train(train_dl, beta=beta)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        if epoch % args.logging_interval == 0:
            valid_loss, valid_output = trainer.test(valid_dl, beta=beta)
            test_loss, test_output = trainer.test(test_dl, beta=beta)
            
            for sfx, loss, output in zip(['Valid','Test'], [valid_loss, test_loss], [valid_output, test_output]):
                acc = accuracy_score(output['label'], output['pred'].argmax(1))
                f1_mi = f1_score(output['label'], output['pred'].argmax(1), average='micro')
                f1_ma = f1_score(output['label'], output['pred'].argmax(1), average='macro')                
#                mse = squared_error(output['input'], output['pred_vec'])
#                csim = cosin_similarity(output['input'], output['pred_vec'])            
                writer.add_scalar(f'Loss/{sfx}', loss, epoch)
                writer.add_scalar(f'KLD/{sfx}', output['kld'].sum(), epoch)
                writer.add_scalar(f'ACC/{sfx}', acc, epoch)
                writer.add_scalar(f'F1-micro/{sfx}', f1_mi, epoch)
                writer.add_scalar(f'F1-macro/{sfx}', f1_ma, epoch)                
#                writer.add_scalar(f'MSE/{sfx}', mse, epoch)
#                writer.add_scalar(f'CSIM/{sfx}', csim, epoch)

                if epoch % (args.logging_interval * 10) == 0:
                    writer.add_histogram(f'Z/{sfx}', output['z'], epoch)
                    writer.add_histogram(f'Mu/{sfx}', output['mu'], epoch)
                    writer.add_histogram(f'Var/{sfx}', output['var'], epoch)

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
            if args.logging:
                print('{:5d} | {:8.4f} | {:8.4f} {:10.6f} | {:8.4f} {:10.6f}'.format(
                    epoch, train_loss, 
                    valid_loss, valid_output['kld'].sum(), 
                    test_loss, test_output['kld'].sum()))

if __name__ == '__main__':
    main(args)