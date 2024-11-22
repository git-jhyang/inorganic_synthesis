import os, pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.networks import GraphCVAE
from src.data import ReactionGraphDataset
from src.trainer import VAETrainer
from src.utils import (NEAR_ZERO, linear_kld_annealing, compute_acc, compute_top_accuracy, 
                       CrossValidation, train_test_split)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from argparse import ArgumentParser

parser = ArgumentParser()
#parser.add_argument('--encoder_hidden_dim', default=32)
#parser.add_argument('--encoder_hidden_layers', default=2)
#parser.add_argument('--decoder_hidden_dim', default=32)
#parser.add_argument('--decoder_hidden_layers', default=2)
parser.add_argument('--data_feature_type', default='composit', type=str)
parser.add_argument('--data_path', default='./data/screened_unique_reaction_ss.pkl.gz', type=str)
parser.add_argument('--output_path', default='/home/jhyang/WORKSPACES/MODELS/isyn/GCVAE', type=str)

parser.add_argument('--split_by_year', action='store_true')
parser.add_argument('--split_cross_valid', default=0, type=int)
parser.add_argument('--split_random_ratio', default=(0.2, 0.1), nargs=2, type=float)

parser.add_argument('--train_epochs', default=2000, type=int)
parser.add_argument('--train_early_stop', default=200, type=int)
parser.add_argument('--train_logging_interval', default=5, type=int)
parser.add_argument('--train_logging', action='store_true')
parser.add_argument('--train_batch_size', default=128, type=int)
#parser.add_argument('--train_cross_validation', default=5, type=int)
#parser.add_argument('--train_chronological_order', action='store_true')
#parser.add_argument('--data_train_time_before', default=2017, type=int)
#parser.add_argument('--data_test_time_after', default=2018, type=int)

parser.add_argument('--model_graph', default='conv', type=str)
parser.add_argument('--model_hidden_dim', default=128, type=int)
parser.add_argument('--model_hidden_layers', default=4, type=int)
parser.add_argument('--model_latent_dim', default=16, type=int)
#parser.add_argument('--model_batch_norm', default=True, type=bool)
#parser.add_argument('--model_dropout', default=0, type=int)

args = parser.parse_args()
# if args.model_batch_norm:
#     args.model_dropout = 0.0
# else:
#     args.model_dropout = 0.5

torch.set_float32_matmul_precision('high')

def record_output(summary_writer, epoch, sfx, loss, output, th=None, record_dist=False):
    if th is not None:
        summary_writer.add_scalar('Hit/Th', th, epoch)
    acc, th = compute_acc(output, th)
    avg_rank, label_acc, label_recall = compute_top_accuracy(output)
    summary_writer.add_scalar(f'Loss/{sfx}', loss, epoch)
    summary_writer.add_scalar(f'KLD/{sfx}', np.vstack(output['kld']).sum(), epoch)

    summary_writer.add_scalar(f'Rank/{sfx}', avg_rank, epoch)
    summary_writer.add_scalar(f'Acc/{sfx}', label_acc, epoch)
    summary_writer.add_scalar(f'Top_3_Recall/{sfx}', label_recall, epoch)
    summary_writer.add_scalar(f'Hit/{sfx}Acc', acc, epoch)

    if record_dist:
        summary_writer.add_histogram(f'Z/{sfx}', np.vstack(output['z']), epoch)
        summary_writer.add_histogram(f'Mu/{sfx}', np.vstack(output['mu']), epoch)
        summary_writer.add_histogram(f'LogVar/{sfx}', np.vstack(output['log_var']), epoch)

def main(args):
    if args.model_graph.lower().startswith('att'):
        args.model_graph = 'attention'
    elif args.model_graph.lower().startswith('conv'):
        args.model_graph = 'convolution'

    identifier = '{:s}_{:s}_m{:02d}.{:03d}.{:1d}_batch_{:04d}'.format(
        args.data_feature_type, args.model_graph, 
        args.model_latent_dim, args.model_hidden_dim, args.model_hidden_layers, 
#        'bn' if args.model_batch_norm else 'do',
        args.train_batch_size, 
    )

    split_type = 'random_{:.2f}_{:.2f}'.format(*args.split_random_ratio)

    if args.split_by_year:
        split_type = 'year_2016_2018'
    elif args.split_cross_valid != 0:
        split_type = 'cv_00'

    output_path = os.path.join(args.output_path, identifier, split_type)
    if os.path.isdir(output_path):
        return

    if args.train_logging:
        print(args)

    DS = ReactionGraphDataset(feat_type=args.data_feature_type)
    DS.from_file(data_path=args.data_path,
                 heat_temp_fnc = lambda x: x['heat_temp']['median'],
                 heat_time_fnc = lambda x: x['heat_time']['median'])

    ############
    n_iter = 1
    if 'year' in split_type:
        years = np.array([d.year for d in DS])
        train_idx = np.where(years < 2017)[0]
        valid_idx = np.where((years > 2016) & (years < 2019))[0]
        test_idx  = np.where(years > 2018)[0]
    elif 'cv' in split_type:
        CV = CrossValidation(n_fold=args.split_cross_valid, n_data=len(DS), return_index=True, seed=100)
        n_iter = args.split_cross_valid
    else:
        train_idx, valid_idx, test_idx = train_test_split(len(DS), *args.split_random_ratio, seed=100)

    ############

    for n in range(n_iter):
        if 'cv' in split_type:
            output_path = os.path.join(args.output_path, identifier, f'cv_{n:02d}')
            train_idx, valid_idx = CV[n]
            test_idx = []

        if args.train_logging:
            print()
            print(output_path)
            head = '{:>11s} | {:>8s}'.format('Epoch','Train')
            if len(valid_idx) != 0:
                head += ' | {:>19s}'.format('Valid (Loss/KLD)')
            if len(test_idx) != 0:
                head += ' | {:>19s}'.format('Test (Loss/KLD)')
            print(head)
        os.makedirs(output_path, exist_ok=True)
        writer = SummaryWriter(output_path)

        train_dl = DataLoader(DS, batch_size=args.train_batch_size, sampler=SubsetRandomSampler(train_idx), collate_fn=DS.cfn)
        valid_dl = DataLoader(DS, batch_size=2048, sampler=valid_idx, collate_fn=DS.cfn)
        test_dl = DataLoader(DS, batch_size=2048, sampler=test_idx, collate_fn=DS.cfn)

        model = GraphCVAE(input_dim = DS.num_precursor_feat, 
                        latent_dim = args.model_latent_dim, 
                        condition_dim = DS.has_temp_info + DS.has_time_info,
                        edge_dim = DS.num_edge_feat,
                        output_dim = DS.NUM_LABEL + 1, 
                        graph = args.model_graph,
                        encoder_hidden_dim = args.model_hidden_dim, 
                        encoder_hidden_layers = args.model_hidden_layers, 
                        decoder_hidden_dim = args.model_hidden_dim, 
                        decoder_hidden_layers = args.model_hidden_layers,
                        batch_norm = True,
                        dropout = 0.0,
        )

        trainer = VAETrainer(model, 1e-5, device='cuda')
        betaset = linear_kld_annealing(args.train_epochs, start=NEAR_ZERO, stop=1, ratio=0.5, period=100)
        best_loss = 1e5
        count = 0
        
        for epoch, beta in zip(range(1, args.train_epochs+1), betaset):
            train_loss = trainer.train(train_dl, beta=beta)
            writer.add_scalar('Loss/Train', train_loss, epoch)
            log = '{:5d} ({:3d}) | {:8.4f}'.format(epoch, count, train_loss)

            if epoch % args.train_logging_interval == 0:
                th = None
                if len(valid_dl) != 0:
                    valid_loss, valid_output = trainer.test(valid_dl, beta=beta)
                    _, th = compute_acc(valid_output, None)
                    record_output(writer, epoch, 'Valid', valid_loss, valid_output, th)
                    log += ' | {:8.4f} {:10.6f}'.format(valid_loss, valid_output['kld'].sum())

                if len(test_dl) != 0:
                    test_loss, test_output = trainer.test(test_dl, beta=beta)
                    record_output(writer, epoch, 'Test', test_loss, test_output, th)
                    log += ' | {:8.4f} {:10.6f}'.format(test_loss, test_output['kld'].sum())
                
                if epoch < 50:
                    pass
                elif (len(valid_dl) != 0) and (valid_loss < best_loss):
                    count = 0
                    best_loss = valid_loss
                    writer.add_scalar('Loss/BestValid', valid_loss, epoch)
                    trainer.model.save(output_path, 'best')
                    with open(os.path.join(output_path, 'output.valid.pkl'),'wb') as f:
                        pickle.dump(valid_output, f)
                    if len(test_dl) != 0:
                        with open(os.path.join(output_path, 'output.test.pkl'),'wb') as f:
                            pickle.dump(test_output, f)
                else:
                    count += args.train_logging_interval
                    if args.train_early_stop > 0 and count > args.train_early_stop:
                        return

                if args.train_logging: 
                    print(log)

if __name__ == '__main__':
    main(args)