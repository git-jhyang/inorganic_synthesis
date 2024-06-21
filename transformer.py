import os, pickle
import torch
import numpy as np
from src.data import ReactionDataset
from src.trainer import SequenceTrainer
from src.networks import TransformerDecoderBlock
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

parser  =  ArgumentParser()
parser.add_argument('--data_type', default='unique', type=str)

parser.add_argument('--feature_type', default='composit', type=str)
parser.add_argument('--include_eos', default=1, type=int)
parser.add_argument('--shuffle_sequence', action='store_true')
parser.add_argument('--sequence_length', default=8, type=int)
parser.add_argument('--weighted_loss', action='store_true')

parser.add_argument('--heads', default=4, type=int)
parser.add_argument('--hidden_dim', default=64, type=int)
parser.add_argument('--hidden_layers', default=2, type=int)
parser.add_argument('--positional_encoding', action='store_true')

parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=5000, type=int)
parser.add_argument('--early_stop', default=200, type=int)
parser.add_argument('--logging_interval', default=5, type=int)

parser.add_argument('--logging', action='store_true')

args = parser.parse_args()
torch.set_float32_matmul_precision('high')

def main(args):
    if args.include_eos not in [0,1]:
        args.include_eos = 'all'
    if args.data_type.startswith('u'):
        args.data_type = 'u'
    else:
        args.data_type = 'c'

    identifier = '{}rxn/{:s}_EOS{}_{}{}{}_Head{:1d}_Dim{:03d}_Lay{:1d}_Bat{:03d}'.format(
        args.data_type, args.feature_type, args.include_eos, 
        'rS' if args.shuffle_sequence else 'oS', 
        'W' if args.weighted_loss else 'uW',
        'PE' if args.positional_encoding else 'nPE', 
        args.heads, args.hidden_dim, args.hidden_layers,
        args.batch_size, 
    )

    output_path = f'/home/jhyang/WORKSPACES/MODELS/isyn/tfdec/{identifier}'
    if os.path.isdir(output_path):
        print("Folder already exists")
        return
    os.makedirs(output_path, exist_ok=True)
    writer = SummaryWriter(output_path)
    if args.logging:
        print(args)
        print(output_path)

    DS = ReactionDataset(feat_type = args.feature_type,
                         include_eos = args.include_eos, 
                         shuffle_sequence = args.shuffle_sequence,
                         sequence_length = args.sequence_length,
                         weights = args.weighted_loss)
    
    if args.data_type == 'u':
        DS.from_file('./data/screened_unique_reaction.pkl.gz')
    else:
        DS.from_file('./data/screened_conditional_reaction.pkl.gz', 
                     heat_temp_key=('heat_temp','median'))

    years = np.array([d.year for d in DS])
    train_mask = years < 2016
    valid_mask = (years >= 2016) & (years < 2018)
    test_mask = years >= 2018

    train_dl = DataLoader(DS, batch_size=args.batch_size, sampler=SubsetRandomSampler(np.where(train_mask)[0]), collate_fn=DS.cfn)
    valid_dl = DataLoader(DS, batch_size=4096, sampler=np.where(valid_mask)[0], collate_fn=DS.cfn)
    test_dl  = DataLoader(DS, batch_size=4096, sampler=np.where(test_mask)[0], collate_fn=DS.cfn)

    model = TransformerDecoderBlock(vocab_dim = DS.NUM_LABEL,
                                    feature_dim = DS.num_precursor_feat,
                                    context_dim = DS.num_condition_feat, 
                                    num_heads = args.heads, 
                                    hidden_dim = args.hidden_dim, 
                                    hidden_layers = args.hidden_layers,
                                    positional_encoding = args.positional_encoding)

    trainer = SequenceTrainer(model, 1e-3)
    best_loss = 1e5
    count = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train(train_dl)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        if epoch % args.logging_interval == 0:
            valid_loss, valid_output = trainer.test(valid_dl)
            test_loss, test_output = trainer.test(test_dl)
            measure = []
            for sfx, loss, output in zip(['Valid','Test'], [valid_loss, test_loss], [valid_output, test_output]):
                pred = output['pred'].argmax(-1)
                n_data, l_seq = pred.shape
                label = output['label']
                mask = np.hstack([np.ones((n_data, 1), dtype=bool), (label != DS.EOS_LABEL)[..., :-1]]).reshape(-1)
                acc = accuracy_score(label.reshape(-1)[mask], pred.reshape(-1)[mask])
                f1_mi = f1_score(label.reshape(-1)[mask], pred.reshape(-1)[mask], average='micro')
                f1_ma = f1_score(label.reshape(-1)[mask], pred.reshape(-1)[mask], average='macro')
                hit_rxn = np.array([(p[m] != l[m]).sum() == 0 for p, l, m in zip(pred, label, mask)]).astype(float).mean()

                writer.add_scalar(f'Loss/{sfx}', loss, epoch)
                writer.add_scalar(f'ACC_prec/{sfx}', acc, epoch)
                writer.add_scalar(f'ACC_rxn/{sfx}', hit_rxn, epoch)
                writer.add_scalar(f'F1-micro_prec/{sfx}', f1_mi, epoch)
                writer.add_scalar(f'F1-macro_prec/{sfx}', f1_ma, epoch)
                measure.append([f1_mi, f1_ma, hit_rxn])

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
                print('{:5d} | {:8.4f} | {:8.4f} {:8.4f} {:8.4f} {:8.4f} | {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format(
                    epoch, train_loss, valid_loss, *measure[0], test_loss, *measure[1]))

if __name__ == '__main__':
    main(args)