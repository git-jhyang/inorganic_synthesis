import os, pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.networks import CVAE
from src.data import ConditionDataset
from src.trainer import VAETrainer
from src.feature import feature_to_ligand_index
from src.utils import squared_error
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--encoder_hidden_dim', default=32)
parser.add_argument('--encoder_hidden_layers', default=2)
parser.add_argument('--decoder_hidden_dim', default=32)
parser.add_argument('--decoder_hidden_layers', default=2)
parser.add_argument('--latent_dim', default=16)
parser.add_argument('--batch_size', default=256)
parser.add_argument('--target_feature_type', default='active_composit')

args = parser.parse_args()

torch.set_float32_matmul_precision('high')

def main(args):
    identifier = '{:s}_batch_{:04d}_mdim_{:02d}_{:03d}_{:1d}_{:03d}_{:1d}'.format(
        args.target_feature_type, args.batch_size, args.latent_dim, 
        args.encoder_hidden_dim, args.encoder_hidden_layers, 
        args.decoder_hidden_dim, args.decoder_hidden_layers, 
    )

    output_path = f'/home/jhyang/WORKSPACES/MODELS/isyn/VAE/{identifier}'
    if os.path.isdir(output_path):
        for i in range(100):
            if not os.path.isdir(output_path + f'_case_{i:02d}'):
                break
        output_path += f'_case_{i:02d}'
    os.makedirs(output_path, exist_ok=True)
    writer = SummaryWriter(output_path)

    DS = ConditionDataset(target_feat_type=args.target_feature_type)
    DS.from_file(data_path='./data/unique_reaction.pkl.gz')
    DS.to('cuda')

    years = np.array([d.year for d in DS])
    test_mask = years > 2018
    train_mask = years < 2017
    valid_mask = ~train_mask & ~test_mask

    train_dl = DataLoader(DS, batch_size=128, sampler=SubsetRandomSampler(np.where(train_mask)[0]), collate_fn=DS.cfn)
    valid_dl = DataLoader(DS, batch_size=4096, sampler=np.where(valid_mask)[0], collate_fn=DS.cfn)
    test_dl = DataLoader(DS, batch_size=4096, sampler=np.where(test_mask)[0], collate_fn=DS.cfn)

    model = CVAE(input_dim = DS.num_precursor_feat, 
                 latent_dim = args.latent_dim, 
                 condition_dim = DS.num_metal_feat + DS.num_target_feat,
                 encoder_hidden_dim = args.encoder_hidden_dim, 
                 encoder_hidden_layers = args.encoder_hidden_layers, 
                 decoder_hidden_dim = args.decoder_hidden_dim, 
                 decoder_hidden_layers = args.decoder_hidden_layers,
    )

    trainer = VAETrainer(model, 1e-4, device='cuda')
    best_loss = 1e5
    count = 0
    for epoch in range(5000):
        train_loss = trainer.train(train_dl)
        writer.add_scalar('Loss/Train', train_loss, epoch+1)

        if (epoch + 1)%5 == 0:
            valid_loss, valid_output = trainer.test(valid_dl)
            v_t = feature_to_ligand_index(valid_output['input'])
            v_p = feature_to_ligand_index(valid_output['pred'])
            v_acc = accuracy_score(v_t, v_p)
            v_f1 = f1_score(v_t, v_p, average='micro')
            v_mse = squared_error(valid_output['input'], valid_output['pred'])
            writer.add_scalar('Loss/Valid', valid_loss, epoch+1)
            writer.add_scalar('KLD/Valid', valid_output['kld'].sum(), epoch+1)
            writer.add_scalar('ACC/Valid', v_acc, epoch + 1)
            writer.add_scalar('F1/Valid', v_f1, epoch + 1)
            writer.add_scalar('MSE/Valid', v_mse, epoch + 1)
    
            test_loss, test_output = trainer.test(test_dl)
            t_t = feature_to_ligand_index(test_output['input'])
            t_p = feature_to_ligand_index(test_output['pred'])
            t_acc = accuracy_score(t_t, t_p)
            t_f1 = f1_score(t_t, t_p, average='micro')
            t_mse = squared_error(test_output['input'], test_output['pred'])
            writer.add_scalar('Loss/Test', test_loss, epoch+1)
            writer.add_scalar('KLD/Test', test_output['kld'].sum(), epoch+1)
            writer.add_scalar('ACC/Test', t_acc, epoch+1)
            writer.add_scalar('F1/Test', t_f1, epoch+1)
            writer.add_scalar('MSE/Test', t_mse, epoch+1)

            if (epoch+1) % 100 == 0:
                trainer.model.save(output_path, f'{epoch+1:05d}.model')
                with open(os.path.join(output_path, f'{epoch+1:05d}.output.valid.pkl'),'wb') as f:
                    pickle.dump(valid_output, f)
                with open(os.path.join(output_path, f'{epoch+1:05d}.output.test.pkl'),'wb') as f:
                    pickle.dump(test_output, f)

            if valid_loss < best_loss:
                count = 0
                best_loss = valid_loss
                writer.add_scalar('Loss/BestValid', valid_loss, epoch+1)
                trainer.model.save(output_path, 'best.model')
                with open(os.path.join(output_path, 'best.output.valid.pkl'),'wb') as f:
                    pickle.dump(valid_output, f)
                with open(os.path.join(output_path, 'best.output.test.pkl'),'wb') as f:
                    pickle.dump(test_output, f)
                with open(os.path.join(output_path, 'best.epoch.txt'),'w') as f: 
                    f.write(str(epoch+1))
            else:
                count += 5
                if count > 200:
                    return

if __name__ == '__main__':
    import time
    t1 = time.time()

    main(args)

    print('{:10.2f} min - {:s}_batch_{:04d}_mdim_{:02d}_{:03d}_{:1d}_{:03d}_{:1d}'.format(
        (time.time() - t1) / 60.0,
        args.target_feature_type, args.batch_size, args.latent_dim, 
        args.encoder_hidden_dim, args.encoder_hidden_layers, 
        args.decoder_hidden_dim, args.decoder_hidden_layers, 
    ))