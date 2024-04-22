import torch
import numpy as np
from src.feature import feature_to_ligand_index

class BaseTrainer:
    def __init__(self, model, lr, device='cpu'):
        self.model = model
        self.model.to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.device = device
    
    def train(self, dataloader):
        self.model.train()
        train_loss = 0
        for batch in dataloader:
            loss, _ = self._eval_batch(batch)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            train_loss += loss
        return train_loss/len(dataloader)
    
    def test(self, dataloader):
        self.model.eval()
        self._init_output()
        test_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                loss, output = self._eval_batch(batch)
                self._parse_output(batch, output)
                test_loss += loss
        return test_loss/len(dataloader), self._output
        
    def predict(self, dataloader):
        self.model.eval()
        self._init_output()
        with torch.no_grad():
            for batch in dataloader:
                output = self._eval_batch(batch, compute_loss=False)
                self._parse_output(batch, output)
        return self._output

    def _init_output(self):
        self._output = None
    
    def _eval_batch(self, batch):
        pass

    def _parse_output(self, batch, output):
        pass

class AETrainer(BaseTrainer):
    def __init__(self, model, lr, device='cuda'):
        super(AETrainer, self).__init__(model, lr, device)

    def _eval_batch(self, batch, compute_loss=True):
        feat, _ = batch
        latent, pred = self.model(feat)
        output = [latent.detach().cpu().numpy(), pred.detach().cpu().numpy()]
        if compute_loss:
            loss = torch.mean(torch.sum(torch.pow(pred - feat, 2), -1))
            # sim = torch.sum(pred * pred, 1, keepdim=True) / (
                #   torch.sqrt(torch.sum(torch.square(pred), dim=-1, keepdim=True)) * \
                #   torch.sqrt(torch.sum(torch.square(feat), dim=-1, keepdim=True))
            # )
            return loss, output
        else:
            return output
            
    def _parse_output(self, batch, output):
        feat, info = batch
        latent, pred = output
        if self._output is None:
            self._output = {
                'info':info,
                'target':feat.cpu().numpy(),
                'pred':pred,
                'latent':latent,
            }
        else:
            self._output['info'].extend(info)
            self._output['target'] = np.vstack([self._output['target'], feat.cpu().numpy()])
            self._output['pred']   = np.vstack([self._output['pred'], pred])
            self._output['latent'] = np.vstack([self._output['latent'], latent])

class VAETrainer(BaseTrainer):
    def __init__(self, model, lr, device='cuda'):
        super(VAETrainer, self).__init__(model, lr, device)
    
    def _eval_batch(self, batch, compute_loss=True):
        feat, _ = batch
        pred, kld = self.model(**feat)
        output = [kld.detach().cpu().numpy(), pred.detach().cpu().numpy()]
        if compute_loss:
            mse = torch.mean(torch.sum(torch.square(feat['x'] - pred), -1))
            loss = mse + kld.sum()
            return loss, output
        else:
            return output
        
    def _parse_output(self, batch, output):
        kld, pred = output
        feat, info = batch
        if self._output is None:
            self._output = {
                'info': info,
                'kld': kld,
                'input': feat['x'].cpu().numpy(),
                'pred': pred,
            }
        else:
            self._output['info'].extend(info)
            self._output['kld'] = np.vstack([self._output['kld'], kld])
            self._output['input'] = np.vstack([self._output['input'], feat['x'].cpu().numpy()])
            self._output['pred'] = np.vstack([self._output['pred'], pred])

    def predict(self, dataloader, n_samples=1000):
        self.model.eval()
        self._init_output()
        with torch.no_grad():
            for batch in dataloader:
                for feat, info in zip(*batch):
                    pred = self.model.sampling(n_samples, **feat)
                    self._parse_output(None, [0, pred.cpu().numpy()[np.newaxis, ...], info])
        return self._output