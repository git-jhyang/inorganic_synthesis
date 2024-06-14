import torch
import numpy as np

class BaseTrainer:
    def __init__(self, model, lr, device='cpu', crit=torch.nn.MSELoss()):
        self.model = model
        self.model.to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.crit = crit
        self.device = device
    
    def train(self, dataloader, *args, **kwargs):
        self.model.train()
        train_loss = 0
        for batch in dataloader:
            loss, _ = self._eval_batch(batch, *args, **kwargs)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            train_loss += loss.item()
        return train_loss/len(dataloader)
    
    def test(self, dataloader, *args, **kwargs):
        self.model.eval()
        self._init_output()
        test_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                loss, output = self._eval_batch(batch, *args, **kwargs)
                self._parse_output(batch, output)
                test_loss += loss.item()
        return test_loss/len(dataloader), self._output
        
    def predict(self, dataloader, *args, **kwargs):
        self.model.eval()
        self._init_output()
        with torch.no_grad():
            for batch in dataloader:
                output = self._eval_batch(batch, compute_loss=False, *args, **kwargs)
                self._parse_output(batch, output)
        return self._output

    def _init_output(self):
        self._output = None
    
    def _eval_batch(self, batch):
        pass

    def _parse_output(self, batch, output):
        pass

class AETrainer(BaseTrainer):
    def __init__(self, model, lr, device='cuda', 
                 crit=lambda x,y: torch.mean(torch.sum(torch.pow(x - y, 2), -1))):
        super().__init__(model, lr, device, crit)

    def _eval_batch(self, batch, compute_loss=True, *args, **kwargs):
        feat, _ = batch
        latent, pred = self.model(feat)
        output = [latent.detach().cpu().numpy(), pred.detach().cpu().numpy()]
        if compute_loss:
            loss = self.crit(pred, feat)
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
                'pred':pred,
                'latent':latent,
            }
            if feat is not None:
                self._output.update({'input':feat['x'].cpu().numpy()})
        else:
            self._output['info'].extend(info)
            self._output['pred']   = np.vstack([self._output['pred'], pred])
            self._output['latent'] = np.vstack([self._output['latent'], latent])
            if feat is not None:
                self._output['input'] = np.vstack([self._output['input'], feat['x'].cpu().numpy()])

class VAETrainer(BaseTrainer):
    def __init__(self, model, lr, device='cuda', 
                 crit=lambda x,y: torch.mean(torch.sum(torch.pow(x - y, 2), -1))):
        super().__init__(model, lr, device, crit)
    
    def _eval_batch(self, batch, compute_loss=True, beta=0.1):
        feat, _ = batch
        pred, kld, l, z = self.model(**feat)
        pred = torch.nn.Hardsigmoid()(pred)
        mu, log_var = torch.chunk(l.detach().cpu(), 2, -1)
        output = [pred.detach().cpu().numpy(), kld.detach().cpu().numpy(), mu.numpy(), log_var.exp().numpy(), z.detach().cpu().numpy()]
        if compute_loss:
            mse = self.crit(feat['x'], pred)
#            celoss = torch.nn.CrossEntropyLoss()()
            loss = mse + beta * kld.sum()
            return loss, output
        else:
            return output
        
    def _parse_output(self, batch, output):
        pred, kld, mu, var, z = output
        feat, info = batch
        if self._output is None:
            self._output = {
                'info': info,
                'kld': np.array(kld),
                'pred': np.array(pred),
                'mu': np.array(mu),
                'var': np.array(var),
                'z': np.array(z),
            }
            if isinstance(feat, dict) and 'input' in feat.keys() and isinstance(feat['x'], torch.Tensor):
                self._output.update({
                    'input': feat['x'].cpu().numpy(),
                    'label': feat['label'].cpu().numpy().reshape(-1),
                })
        else:
            self._output['info'].extend(info)
            self._output['kld'] = np.vstack([self._output['kld'], kld])
            self._output['pred'] = np.vstack([self._output['pred'], pred])
            self._output['mu'] = np.vstack([self._output['mu'], mu])
            self._output['var'] = np.vstack([self._output['var'], var])
            self._output['z'] = np.vstack([self._output['z'], z])
            if isinstance(feat, dict) and 'input' in feat.keys() and isinstance(feat['x'], torch.Tensor):
                self._output['input'] = np.vstack([self._output['input'], feat['x'].cpu().numpy()])
                self._output['label'] = np.hstack([self._output['label'], feat['label'].cpu().numpy().reshape(-1)])

    def predict(self, dataloader, n_samples=1000):
        self.model.eval()
        self._init_output()
        with torch.no_grad():
            for batch in dataloader:
                for feat, info in zip(*batch):
                    pred = self.model.sampling(n_samples, **feat)
                    self._parse_output([None, info], [pred.cpu().numpy(), *np.zeros((4,1,1))])
        return self._output

class VAEClassTrainer(VAETrainer):
    def __init__(self, model, lr, device='cuda', crit=torch.nn.CrossEntropyLoss()):
        super().__init__(model, lr, device)
    
    def _eval_batch(self, batch, compute_loss=True, beta=0.1):
        feat, _ = batch
        pred, kld, l, z = self.model(**feat)
        mu, log_var = torch.chunk(l.detach().cpu(), 2, -1)
        output = [pred.detach().cpu().numpy(), kld.detach().cpu().numpy(), mu.numpy(), log_var.exp().numpy(), z.detach().cpu().numpy()]
        if compute_loss:
            celoss = self.crit(pred, feat['label'])
#            mse = torch.mean(torch.sum(torch.square(feat['x'] - pvec), -1))
            loss = celoss + beta * kld.sum()
            return loss, output
        else:
            return output
        
class SequenceTrainer(BaseTrainer):
    def __init__(self, model, lr, device='cuda', crit=torch.nn.CrossEntropyLoss(reduction='none')):
        super().__init__(model, lr, device, crit)
    
    def _eval_batch(self, batch, compute_loss=True):
        _feat, _ = batch
        feat = {k:v.to(self.device) for k,v in _feat.items()}
        pred = self.model(**feat)
        if compute_loss:
            _loss = self.crit(pred.view(feat['label'].shape[0], -1), feat['label'])[feat['mask']]
#            print(_loss.shape, feat['weight'].shape, feat['mask'].shape)
            loss = (_loss * feat['weight'][feat['mask']]).mean()
            return loss, pred.detach().cpu().numpy()
        else:
            return pred.detach().cpu().numpy()
    
    def _parse_output(self, batch, output):
        feat, info = batch
        if self._output is None:
            self._output = {
                'info' : info,
                'pred' : output
            }
            if feat['weight'] is not None:
                n = feat['context'].shape[0]
                self._output.update({
                    'label': feat['label'].cpu().numpy().reshape(n, -1),
                    'weight': feat['weight'].cpu().numpy().reshape(n, -1)[:, 0],
                })
        else:
            self._output['info'].extend(info)
            self._output['pred'] = np.vstack([self._output['pred'], output])
            if feat['weight'] is not None:
                n = feat['context'].shape[0]
                self._output['label'] = np.vstack([self._output['label'], feat['label'].cpu().numpy().reshape(n, -1)])
                self._output['weight'] = np.hstack([self._output['weight'], feat['weight'].cpu().numpy().reshape(n, -1)[:,0]])