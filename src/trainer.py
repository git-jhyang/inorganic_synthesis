import torch

class DefaultTrainer:
    def __init__(self, model, lr, crit, device='cpu'):
        self.model = model
        self.model.to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.crit = crit
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
        return test_loss/len(dataloader), self.output
        
    def predict(self, dataloader):
        self.model.eval()
        self._init_output()
        with torch.no_grad():
            for batch in dataloader:
                output = self._eval_batch(batch, compute_loss=False)
                self._parse_output(batch, output)
        return self.output

    def _init_output(self):
        self.output = {k:None for k in self._output_keys}

class AETrainer(DefaultTrainer):
    def __init__(self, model, lr, device='cuda'):
        super(AETrainer, self).__init__(model, lr, None, device)
        self._output_keys = ['target','pred','latent', 'info']

    def _eval_batch(self, batch, compute_loss=True):
        feat, _ = batch
        latent, pred = self.model(feat)
        if compute_loss:
            loss = torch.mean(torch.sum(torch.pow(pred - feat, 2), -1))
            # sim = torch.sum(pred * pred, 1, keepdim=True) / (
                #   torch.sqrt(torch.sum(torch.square(pred), dim=-1, keepdim=True)) * \
                #   torch.sqrt(torch.sum(torch.square(feat), dim=-1, keepdim=True))
            # )
            return loss, [latent.detach().cpu().numpy(), pred.detach().cpu().numpy()]
        else:
            return [latent.detach().cpu().numpy(), pred.detach().cpu().numpy()]
            
    def _parse_output(self, batch, output):
        feat, info = batch
        latent, pred = output
        for label, value in zip(self._output_keys, [feat.cpu().numpy(), pred, latent, info]):
            if self.output[label] is None:
                self.output[label] = value
            else:
                self.output[label] = np.vstack([self.output[label], value])

class VAETrainer(DefaultTrainer):
    def __init__(self):
        pass