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