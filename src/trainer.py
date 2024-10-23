import torch
import numpy as np

class BaseTrainer:
    def __init__(self, model, lr, device='cpu', crit=torch.nn.MSELoss(),
                 feat_keys = ['label'], output_keys = ['pred']):
        self.model = model
        self.model.to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.crit = crit
        self.device = device
        self.feat_keys = feat_keys
        self.output_keys = output_keys
    
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
    
    def _parse_output(self, batch, output):
        feat, info = batch
        if self._output is None:
            self._output = {'info':info}
            if feat is not None:
                for k in self.feat_keys:
                    if isinstance(feat[k], torch.Tensor):
                        self._output[k] = [feat[k].cpu().numpy()]
                    else:
                        self._output[k] = [np.array(feat[k])]
            for k, v in zip(self.output_keys, output):
                self._output[k] = [v.cpu().numpy()]
        else:
            self._output['info'].extend(info)
            if feat is not None:
                for k in self.feat_keys:
                    if isinstance(feat[k], torch.Tensor):
                        self._output[k].append(feat[k].cpu().numpy())
                    else:
                        self._output[k].append(np.array(feat[k]))
            for k, v in zip(self.output_keys, output):
                self._output[k].append(v.cpu().numpy())

    def _eval_batch(self, batch):
        pass

class AETrainer(BaseTrainer):
    def __init__(self, model, lr, device='cuda', 
                 crit=lambda x,y: torch.mean(torch.sum(torch.pow(x - y, 2), -1)),
                 feat_keys = ['label'], output_keys = ['latent','pred']):
        super().__init__(model, lr, device, crit, feat_keys, output_keys)

    def _eval_batch(self, batch, compute_loss=True, *args, **kwargs):
        feat, _ = batch
        latent, pred = self.model(feat)
        output = [latent.detach().cpu(), pred.detach().cpu()]
        if compute_loss:
            loss = self.crit(pred, feat)
            # sim = torch.sum(pred * pred, 1, keepdim=True) / (
                #   torch.sqrt(torch.sum(torch.square(pred), dim=-1, keepdim=True)) * \
                #   torch.sqrt(torch.sum(torch.square(feat), dim=-1, keepdim=True))
            # )
            return loss, output
        else:
            return output

class VAETrainer(BaseTrainer): # Classification
    def __init__(self, model, lr, device='cuda', 
                 crit=torch.nn.CrossEntropyLoss(reduction='none'),
                 feat_keys = ['label','weight','rxn_id'], 
                 output_keys = ['pred_has','pred_label','kld','mu','log_var','z']):
        super().__init__(model, lr, device, crit, feat_keys, output_keys)
    
    def _eval_batch(self, batch, compute_loss=True, beta=0.01):
        _feat, _ = batch
        precursor_feat = _feat['precursor_feat'].to(self.device)
        condition = torch.hstack([_feat['meta_feat'], _feat['condition_feat'][_feat['rxn_id']]]).to(self.device)
        edge_index = _feat['edge_index'].to(self.device)
        edge_attr = _feat['edge_attr'].to(self.device)
        weight = _feat['weight'].to(self.device)

        pred, kld, l, z = self.model(x=precursor_feat, condition=condition, edge_index=edge_index, edge_attr=edge_attr)
        mu, log_var = torch.chunk(l.detach().cpu(), 2, -1)
        pred_has = pred[:, 0]
        pred_lbl = pred[:, 1:] + ((weight > 0).long().float() - 1) * 1e5
        output = [1/(1 + torch.exp(-pred_has.detach())), 
                  torch.nn.functional.softmax(pred_lbl.detach(), dim=1), 
                  kld.detach(), mu, log_var.exp(), z.detach()]
        if compute_loss:
            is_last = torch.from_numpy(np.hstack([_feat['rxn_id'][1:] != _feat['rxn_id'][:-1], [True]])).bool()

            label = _feat['label'].to(self.device)
            label_has = label.sum(1)
            label_loc, label_index = torch.where(label)
            w = weight[label.bool()]
            w[pred_lbl[label_loc].argmax(1) != label_index] = 1.0

            bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(pred_has, label_has)[is_last]
#            focal_loss = (label_has[is_last] - 0.9).abs() * bce_loss * (1 - torch.exp(-bce_loss)) ** 2

            ce_loss = self.crit(pred_lbl[label_loc], label_index) * w
            loss = bce_loss.mean() + ce_loss.mean() + beta * kld.sum()
            return loss, output
        else:
            return output

class SequenceTrainer(BaseTrainer):
    def __init__(self, model, lr, device='cuda', 
                 crit=torch.nn.CrossEntropyLoss(reduction='none'),
                 feat_keys = ['label','weight','sequence_mask'], output_keys = ['pred']):
        super().__init__(model, lr, device, crit, feat_keys, output_keys)
    
    def _eval_batch(self, batch, compute_loss=True):
        _feat, _ = batch
        feat = {k:v.to(self.device) for k,v in _feat.items()}
        pred = self.model(**feat)
        B, S, L = pred.shape
        output = [pred.detach().cpu()]
        if compute_loss:
            _loss = self.crit(pred.view(B * S, -1), feat['label']) * feat['weight']
#            print(_loss.shape, feat['weight'].shape, feat['mask'].shape)
            loss = (_loss * feat['weight'][feat['sequence_mask']]).mean()
            return loss, output
        else:
            return output