from torch import nn as tnn
from torch_geometric import nn as gnn
from typing import Union, List
import torch, os, pickle

class BaseNetwork(tnn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self._model_param = {}
    
    def _save(self, path, file_name='output.model', overwrite=True):
        model_path = os.path.join(path, file_name)
        if not overwrite and os.path.isfile(model_path):
            raise FileExistsError(model_path)
        
        state_dict = {k:v.cpu().numpy() for k, v in self.state_dict().items()}
        
        with open(model_path,'wb') as f:
            pickle.dump({'model_param':self._model_param, 'state_dict':state_dict}, f)
        
    def _load(self, cls, path, file_name='output.model', requires_grad=False):
        model_path = os.path.join(path, file_name)
        
        with open(model_path,'rb') as f:
            obj = pickle.load(f)
        model_param = obj['model_param']
        model_state_dict = {k:torch.from_numpy(v) for k,v in obj['state_dict'].items()}
        cls.__init__(**model_param)
        cls.load_state_dict(model_state_dict)
        cls.requires_grad_(requires_grad=requires_grad)
        return cls
    
class DNNBlock(BaseNetwork):
    def __init__(self, 
                 input_dim:int, 
                 output_dim:int, 
                 hidden_dim:int = 32,
                 hidden_layers:int = 2,
                 batch_norm:bool = True, 
                 dropout:float = 0,
                 activation:str = 'LeakyReLU',
                 **kwargs): 
        super(DNNBlock, self).__init__()
        self._model_param = {
                 'input_dim':input_dim,
                 'output_dim':output_dim,
                 'hidden_dim':hidden_dim,
                 'hidden_layers':hidden_layers,
                 'batch_norm':batch_norm,
                 'dropout':dropout,
                 'activation':activation
        }
        
        self.embed_layer = tnn.Linear(input_dim, hidden_dim)
        
        self.hidden_layer = tnn.ModuleList()
        for _ in range(hidden_layers):
            layer = [tnn.Linear(hidden_dim, hidden_dim)]
            if batch_norm:
                layer.append(tnn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                layer.append(tnn.Dropout(dropout))
            layer.append(eval(f'tnn.{activation}()'))
            self.hidden_layer.append(tnn.Sequential(*layer))
        
        self.output_layer = tnn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h = self.embed_layer(x)
        for hidden_layer in self.hidden_layer:
            h = hidden_layer(h)
        out = self.output_layer(h)
        return out

class AutoEncoder(BaseNetwork):
    def __init__(self,
                 input_dim:int, 
                 latent_dim:int, 
                 encoder_hidden_dim:int = 32,
                 encoder_hidden_layers:int = 2,
                 decoder_hidden_dim:int = 32,
                 decoder_hidden_layers:int = 2,
                 batch_norm:bool = True, 
                 dropout:float = 0,
                 activation:str = 'LeakyReLU',
                 **kwargs): 
        super(AutoEncoder, self).__init__()
        self._model_param = {
            'input_dim':input_dim,
            'latent_dim':latent_dim,
            'encoder_hidden_dim':encoder_hidden_dim,
            'encoder_hidden_layers':encoder_hidden_layers,
            'decoder_hidden_dim':decoder_hidden_dim,
            'decoder_hidden_layers':decoder_hidden_layers,
            'batch_norm':batch_norm,
            'dropout':dropout,
            'activation':activation,
        }
        self.encoder = DNNBlock(input_dim, latent_dim, encoder_hidden_dim, 
                                encoder_hidden_layers, batch_norm, dropout, activation)
        self.decoder = DNNBlock(latent_dim, input_dim, decoder_hidden_dim, 
                                decoder_hidden_layers, batch_norm, dropout, activation)
        self._dummy = tnn.Parameter(torch.empty(0))

    def forward(self, x):
        l = self.encoder(x)
        y = torch.nn.Sigmoid()(self.decoder(l))
        return l, y

    def save(self, path, prefix, overwrite=True):
        self._save(path, f'{prefix}_full.model', overwrite)
        self.encoder._save(path, f'{prefix}_encoder.model', overwrite)
        self.decoder._save(path, f'{prefix}_decoder.model', overwrite)

    def load_encoder(self, path, prefix, requires_grad=True):
        pass

    def load_decoder(self, path, prefix, requires_grad=True):
        pass

    def load(self, path, prefix, requires_grad=True):
        pass

class VAE(AutoEncoder):
    def __init__(self,
                 input_dim:int, 
                 latent_dim:int, 
                 encoder_hidden_dim:int = 32,
                 encoder_hidden_layers:int = 2,
                 decoder_hidden_dim:int = 32,
                 decoder_hidden_layers:int = 2,
                 batch_norm:bool = True, 
                 dropout:float = 0,
                 activation:str = 'LeakyReLU',
                 **kwargs): 
        super(VAE, self).__init__(input_dim, latent_dim, encoder_hidden_dim, encoder_hidden_layers,
                                  decoder_hidden_dim, decoder_hidden_layers, batch_norm, 
                                  dropout, activation, **kwargs)
        self.encoder = DNNBlock(input_dim, latent_dim * 2, encoder_hidden_dim, 
                                encoder_hidden_layers, batch_norm, dropout, activation)

    def reparameterization(self, l):
        mu, log_var = torch.chunk(l, 2, -1)
        z = mu + torch.randn_like(log_var) * torch.exp(0.5 * log_var) # mu + N * STD
        kld = 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return z, kld

    def forward(self, x):
        l = self.encoder(x)
        z, kld = self.reparameterization(l)
        y = self.decoder(z)
        return y, kld
    
    def sampling(self, n):
        z = torch.randn(n, self._parameters['latent_dim']).to(self._dummy.device)
        y = self.decoder(z)
        return y

class CVAE(VAE):
    def __init__(self,
                 input_dim:int, 
                 latent_dim:int, 
                 condition_dim:int,
                 encoder_hidden_dim:int = 32,
                 encoder_hidden_layers:int = 2,
                 decoder_hidden_dim:int = 32,
                 decoder_hidden_layers:int = 2,
                 batch_norm:bool = True, 
                 dropout:float = 0,
                 activation:str = 'LeakyReLU',
                 **kwargs): 
        super(CVAE, self).__init__(input_dim, latent_dim, encoder_hidden_dim, encoder_hidden_layers,
                                   decoder_hidden_dim, decoder_hidden_layers, batch_norm, 
                                   dropout, activation, **kwargs)
        self._model_param.update({'condition_dim':condition_dim})
        self.decoder = DNNBlock(latent_dim + condition_dim, input_dim, encoder_hidden_dim, 
                                encoder_hidden_layers, batch_norm, dropout, activation)

    def make_conditional_input(self, z, c):
        return torch.concat([c, z], 0)

    def forward(self, x, c):
        l = self.encoder(x)
        z, kld = self.reparameterization(l)
        y = self.decoder(self.make_conditional_input(z, c))
        return y, kld
    
    def sampling(self, n, c):
        z = torch.randn(n, self._parameters['latent_dim']).to(self._dummy.device)
        y = self.decoder(self.make_conditional_input(z, c))
        return y