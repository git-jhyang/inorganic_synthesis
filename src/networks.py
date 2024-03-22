from torch import nn as tnn
from torch_geometric import nn as gnn
from typing import Union, List
import torch, os, pickle

class BaseNetwork(tnn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self._model_param = {}
    
    def save_model(self, path, file_name='output.model', overwrite=True):
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
        
