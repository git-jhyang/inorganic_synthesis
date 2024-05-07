from json import encoder
from turtle import forward
from typing import Dict, List
import torch_geometric as pyg
import torch, os, pickle

class BaseNetwork(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._model_param = kwargs
        self._dummy = torch.tensor([0])
    
    @property
    def device(self):
        return self._dummy.device

    def _save(self, path, file_name='output.model', overwrite=True):
        model_path = os.path.join(path, file_name)
        if not overwrite and os.path.isfile(model_path):
            raise FileExistsError(model_path)
        
        state_dict = {k:v.cpu().numpy() for k, v in self.state_dict().items()}
        
        with open(model_path,'wb') as f:
            pickle.dump({'model_param':self._model_param, 'state_dict':state_dict}, f)
        
    def _load(self, path, file_name='output.model', requires_grad=False):
        model_path = os.path.join(path, file_name)
        
        with open(model_path,'rb') as f:
            obj = pickle.load(f)
        model_param = obj['model_param']
        model_state_dict = {k:torch.from_numpy(v) for k,v in obj['state_dict'].items()}
        self.__init__(**model_param)
        self.load_state_dict(model_state_dict)
        self.requires_grad_(requires_grad=requires_grad)
        return self
    
class FCNNBlock(BaseNetwork):
    def __init__(self, 
                 input_dim:int, 
                 output_dim:int, 
                 hidden_dim:int = 32,
                 hidden_layers:int = 2,
                 batch_norm:bool = True, 
                 negative_slope:float = 0.1,
                 dropout:float = 0.0,
                 activation:str = 'LeakyReLU',
                 **kwargs): 
        super().__init__(input_dim = input_dim,
                         output_dim = output_dim,
                         hidden_dim = hidden_dim,
                         hidden_layers = hidden_layers,
                         batch_norm = batch_norm,
                         dropout = dropout,
                         activation = activation)
        
        try:
            activation = eval(f'torch.nn.{activation}({negative_slope})')
        except:
            activation = eval(f'torch.nn.{activation}()')

        self.embed_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            activation,
        )

        self.hidden_layer = torch.nn.ModuleList()
        for _ in range(hidden_layers):
            layer = [torch.nn.Linear(hidden_dim, hidden_dim)]
            if batch_norm:
                layer.append(torch.nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                layer.append(torch.nn.Dropout(dropout))
            layer.append(activation)
            self.hidden_layer.append(torch.nn.Sequential(*layer))
        
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h = self.embed_layer(x)
        for hidden_layer in self.hidden_layer:
            h = hidden_layer(h)
        out = self.output_layer(h)
        return out

class GraphAttentionBlock(BaseNetwork):
    def __init__(self,
                 input_dim:int, 
                 edge_dim:int,
                 output_dim:int, 
                 heads:int = 8,
                 negative_slope:float = 0.1,
                 hidden_dim:int = 32,
                 hidden_layers:int = 2,
                 dropout:float = 0.5,
                 activation:str = 'LeakyReLU',
                 **kwargs): 

        super().__init__(input_dim = input_dim,
                         edge_dim = edge_dim,
                         output_dim = output_dim,
                         heads = heads,
                         negative_slope = negative_slope,
                         hidden_dim = hidden_dim,
                         hidden_layers = hidden_layers,
                         dropout = dropout,
                         activation = activation)
        
        try:
            activation = eval(f'torch.nn.{activation}({negative_slope})')
        except:
            activation = eval(f'torch.nn.{activation}()')

        self.input_embed_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Dropout(dropout),
            activation,
        )
#        self.edge_embed_layer = torch.nn.Sequential(
#            torch.nn.Linear(edge_dim, hidden_dim),
#            torch.nn.Dropout(dropout),
#            activation,
#        )

        layers = []
        for _ in range(hidden_layers):
            layer = pyg.nn.GATv2Conv(in_channels=hidden_dim,
#                                     edge_dim=hidden_dim,
                                     edge_dim=edge_dim,
                                     out_channels=hidden_dim,
                                     heads=heads,
                                     concat=False,
                                     dropout=dropout,
                                     negative_slope=negative_slope)
            layers.append((layer, 'x, edge_index, edge_attr -> x'))

        self.graph_layer = pyg.nn.Sequential('x, edge_index, edge_attr', layers)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Dropout(dropout),
            activation,
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.input_embed_layer(x)
#        edge_attr = self.edge_embed_layer(edge_attr)
        h = self.graph_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        out = self.output_layer(h)
        return out

class GraphConvolutionBlock(BaseNetwork):
    def __init__(self,
                 input_dim:int, 
                 edge_dim:int,
                 output_dim:int, 
                 hidden_dim:int = 32,
                 hidden_layers:int = 2,
                 aggr:str='add',
#                 batch_norm:bool = True,
                 negative_slope:float = 0.1,
                 dropout = 0.3,
                 activation:str = 'LeakyReLU',
                 **kwargs):
        
        super().__init__(input_dim = input_dim,
                         edge_dim = edge_dim,
                         output_dim = output_dim,
                         hidden_dim = hidden_dim,
                         hidden_layers = hidden_layers,
                         aggr = aggr,
#                         batch_norm = batch_norm,
                         dropout = dropout,
                         activation = activation)

        try:
            activation = eval(f'torch.nn.{activation}({negative_slope})')
        except:
            activation = eval(f'torch.nn.{activation}()')

        self.input_embed_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
#            torch.nn.Dropout(dropout),
            activation,
        )
        self.edge_embed_layer = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
#            torch.nn.Dropout(dropout),
            activation,
        )

        layers = []
        for _ in range(hidden_layers):
            layer = pyg.nn.CGConv(channels=hidden_dim,
                                  dim=hidden_dim,
#                                  dim=edge_dim,
                                  aggr=aggr,
                                  batch_norm=False)

#            layer = pyg.nn.ResGatedGraphConv(in_channels = hidden_dim,
#                                             out_channels = hidden_dim,
#                                             act = activation,
##                                             edge_dim = edge_dim
#                                              edge_dim = hidden_dim
#                                             )
            layers.append((layer, 'x, edge_index, edge_attr -> x'))
#            if batch_norm:
            layers.append((torch.nn.Dropout(dropout), 'x -> x'))

        self.graph_layer = pyg.nn.Sequential('x, edge_index, edge_attr', layers)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim),
#            torch.nn.Dropout(dropout),
            torch.nn.BatchNorm1d(output_dim),
            activation,
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.input_embed_layer(x)
        edge_attr = self.edge_embed_layer(edge_attr)
        h = self.graph_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        out = self.output_layer(h)
        return out

class AutoEncoder(BaseNetwork):
    def __init__(self,
                 input_dim:int, 
                 latent_dim:int, 
                 output_dim:int=None,
                 encoder_hidden_dim:int = 32,
                 encoder_hidden_layers:int = 2,
                 decoder_hidden_dim:int = 32,
                 decoder_hidden_layers:int = 2,
                 batch_norm:bool = True, 
                 dropout:float = 0.0,
                 activation:str = 'LeakyReLU',
                 **kwargs): 
        super().__init__(input_dim = input_dim,
                         latent_dim = latent_dim,
                         output_dim = output_dim,
                         encoder_hidden_dim = encoder_hidden_dim,
                         encoder_hidden_layers = encoder_hidden_layers,
                         decoder_hidden_dim = decoder_hidden_dim,
                         decoder_hidden_layers = decoder_hidden_layers,
                         batch_norm = batch_norm,
                         dropout = dropout,
                         activation = activation)

        self.encoder = FCNNBlock(input_dim = input_dim, 
                                 output_dim = latent_dim, 
                                 hidden_dim = encoder_hidden_dim, 
                                 hidden_layers = encoder_hidden_layers, 
                                 batch_norm = batch_norm, 
                                 dropout = dropout, 
                                 activation = activation)
        self.decoder = FCNNBlock(input_dim = latent_dim, 
                                 output_dim = output_dim if isinstance(output_dim, int) else input_dim, 
                                 hidden_dim = decoder_hidden_dim, 
                                 hidden_layers = decoder_hidden_layers, 
                                 batch_norm = batch_norm, 
                                 dropout = dropout, 
                                 activation = activation)

    def forward(self, x, *args, **kwargs):
        l = self.encoder(x)
        y = torch.nn.Sigmoid()(self.decoder(l))
        return l, y

    def save(self, path, prefix, overwrite=True):
        self.encoder._save(path, f'{prefix}_encoder.model', overwrite)
        self.decoder._save(path, f'{prefix}_decoder.model', overwrite)

    def load_encoder(self, path, prefix, requires_grad=True):
        self.encoder._load(path, file_name=f'{prefix}_encoder.model', requires_grad=requires_grad)
        return self.encoder

    def load_decoder(self, path, prefix, requires_grad=True):
        self.decoder._load(path, file_name=f'{prefix}_decoder.model', requires_grad=requires_grad)
        return self.decoder

    def load(self, path, prefix, requires_grad=True):
        self.load_encoder(path, prefix, requires_grad)
        self.load_decoder(path, prefix, requires_grad)
        return self

class VAE(AutoEncoder):
    def __init__(self,
                 input_dim:int, 
                 latent_dim:int, 
                 output_dim:int=None,
                 encoder_hidden_dim:int = 32,
                 encoder_hidden_layers:int = 2,
                 decoder_hidden_dim:int = 32,
                 decoder_hidden_layers:int = 2,
                 batch_norm:bool = False, 
                 dropout:float = 0.4,
                 activation:str = 'LeakyReLU',
                 **kwargs): 
        super(AutoEncoder, self).__init__(input_dim = input_dim,
                                          latent_dim = latent_dim,
                                          output_dim = output_dim,
                                          encoder_hidden_dim = encoder_hidden_dim,
                                          encoder_hidden_layers = encoder_hidden_layers,
                                          decoder_hidden_dim = decoder_hidden_dim,
                                          decoder_hidden_layers = decoder_hidden_layers,
                                          batch_norm = batch_norm,
                                          dropout = dropout,
                                          activation = activation)
        
        self.encoder = FCNNBlock(input_dim = input_dim, 
                                 output_dim = latent_dim * 2, 
                                 hidden_dim = encoder_hidden_dim, 
                                 hidden_layers = encoder_hidden_layers, 
                                 batch_norm = batch_norm, 
                                 dropout = dropout, 
                                 activation = activation)
        
        self.decoder = FCNNBlock(input_dim = latent_dim, 
                                 output_dim = output_dim if isinstance(output_dim, int) else input_dim, 
                                 hidden_dim = decoder_hidden_dim, 
                                 hidden_layers = decoder_hidden_layers, 
                                 batch_norm = batch_norm, 
                                 dropout = dropout, 
                                 activation = activation)

    def reparameterization(self, l):
        mu, log_var = torch.chunk(l, 2, -1)
        z = mu + torch.randn_like(log_var) * torch.exp(0.5 * log_var) # mu + N * STD
        kld = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), -1, keepdim=True)
        return z, kld

    def forward(self, x, *args, **kwargs):
        l = self.encoder(x)
        z, kld = self.reparameterization(l)
        y = self.decoder(z)
        return y, kld, l, z
    
    def sampling(self, n, *args, **kwargs):
        z = torch.randn(n, self._model_param['latent_dim']).to(self.device)
        y = self.decoder(z)
        return y

class CVAE(VAE):
    def __init__(self,
                 input_dim:int, 
                 latent_dim:int, 
                 condition_dim:int,
                 output_dim:int=None,
                 encoder_hidden_dim:int = 32,
                 encoder_hidden_layers:int = 2,
                 decoder_hidden_dim:int = 32,
                 decoder_hidden_layers:int = 2,
                 batch_norm:bool = False, 
                 dropout:float = 0.4,
                 activation:str = 'LeakyReLU',
                 **kwargs): 
        super(AutoEncoder, self).__init__(input_dim = input_dim,
                                          latent_dim = latent_dim,
                                          condition_dim = condition_dim,
                                          output_dim = output_dim,                                          
                                          encoder_hidden_dim = encoder_hidden_dim,
                                          encoder_hidden_layers = encoder_hidden_layers,
                                          decoder_hidden_dim = decoder_hidden_dim,
                                          decoder_hidden_layers = decoder_hidden_layers,
                                          batch_norm = batch_norm,
                                          dropout = dropout,
                                          activation = activation)
        
        self.encoder = FCNNBlock(input_dim = input_dim, 
                                 output_dim = latent_dim * 2, 
                                 hidden_dim = encoder_hidden_dim, 
                                 hidden_layers = encoder_hidden_layers, 
                                 batch_norm = batch_norm, 
                                 dropout = dropout, 
                                 activation = activation)
        
        self.decoder = FCNNBlock(input_dim = latent_dim + condition_dim, 
                                 output_dim = output_dim if isinstance(output_dim, int) else input_dim,                                  
                                 hidden_dim = decoder_hidden_dim, 
                                 hidden_layers = decoder_hidden_layers, 
                                 batch_norm = batch_norm, 
                                 dropout = dropout, 
                                 activation = activation)

    def forward(self, x, condition, *args, **kwargs):
        l = self.encoder(x)
        z, kld = self.reparameterization(l)
        y = self.decoder(torch.concat([z, condition], -1))
        return y, kld, l, z
    
    def sampling(self, n, condition, *args, **kwargs):
        z = torch.randn(n, self._model_param['latent_dim']).to(self.device)
        y = self.decoder(torch.concat([z, condition.repeat((n, 1)) ], -1))
        return y

class GraphCVAE(VAE):
    def __init__(self,
                 input_dim:int, 
                 latent_dim:int, 
                 condition_dim:int,
                 edge_dim:int,
                 output_dim:int=None,
                 graph:str='conv',
                 aggr:str='mean',
                 heads:int = 8,
                 negative_slope:float=0.1,
                 encoder_hidden_dim:int = 32,
                 encoder_hidden_layers:int = 2,
                 decoder_hidden_dim:int = 32,
                 decoder_hidden_layers:int = 2,
                 batch_norm:bool = True, 
                 dropout:float = 0.5,
                 activation:str = 'LeakyReLU',
                 **kwargs): 
        
        if graph.lower().startswith('conv'):
            super(AutoEncoder, self).__init__(input_dim = input_dim,
                                              latent_dim = latent_dim,
                                              output_dim = output_dim,
                                              edge_dim = edge_dim,
                                              graph = 'convolution',
                                              condition_dim = condition_dim,
                                              encoder_hidden_dim = encoder_hidden_dim,
                                              encoder_hidden_layers = encoder_hidden_layers,
                                              decoder_hidden_dim = decoder_hidden_dim,
                                              decoder_hidden_layers = decoder_hidden_layers,
                                              batch_norm = batch_norm,
                                              aggr = aggr,
                                              activation = activation)

            BLOCK = GraphConvolutionBlock
        elif graph.lower().startswith('atte'):
            super(AutoEncoder, self).__init__(input_dim = input_dim,
                                              latent_dim = latent_dim,
                                              output_dim = output_dim,
                                              edge_dim = edge_dim,
                                              graph = 'attention',
                                              condition_dim = condition_dim,
                                              encoder_hidden_dim = encoder_hidden_dim,
                                              encoder_hidden_layers = encoder_hidden_layers,
                                              decoder_hidden_dim = decoder_hidden_dim,
                                              decoder_hidden_layers = decoder_hidden_layers,
                                              heads = heads,
                                              dropout = dropout,
                                              negative_slope = negative_slope,
                                              activation = activation)
            BLOCK = GraphAttentionBlock
        else:
            raise SyntaxError(f'`{graph}` is not supported graph type')
        
        self.encoder = BLOCK(input_dim=input_dim,
                             edge_dim=edge_dim,
                             output_dim=latent_dim * 2,
                             hidden_dim=encoder_hidden_dim,
                             hidden_layers=encoder_hidden_layers,
                             batch_norm=batch_norm,
                             heads=heads,
                             negative_slope=negative_slope,
                             dropout=dropout,
                             activation=activation)
         
        self.decoder = BLOCK(input_dim=latent_dim + condition_dim,
                             edge_dim=edge_dim,
                             output_dim = output_dim if isinstance(output_dim, int) else input_dim,                              
                             hidden_dim=encoder_hidden_dim,
                             hidden_layers=encoder_hidden_layers,
                             batch_norm=batch_norm,
                             heads=heads,
                             negative_slope=negative_slope,
                             dropout=dropout,
                             activation=activation)

    def forward(self, x, edge_index, edge_attr, condition, *args, **kwargs):
        l = self.encoder(x = x, 
                         edge_index = edge_index, 
                         edge_attr = edge_attr)
        
        z, kld = self.reparameterization(l)

        y = self.decoder(x = torch.concat([z, condition], -1), 
                         edge_index = edge_index, 
                         edge_attr = edge_attr)
        return y, kld, l, z

    def sampling(self, n, edge_index, edge_attr, condition, *args, **kwargs):
        y = []
        for _ in range(n):
            z = torch.randn(condition.shape[0], self._model_param['latent_dim']).to(self.device)
            y.append(self.decoder(x = torch.concat([z, condition], -1),
                                  edge_index = edge_index,
                                  edge_attr = edge_attr))
        return 