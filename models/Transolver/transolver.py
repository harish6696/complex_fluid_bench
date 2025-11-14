import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch import Tensor
from .transolver_utils import Physics_Attention_Structured_Mesh_2D
from .transolver_utils import TransolverConfig
from transformers import PreTrainedModel
from utils import activation_func
from utils.grid_utils import twod_meshgrid
from utils.model_utils import CustomNorm

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, activation_fn_name='gelu', res=True):
        super(MLP, self).__init__()

        self.activation = activation_func.get_activation(activation_fn_name)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation_fn_name} not implemented")
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), self.activation)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), self.activation) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            config,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            activation_fn_name='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            H=85,
            W=85
    ):
        super().__init__()
        self.last_layer = last_layer
        #self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_1 = CustomNorm(config=config, num_channels=hidden_dim, array_length=3, channel_at_last_position=True)
        self.Attn = Physics_Attention_Structured_Mesh_2D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num, H=H, W=W)

        #self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ln_2 = CustomNorm(config=config, num_channels=hidden_dim, array_length=3, channel_at_last_position=True)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, activation_fn_name=activation_fn_name)
        if self.last_layer:
            #self.ln_3 = nn.LayerNorm(hidden_dim)
            self.ln_3 = CustomNorm(config=config, num_channels=hidden_dim, array_length=3, channel_at_last_position=True)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, **kwargs):
        fx = self.Attn(self.ln_1(fx, **kwargs)) + fx
        fx = self.mlp(self.ln_2(fx, **kwargs)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx, **kwargs))
        else:
            return fx

class Transolver(PreTrainedModel):

    main_input_name = "input_data"
    conditioning_input_name = "conditioning_input_data"
    config_class = TransolverConfig
     
    def __init__(self, config):
        super().__init__(config)    

        self.dimension = config.dimension
        self.transolver = self.build_Transolver()(config)

        self.post_init()

    def build_Transolver(self):
        if self.dimension == 2:
            return Transolver2D
        else:
            raise NotImplementedError("Invalid dimensionality. Only 2D ScOT implemented.")

    def forward(self, input_data: Tensor, **kwargs) -> Tensor:        
        if "conditioning_input_data" in kwargs:
            #NOTE: Conditioning data can be passed into a conv network before concatination with input_data.
            conditioning_input_data = kwargs["conditioning_input_data"]
            input_data = torch.cat([input_data, conditioning_input_data], dim=2)
        else:
            conditioning_input_data = None
        
        batch, input_seq, channels, *spatial = input_data.shape
        x= input_data.reshape(batch, input_seq * channels, *spatial)
        return self.transolver(x, **kwargs)

class Transolver2D(PreTrainedModel):
    def __init__(self,
                config,
                #  space_dim=1,
                #  n_layers=5,
                #  n_hidden=256,
                #  dropout=0.0,
                #  n_head=8,
                #  Time_Input=False,
                #  act='gelu',
                #  mlp_ratio=1,
                #  fun_dim=1,
                #  out_dim=1,
                #  slice_num=32,
                #  ref=8,
                #  unified_pos=False,
                #  H=85,
                #  W=85,
                 ):
        super().__init__(config)
        
        self.H = config.grid_resolution[0]
        self.W = config.grid_resolution[1]
        self.ref = config.ref
        self.unified_pos = config.unified_pos
        self.fun_dim = config.in_channels*config.sequence_info[0]
        self.out_dim = config.out_channels*config.sequence_info[1]
        self.n_hidden = config.latent_channels

        if self.unified_pos:
            pos = self.get_grid()
            self.register_buffer("pos", pos)
            self.preprocess = MLP(self.fun_dim + self.ref * self.ref, 
                                    self.n_hidden * 2, 
                                    self.n_hidden, 
                                    n_layers=0, 
                                    res=False, 
                                    activation_fn_name=config.activation_fn_name)
        else:
            self.preprocess = MLP(self.fun_dim + config.embedding_dim, 
                                    self.n_hidden * 2, self.n_hidden, 
                                    n_layers=0, 
                                    res=False, 
                                    activation_fn_name=config.activation_fn_name)

        self.Time_Input = config.time_input
        self.n_hidden = config.latent_channels
        self.space_dim = config.embedding_dim
        if config.time_input:
            self.time_fc = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), nn.SiLU(), nn.Linear(self.n_hidden, self.n_hidden))

        self.blocks = nn.ModuleList([Transolver_block(config=config, num_heads=config.num_heads, 
                                                      hidden_dim=self.n_hidden,
                                                      dropout=config.dropout,
                                                      activation_fn_name=config.activation_fn_name,
                                                      mlp_ratio=config.mlp_ratio,
                                                      out_dim=self.out_dim,
                                                      slice_num=config.slice_num,
                                                      H=config.grid_resolution[0],
                                                      W=config.grid_resolution[1],
                                                      last_layer=(_ == config.num_layers - 1))
                                     for _ in range(config.num_layers)])
        self.initialize_weights()
        #self.placeholder = nn.Parameter((1 / (self.n_hidden)) * torch.rand(self.n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos

    def forward(self, fx, **kwargs):
        #if self.config.coord_features:
            #coord_feat = twod_meshgrid(list(fx.shape), fx.device)
        fx_original_input_size = fx.shape[1]
        fx = fx.permute(0,2,3,1).reshape(fx.shape[0], -1, fx_original_input_size)  # B N C

        if self.unified_pos:
            x = self.pos.repeat(fx.shape[0], 1, 1, 1).reshape(fx.shape[0], self.H * self.W, self.ref * self.ref)
            fx = torch.cat((x, fx), -1)
        
        fx = self.preprocess(fx)

        for block in self.blocks:
            fx = block(fx, **kwargs)
        fx = fx.reshape(fx.shape[0], self.out_dim, self.H, self.W)
        return fx