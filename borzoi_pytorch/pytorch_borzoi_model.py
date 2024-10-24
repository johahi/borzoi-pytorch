# Copyright 2023 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================



from borzoi_pytorch.config_borzoi import BorzoiConfig
from transformers import PreTrainedModel
import torch.nn as nn
import torch
import math

from .pytorch_borzoi_utils import Residual, TargetLengthCrop 
from .pytorch_borzoi_transformer import Attention, FlashAttention


#torch.backends.cudnn.deterministic = True

#torch.set_float32_matmul_precision('high')
  
class ConvDna(nn.Module):
    def __init__(self):
        super(ConvDna, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels = 4,out_channels = 512, kernel_size = 15, padding="same")
        self.max_pool = nn.MaxPool1d(kernel_size = 2, padding = 0)

    def forward(self, x):
        return self.max_pool(self.conv_layer(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels,out_channels=None, kernel_size=1,
                 conv_type="standard"):
        super(ConvBlock, self).__init__()
        if conv_type == "separable":
            self.norm = nn.Identity()
            depthwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=in_channels, padding = 'same', bias = False)
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = nn.Identity()
        else:
            self.norm = nn.BatchNorm1d(in_channels, eps = 0.001)
            self.activation =nn.GELU(approximate='tanh')
            self.conv_layer = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size,
                padding='same')    
            
    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x

    
class Borzoi(PreTrainedModel):
    config_class = BorzoiConfig
    base_model_prefix = "borzoi"

    @staticmethod
    def from_hparams(**kwargs):
        return Borzoi(BorzoiConfig(**kwargs))
    
    
    def __init__(self, config):
        super(Borzoi, self).__init__(config)
        self.flashed = config.flashed if "flashed" in config.__dict__.keys() else False
        self.enable_mouse_head = config.enable_mouse_head   
        self.conv_dna = ConvDna()
        self._max_pool = nn.MaxPool1d(kernel_size = 2, padding = 0)
        self.res_tower = nn.Sequential(
            ConvBlock(in_channels = 512,out_channels = 608,kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 608,out_channels = 736,kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 736,out_channels = 896,kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 896,out_channels = 1056,kernel_size = 5),
            self._max_pool,
            ConvBlock(in_channels = 1056,out_channels = 1280,kernel_size = 5),
        )
        self.unet1 = nn.Sequential(
            self._max_pool,
            ConvBlock(in_channels = 1280,out_channels = config.dim,kernel_size = 5),
        )
        transformer = []
        for _ in range(config.depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim, eps = 0.001),
                    Attention(
                        config.dim,
                        heads = config.heads,
                        dim_key = config.attn_dim_key,
                        dim_value = config.attn_dim_value,
                        dropout = config.attn_dropout,
                        pos_dropout = config.pos_dropout,
                        num_rel_pos_features = 32
                    ) if not self.flashed else
                    FlashAttention(
                        config.dim,
                        heads = config.heads,
                        dropout = config.attn_dropout,
                        pos_dropout = config.pos_dropout,
                    ),
                    nn.Dropout(0.2))
                ),
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim, eps = 0.001),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate)
                )))
            )
        self.horizontal_conv0,self.horizontal_conv1 = ConvBlock(in_channels = 1280, out_channels = config.dim, kernel_size = 1),ConvBlock(in_channels = config.dim, out_channels = config.dim,kernel_size = 1)
        self.upsample = torch.nn.Upsample(scale_factor = 2)
        self.transformer = nn.Sequential(*transformer)
        self.upsampling_unet1 = nn.Sequential(
            ConvBlock(in_channels = config.dim, out_channels = config.dim,  kernel_size = 1),
            self.upsample,
        )
        self.separable1 = ConvBlock(in_channels = config.dim, out_channels = config.dim,  kernel_size = 3, conv_type = 'separable')
        self.upsampling_unet0 = nn.Sequential(
            ConvBlock(in_channels = config.dim,out_channels = config.dim,kernel_size = 1),
            self.upsample,
        )
        self.separable0 = ConvBlock(in_channels = config.dim, out_channels = config.dim,  kernel_size = 3, conv_type = 'separable')
        if config.return_center_bins_only:
            self.crop = TargetLengthCrop(6144)
        else:
            self.crop = TargetLengthCrop(16384 - 32) # as in Borzoi     
        self.final_joined_convs = nn.Sequential(
            ConvBlock(in_channels = config.dim, out_channels = 1920, kernel_size = 1),
            nn.Dropout(0.1),
            nn.GELU(approximate='tanh'),
        )
        self.human_head = nn.Conv1d(in_channels = 1920, out_channels = 7611, kernel_size = 1)
        if self.enable_mouse_head:
            self.mouse_head = nn.Conv1d(in_channels = 1920, out_channels = 2608, kernel_size = 1)
        self.final_softplus = nn.Softplus()


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()


    def forward(self, x, is_human = True, data_parallel_training = False):
        x = self.conv_dna(x)
        x_unet0 = self.res_tower(x)
        x_unet1 = self.unet1(x_unet0)
        x = self._max_pool(x_unet1)
        x_unet1 = self.horizontal_conv1(x_unet1)
        x_unet0 = self.horizontal_conv0(x_unet0)
        x = self.transformer(x.permute(0,2,1))
        x = x.permute(0,2,1)
        x = self.upsampling_unet1(x)
        x += x_unet1
        x = self.separable1(x)
        x = self.upsampling_unet0(x)
        x += x_unet0
        x = self.separable0(x)
        x = self.crop(x.permute(0,2,1))
        x = self.final_joined_convs(x.permute(0,2,1))
        # disable autocast for more precision in final layer
        with torch.cuda.amp.autocast(enabled=False):
            if data_parallel_training:
                # we need this to get gradients for both heads if doing DDP training
                if is_human:
                    human_out = self.final_softplus(self.human_head(x.float())) + 0 * self.mouse_head(x.float()).sum()
                    return human_out
                else:
                    mouse_out = self.final_softplus(self.mouse_head(x.float())) + 0 * self.human_head(x.float()).sum()
                    return mouse_out
            else:
                if is_human:
                    return self.final_softplus(self.human_head(x.float()))
                else:
                    return self.final_softplus(self.mouse_head(x.float()))