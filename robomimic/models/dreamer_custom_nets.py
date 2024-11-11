import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InitNet(nn.Module):
    """
    Input: initial embedding
    Output: initial h_0, z_0
    """
    def __init__(
            self,
            embed_dim,
            stoch_dim,
            deter_dim,
            hidden_dim,
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            layer_norm=nn.LayerNorm,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        super(InitNet, self).__init__()

        self.mlp_embed = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_mlp = nn.Linear(hidden_dim, deter_dim + stoch_dim)

        self._hidden_dim = hidden_dim
        self._output_dim = deter_dim
        self._layer_norm = layer_norm(hidden_dim, eps=1e-3)
        self._output_dim = deter_dim + stoch_dim # h_0 and z_0

    def output_shape(self, input_shape=None):
        return [self._output_dim]

    def forward(self, embed):
        x = self.mlp_embed(embed)
        x = self._layer_norm(x)
        post_in = F.elu(x)
        post = self.post_mlp(post_in)
        chunk_sizes = [deter_dim, stoch_dim]
        post = torch.split(post, chunk_sizes)
        h_t, z_t = post[0], post[1]
        print("h_t: ", h_t.shape)
        print("z_t: ", z_t.shape)
        return h_t, z_t


class ImageDecoderResnet(nn.Module):

    def __init__(self, 
                 shape, 
                 depth=96, 
                 blocks=0, 
                 resize="stride", 
                 minres=4, 
                 sigmoid=False, 
                 ):
        super(ImageDecoderResnet, self).__init__()
        self.shape = shape
        self.depth = depth
        self.blocks = blocks
        self.resize = resize
        self.minres = minres
        self.sigmoid = sigmoid
        self.kw = {
            "winit": "normal", 
            "fan": "avg", 
            "outscale": 1.0,
            "minres": 4,
            }

        stages = int(np.log2(self.shape[-2]) - np.log2(self.minres))
        depth = self.depth * 2 ** (stages - 1)
        self.in_linear = nn.Linear(np.prod(self.shape), self.minres * self.minres * depth)

        layers = []
        for i in range(stages):
            for j in range(self.blocks):
                layers.append(nn.Conv2d(depth, depth, 3, padding=1, **self.kw))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Conv2d(depth, depth, 3, padding=1, **self.kw))
                layers.append(nn.ReLU(inplace=True))

            if i == stages - 1:
                depth_out = self.shape[-1]
            else:
                depth_out = depth // 2

            if self.resize == 'stride':
                layers.append(nn.ConvTranspose2d(depth, depth_out, 4, stride=2, padding=1, **self.kw))
            elif self.resize == 'stride3':
                s = 3 if i == stages - 1 else 2
                k = 5 if i == stages - 1 else 4
                layers.append(nn.ConvTranspose2d(depth, depth_out, k, stride=s, padding=2, **self.kw))
            elif self.resize == 'resize':
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                layers.append(nn.Conv2d(depth, depth_out, 3, padding=1, **self.kw))
            else:
                raise NotImplementedError(self.resize)

            depth = depth_out

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.in_linear(x)
        x = x.view(x.shape[0], self.depth, self.minres, self.minres)
        x = self.layers(x)

        if max(x.shape[1:-1]) > max(self.shape[:-1]):
            padh = (x.shape[1] - self.shape[0]) // 2
            padw = (x.shape[2] - self.shape[1]) // 2
            x = x[:, :, padh: -padh, padw: -padw]

        assert x.shape[-3:] == self.shape, (x.shape, self.shape)

        if self.sigmoid:
            x = torch.sigmoid(x)
        else:
            x = x + 0.5

        return x



class ConvNetDecoder(nn.Module):

    def __init__(self,
                 latent_dim,
                 in_channels=3,
                 hidden_dims=None,
                 ):

        out_channels = in_channels

        # Build Decoder
        modules = []

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def forward(self, z):

        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result