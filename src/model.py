import torch
import torch.nn as nn
import torch.nn.functional as F

layer_factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

class FCLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.fc(x)
        
    
class MappingNet(nn.Module):
    def __init__(self, fc_dim=512, num_layers=8):
        super().__init__()
        self.map = nn.Sequential(
            *[FCLayer(fc_dim) for _ in range(num_layers)]
        )
    
    def forward(self, z):
        return self.map(z)
    
class AdaIN(nn.Module):
    def __init__(self, out_channels, w_dim=512):
        super().__init__()
        self.out_channels = out_channels
        self.eps = 1e-8
        # each channel will have it's own y_scale and y_bias
        self.A = nn.Linear(in_features=w_dim, out_features=2*out_channels, bias=True)
        self.A.weight.data.normal_(0, 1.0)
        self.A.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, w: torch.Tensor):
        style_params = self.A(w)# [ys, yb], scale and bias
        y_scale = style_params[:, :self.out_channels].unsqueeze(-1).unsqueeze(-1) # (B, out_channels, 1, 1)
        y_bias = style_params[: , self.out_channels: ].unsqueeze(-1).unsqueeze(-1) # (B, out_channels, 1, 1)
        return y_scale * ((x - x.mean(dim=[-2, -1], keepdim=True)) / (x.std(dim=[-2, -1], keepdim=True) + self.eps)) + y_bias

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale_b = nn.Parameter(torch.zeros(channels))
        
    def forward(self, x: torch.Tensor):
        batch_size, C, H, W = x.shape
        noise = torch.randn((batch_size, 1, H, W), device=x.device)
        scaled_noise = noise * self.scale_b.view(1, C, 1, 1)
        return x + scaled_noise

class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        # channels is unused but kept for API compatibility
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # (b, c, h, w)
        return x / (torch.sqrt(torch.mean(x*x, dim=1, keepdim=True)) + self.eps)
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      padding=padding, 
                      stride=stride)
        
    def forward(self, x: torch.Tensor):
        return self.conv_layer(x)

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim=512):
        super().__init__()
        # either use F.interpolate(slow, low accuracy) or conv pixel suffle (more compute, more accuracy)
        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=out_channels)
        self.conv2 = ConvLayer(in_channels=out_channels, out_channels=out_channels)
        self.noise1 = NoiseInjection(out_channels)
        self.noise2 = NoiseInjection(out_channels)
        self.adain1 = AdaIN(out_channels=out_channels, w_dim=w_dim)
        self.adain2 = AdaIN(out_channels=out_channels, w_dim=w_dim)
        
    def forward(self, x: torch.Tensor, w: torch.Tensor):
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')  # Upsample
        x = self.conv1(x)
        x = self.noise1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.adain1(x, w)
        
        x = self.conv2(x)
        x = self.noise2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.adain2(x, w)
        return x

class InitialBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # learned constant lantent input
        self.const_input = nn.Parameter(torch.randn(1, channels, 4, 4))
        
        self.conv = ConvLayer(in_channels=channels, out_channels=channels)
        self.adain1 = AdaIN(out_channels=channels)
        self.adain2 = AdaIN(out_channels=channels)
        
        self.noise1 = NoiseInjection(channels=channels)
        self.noise2 = NoiseInjection(channels=channels)
    
    def forward(self, w):
        b = w.shape[0]
        x = self.const_input.repeat(b, 1, 1, 1)
        x = self.noise1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.adain1(x, w)
        x = self.conv(x)
        x = self.noise2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.adain2(x, w)
        return x

class ToRGB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = ConvLayer(in_channels=in_channels, out_channels=3, kernel_size=1, stride=1, padding=0) # conv1x1
        
    def forward(self, x: torch.Tensor):
        return self.conv(x)
    
class FromRGB(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv = ConvLayer(in_channels=3, out_channels=out_channels, kernel_size=1, stride=1, padding=0) # conv1x1
        
    def forward(self, x: torch.Tensor):
        return self.conv(x)
    
class InputLayer(nn.Module):
    def __init__(self, latent_dim=512, num_layers=8):
        super().__init__()
        self.mapping_net = MappingNet(fc_dim=latent_dim, num_layers=num_layers)
        
    def forward(self, z: torch.Tensor): # (b, latent_dim)
        return self.mapping_net(z)

class Generator(nn.Module):
    def __init__(self, channel_size=512, w_dim=512, num_map_layers=8):
        super().__init__()
        # initial block (input layer)
        self.norm = PixelNorm()
        self.input_layer = InputLayer(latent_dim=w_dim, num_layers=num_map_layers)
        self.initial_block = InitialBlock(channels=channel_size)
        
        # all hidden layers
        self.hidden_layers = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        # if stage == 0 (no prev rgb to blend)
        
        self.to_rgb_layers.append(ToRGB(in_channels=512))
        
        for i in range(len(layer_factors) - 1): # i + 1 shouldn't out of the range
            in_channels = int(channel_size * layer_factors[i])
            out_channels = int(channel_size * layer_factors[i+1])
            self.hidden_layers.append(GBlock(in_channels=in_channels, out_channels=out_channels, w_dim=w_dim))
            self.to_rgb_layers.append(ToRGB(in_channels=out_channels))
    
    def forward(self, z, stage, alpha):
        # inputs
        z = self.norm(z)
        w = self.input_layer(z)
        x = self.initial_block(w) # for 4x4 
        
        if stage == 0:
            # upscaled = F.interpolate(x, scale_factor=2, mode='bilinear')
            # x = self.first_stage_block(upscaled, w)
            x = self.to_rgb_layers[0](x)
            return F.tanh(x)
        
        for i in range(stage):  # go until prev stage for fade in
            upscaled = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.hidden_layers[i](upscaled, w)
        
        # After the for loop we will get, upscaled and generated
        prev = self.to_rgb_layers[stage-1](upscaled) # rgb layers are one step ahead so, stage will be prev and stage + 1 will current
        generated = self.to_rgb_layers[stage](x)
        
        x = (alpha * generated) + ((1 - alpha) * prev)
        return F.tanh(x)

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim=512):
        super().__init__()
        # either use F.interpolate(slow, low accuracy) or conv pixel suffle (more compute, more accuracy)
        self.block = nn.Sequential(
            ConvLayer(in_channels=in_channels, out_channels=in_channels),
            nn.LeakyReLU(0.2),
            ConvLayer(in_channels=in_channels, out_channels=out_channels),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x: torch.Tensor):
        return self.block(x)
        
        

class Discriminator(nn.Module):
    def __init__(self, channels_size=512):
        super().__init__()
        self.downscale = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.final_block = nn.Sequential(
            ConvLayer(in_channels=channels_size+1, out_channels=channels_size),
            nn.LeakyReLU(0.2),
            ConvLayer(in_channels=channels_size, out_channels=channels_size, kernel_size=4, padding=0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(in_features=channels_size, out_features=1),
        )
        
        self.progblock, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        
        self.rgb_layers.append(FromRGB(out_channels=16)) # this is needed for the first stage, for fade in, we need two rgb layer

        # we reverse the module list, start from 4x4 to img_channel, so at inference we need to reverse through this module list, from img_channel to 4x4 -> 1
        
        for i in range(len(layer_factors) - 1, 0, -1): # reverse order [16, 32, 64 .. 512]
            in_channels = int(channels_size * layer_factors[i])
            out_channels = int(channels_size * layer_factors[i - 1])
            
            self.progblock.append(DBlock(in_channels=in_channels, out_channels=out_channels))
            self.rgb_layers.append(FromRGB(out_channels=out_channels))

    def calc_minbatch_std(self, x):
        std_channel = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.concat([x, std_channel], dim=1)

    def forward(self, x, stage, alpha):
        curr_step = len(self.progblock) - stage
        
        x_rgb = F.leaky_relu(self.rgb_layers[curr_step](x), 0.2) 
        
        # for 4x4 img    
        if stage == 0:
            x_rgb = self.calc_minbatch_std(x_rgb)
            x = self.final_block(x_rgb)
            return x
        
        x_downscaled = self.downscale(x)
        x_downscaled_rgb = F.leaky_relu(self.rgb_layers[curr_step + 1](x_downscaled), 0.2)
        
        x_real = self.progblock[curr_step](x_rgb)
        out = alpha * x_real + (1 - alpha) * x_downscaled_rgb
        
        for i in range(curr_step + 1, len(self.progblock)): # we have final block seperate, so go until -1
            out = self.progblock[i](out)
        
        out = self.calc_minbatch_std(out)
        x = self.final_block(out)
        return x
        
 


# Conv layer with learned noice input


if __name__ == "__main__":
    x = torch.randn((5, 512))
    img = torch.randn(5, 5, 24, 24)
    model = MappingNet()
    print(f"Successfull\nBefore: {x.shape}; After: {model(x).shape}")
    adain = AdaIN(out_channels=5, w_dim=512)
    print(f"Successfull\nBefore: {x.shape, img.shape}; After: {model(x).shape, adain(img, x).shape}")
    
    gen = Generator()
    x = torch.randn(1, 512)
    print(gen(x, 3,0.5))
    
    x = torch.rand(1 ,8, 1024, 1024)
    des = Discriminator()
    des(x, 8, 1).shape