import torch
import torch.nn as nn

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
        # print(y_scale.shape, y_bias.shape)
        return y_scale * ((x - x.mean(dim=[-2, -1], keepdim=True)) / x.std(dim=[-2, -1], keepdim=True) + self.eps) + y_bias

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
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')  # Upsample
        x = self.conv1(x)
        x = self.noise1(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.adain1(x, w)
        
        x = self.conv2(x)
        x = self.noise2(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.adain2(x, w)
        return x

class InitialBlock(nn.Module):
    def __init__(self,channels):
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
        x = nn.functional.leaky_relu(x)
        x = self.adain1(x, w)
        x = self.conv(x)
        x = self.noise2(x)
        x = nn.functional.leaky_relu(x)
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
    def __init__(self, in_channels=512, w_dim=512,num_map_layers=8, channels=[512, 512, 512, 512, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16]):
        super().__init__()
        # initial block (input layer)
        self.norm = PixelNorm()
        self.input_layer = InputLayer(latent_dim=w_dim, num_layers=num_map_layers)
        self.initial_block = InitialBlock(channels=512)
        
        # if stage == 0 (no prev rgb to blend)
        self.first_stage_block = GBlock(in_channels=512, out_channels=16)
        self.first_stage_torgb = ToRGB(in_channels=16)
        
        # all hidden layers
        self.hidden_layers = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        for i in range(len(channels) - 1): # i + 1 shouldn't out of the range
            self.hidden_layers.append(GBlock(in_channels=channels[i], out_channels=channels[i+1], w_dim=w_dim))
            self.to_rgb_layers.append(ToRGB(in_channels=channels[i+1]))
    
    def forward(self, z, stage, alpha):
        # inputs
        z = self.norm(z)
        w = self.input_layer(z)
        x = self.initial_block(w)
        
        if stage == 0:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.first_stage_block(x, w)
            return self.first_stage_torgb(x)
        
        for i in range(stage - 1):  # go until prev stage for fade in
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.hidden_layers[i](x, w) # this will never be -1 since we have different condition for zero,
        
        x_upsample = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        
        x_prev = self.to_rgb_layers[stage - 1](x_upsample)
        
        x_new = self.hidden_layers[stage](x, w)
        x_new = self.to_rgb_layers[stage](x_new)
        return (alpha * x_prev) + ((1 - alpha) * x_new)
        
        
        
        
        
        
            
        
        
        
        
    
        
        
        
        
        
        
        
    
        


# Conv layer with learned noice input


if __name__ == "__main__":
    x = torch.randn((5, 512))
    img = torch.randn(5, 5, 24, 24)
    model = MappingNet()
    print(f"Successfull\nBefore: {x.shape}; After: {model(x).shape}")
    adain = AdaIN(out_channels=5, w_dim=512)
    print(f"Successfull\nBefore: {x.shape, img.shape}; After: {model(x).shape, adain(img, x).shape}")
    
    
    