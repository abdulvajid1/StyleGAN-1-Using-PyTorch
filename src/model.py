import torch
import torch.nn as nn

class FCLayer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.ReLU()
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
    
    def forward(self, x: torch.Tensor, w: torch.Tensor):
        style_params = self.A(w)# [ys, yb], scale and bias
        y_scale = style_params[:, :self.out_channels].unsqueeze(-1).unsqueeze(-1) # (B, out_channels, 1, 1)
        y_bias = style_params[: , self.out_channels: ].unsqueeze(-1).unsqueeze(-1) # (B, out_channels, 1, 1)
        print(y_scale.shape, y_bias.shape)
        return y_scale * ((x - x.mean(dim=[-2, -1], keepdim=True)) / x.std(dim=[-2, -1], keepdim=True) + self.eps) + y_bias
        
        


# Conv layer with learned noice input


if __name__ == "__main__":
    x = torch.randn((5, 512))
    img = torch.randn(5, 5, 24, 24)
    model = MappingNet()
    print(f"Successfull\nBefore: {x.shape}; After: {model(x).shape}")
    adain = AdaIN(out_channels=5, w_dim=512)
    print(f"Successfull\nBefore: {x.shape, img.shape}; After: {model(x).shape, adain(img, x).shape}")
    
    
    