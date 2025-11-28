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
    def __init__(self, w_dim=512):
        super().__init__()
        self.A = nn.Linear(in_features=w_dim, out_features=2, bias=True)
    
    def forward(self, x: torch.Tensor, w: torch.Tensor):
        style_params = self.A(w) # [ys, yb], scale and bias
        style_params[:, 0] * ((x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)) + style_params[:, 1]
        
        


# Conv layer with learned noice input


if __name__ == "__main__":
    x = torch.randn((5, 512))
    model = MappingNet()
    print(f"Successfull\nBefore: {x.shape}; After: {model(x).shape}")
    