import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


EPOCH = 100
SAVE_STEPS = 50

# add "high"


import torch
import torch.nn.functional as F

def gradient_penalty(critic, real, fake, device="cuda"):
    batch_size, C, H, W = real.shape

    # Sample epsilon uniformly from [0, 1]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)

    # Interpolate between real and fake
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)

    # Critic output on interpolated images
    mixed_scores = critic(interpolated)

    # Compute gradients w.r.t. interpolated samples
    grads = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # grads shape: (N, C, H, W)
    grads = grads.view(batch_size, -1)

    # L2 norm of gradients
    grad_norm = grads.norm(2, dim=1)

    # (||grad||2 − 1)^2
    gp = torch.mean((grad_norm - 1) ** 2)

    return gp


def train(generator, discriminator, g_optimizer, d_optimizer, train_loader, stage, alpha, save_step, device, lambda_gp):
    loader = tqdm.tqdm(train_loader, dynamic_ncols=True, smoothing=0.7, desc='Epoch: ')
    
    for step, real in enumerate(loader):
        real = real.to(device)
        batch_size = real.shape[0]
    
        z = torch.randn((batch_size, 512), device=device) 
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            fake = generator(z, stage, alpha)
            fake_critic = discriminator(fake.detach(), stage, alpha)
            real_critic = discriminator(real, stage, alpha)
            gp = gradient_penalty(discriminator, real, fake.detach(), device)
            disc_loss = (fake_critic - real_critic).mean() + lambda_gp * gp # maximize (real - fake) -> minimize - (real - fake)
        
        d_optimizer.zero_grad() # keeping zero grad before is not problem, since the gradient in discriminator will not update at generator update , so even if discriminator have mixed gradients it will not update
        disc_loss.backward()
        d_optimizer.step()
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            fake_critic = discriminator(fake, stage, alpha).mean()
            gen_loss = - fake_critic # maximize fake critic score
            
        g_optimizer.zero_grad()
        gen_loss.backward()
        g_optimizer.step()
            
            
        
    
    pass

def main():
    pass

if __name__ == "__main__":
    main()