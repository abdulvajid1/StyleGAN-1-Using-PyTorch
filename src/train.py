import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from math import log2
from rich.logging import RichHandler
import logging
from model import Generator, Discriminator
from utils import save_checkpoint, load_checkpoint
from tensorboardX import SummaryWriter
from utils import plot_to_tensorboard
import albumentations.pytorch as A
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from pathlib import Path

logging.basicConfig(
    # filename="training.log",
    level=logging.INFO,
    datefmt="[%X]",                # optional time format
    handlers=[RichHandler()]
    )

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

START_TRAIN_AT_IMG_SZ = 4
CHECKPOINT_GEN = 'generator.pth'
CHECKPOINT_DISC = 'discriminator.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [16, 16, 16, 16, 8, 8, 8, 4, 4]
IMG_SIZE = 1024
CHANNEL_SIZE = 3
Z_DIM = 512
LAMBDA_GP = 10
IN_CHANNEL = 512
NUM_STEPS = int(log2(IMG_SIZE) / 4) + 1
EPOCH = 100
SAVE_STEPS = 50
PROGRESSIVE_EPOCH = [10] * len(BATCH_SIZES)
FIXED_NOICE = torch.randn((1, Z_DIM), device=DEVICE)
NUM_WORKERS = 0
IMG_PATH = 'src/images'
PIN_MEMORY = False



# add "high"
# TODO : minbatch std



def get_dataloader(img_size, step):
    
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    batch_size = BATCH_SIZES[step]
    logging.info(f"Batch size in step {step} is {batch_size}")
    img_path = Path(IMG_PATH)
    logging.info(f'images in {img_path.absolute()}')
    dataset = datasets.ImageFolder(root=IMG_PATH, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    logging.info("Data loader completed")
    return loader, dataset
    
    
    

def gradient_penalty(critic, real, fake, device="cuda", stage=1, alpha=1):
    batch_size, C, H, W = real.shape

    # Sample epsilon uniformly from [0, 1]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)

    # Interpolate between real and fake
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)

    # Critic output on interpolated images
    mixed_scores = critic(interpolated, stage, alpha)

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


def train(generator, discriminator, g_optimizer, d_optimizer, train_loader, stage, alpha, save_step, device, lambda_gp, writer, dataset, tensorboard_step):
    loader = tqdm.tqdm(train_loader, dynamic_ncols=True, smoothing=0.7, desc='Epoch: ', leave=True)
    
    for step, (real, _) in enumerate(loader):
        real = real.to(device)
        batch_size = real.shape[0]
    
        z = torch.randn((batch_size, Z_DIM), device=device) 
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            fake = generator(z, stage, alpha)
            fake_critic = discriminator(fake.detach(), stage, alpha)
            real_critic = discriminator(real, stage, alpha)
            gp = gradient_penalty(discriminator, real, fake.detach(), device, stage, alpha)
            disc_loss = ((fake_critic - real_critic).mean()
                         + lambda_gp * gp
                         + 0.001 * torch.mean(real_critic)**2) # maximize (real - fake) -> minimize - (real - fake)
        
        d_optimizer.zero_grad() # keeping zero grad before is not problem, since the gradient in discriminator will not update at generator update , so even if discriminator have mixed gradients it will not update
        disc_loss.backward()
        d_optimizer.step()
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            fake_critic = discriminator(fake, stage, alpha).mean()
            gen_loss = - fake_critic # maximize fake critic score
            
        g_optimizer.zero_grad()
        gen_loss.backward()
        g_optimizer.step()
        
        alpha += batch_size / (len(dataset))
        
        
        plot_to_tensorboard(writer=writer,
                            loss_critic=disc_loss.item(),
                            loss_gen=gen_loss.item(),
                            real=real.detach().float(),
                            fake=fake.detach().float(),
                            tensorboard_step=tensorboard_step)
        
        tensorboard_step += 1
        
        return tensorboard_step, alpha

def main():
    # generator = torch.compile(Generator(channel_size=IN_CHANNEL, w_dim=Z_DIM, num_map_layers=8).to(DEVICE))
    # discriminator = torch.compile(Discriminator(channels_size=IN_CHANNEL).to(DEVICE))
    
    generator = Generator(channel_size=IN_CHANNEL, w_dim=Z_DIM, num_map_layers=8).to(DEVICE)
    discriminator = Discriminator(channels_size=IN_CHANNEL).to(DEVICE)
    
    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=LEARNING_RATE)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=LEARNING_RATE)
    
    logging.info("Init Model & Optimizer")
    
    writer = SummaryWriter(f"logs/gen")    
    
    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_GEN, generator, g_optimizer)
        load_checkpoint(CHECKPOINT_DISC, discriminator, d_optimizer)
        logging.info("loaded checkpoints")
    
    generator.train()
    discriminator.train()
    
    step = int(log2(START_TRAIN_AT_IMG_SZ / 4))
    
    logging.info(f"current step is {step}")
    
    for num_epochs in PROGRESSIVE_EPOCH[step: ]:
        loader, dataset = get_dataloader(4*2**step, step)
        logging.info(f"dataloader created for step {step}")
        alpha = 1e-8
        
        logging.info(f"Image size is {4*2**step}")
        
        tensorboard_step = 0
        
        for epoch in range(num_epochs):
            tensorboard_step, alpha = train(generator, discriminator, g_optimizer, d_optimizer, loader, step, alpha, SAVE_STEPS, DEVICE, LAMBDA_GP, writer, dataset, tensorboard_step)
        
        step += 1
        

if __name__ == "__main__":
    main()