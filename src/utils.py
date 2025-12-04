import torch
import torchvision

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")

    # Restore weights
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Fix LR so it doesn't use old checkpoint LR
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Return training step
    global_step = checkpoint.get("global_step", 0)

    return global_step

def save_checkpoint(model, optimizer, global_step, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    torch.save(checkpoint, filename)

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(writer, loss_critic, loss_gen, real, fake, tensorboard_step):
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)
    writer.add_scalar("Loss Gen", loss_gen, global_step=tensorboard_step)
    with torch.no_grad():
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)