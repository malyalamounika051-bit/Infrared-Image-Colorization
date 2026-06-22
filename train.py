import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from PIL import Image

# Import local modules
from dataset import generate_synthetic_data, SatelliteDataset
from models import DualBranchUNet, PatchGANDiscriminator
from losses import GANLoss, VGGPerceptualLoss

def calculate_psnr(img1, img2):
    """
    img1, img2: PyTorch tensors of shape (C, H, W) normalized to [-1, 1]
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    # Normalized image is between -1 and 1, so peak-to-peak amplitude is 2.0
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2):
    """
    A simple PyTorch implementation of SSIM (Structural Similarity Index) for 3D tensors.
    """
    # Rescale to [0, 1]
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    
    sigma1_sq = torch.mean((img1 - mu1) ** 2)
    sigma2_sq = torch.mean((img2 - mu2) ** 2)
    
    # Covariance
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / (denominator + 1e-8)
    return ssim.item()

def save_sample_grid(ir, enh_real, enh_fake, rgb_real, rgb_fake, filepath):
    """
    Saves a grid of input, real, and generated outputs.
    All tensors shape: (3, H, W) or (1, H, W) in range [-1, 1].
    """
    def to_pil_img(tensor, is_rgb=True):
        tensor = (tensor.detach().cpu() + 1.0) / 2.0 # [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        if is_rgb:
            # If 1 channel, repeat
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
            return Image.fromarray(ndarr)
        else:
            if tensor.shape[0] == 3:
                tensor = tensor[0:1] # take first channel
            ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).squeeze(0).to(torch.uint8).numpy()
            return Image.fromarray(ndarr)

    ir_pil = to_pil_img(ir, is_rgb=False)
    enh_real_pil = to_pil_img(enh_real, is_rgb=False)
    enh_fake_pil = to_pil_img(enh_fake, is_rgb=False)
    rgb_real_pil = to_pil_img(rgb_real, is_rgb=True)
    rgb_fake_pil = to_pil_img(rgb_fake, is_rgb=True)

    w, h = ir_pil.size
    grid = Image.new('RGB', (w * 5, h))
    grid.paste(ir_pil.convert('RGB'), (0, 0))
    grid.paste(enh_real_pil.convert('RGB'), (w, 0))
    grid.paste(enh_fake_pil.convert('RGB'), (w * 2, 0))
    grid.paste(rgb_real_pil, (w * 3, 0))
    grid.paste(rgb_fake_pil, (w * 4, 0))
    
    grid.save(filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--lambda_recon", type=float, default=100.0, help="Weight for L1 reconstruction loss")
    parser.add_argument("--lambda_percep", type=float, default=10.0, help="Weight for perceptual loss")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for saving models")
    parser.add_argument("--samples_dir", type=str, default="samples", help="Directory for validation samples")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)

    # 1. Dataset Generation / Loading
    train_dir = os.path.join(args.data_dir, "train")
    if not os.path.exists(train_dir) or len(os.listdir(os.path.join(train_dir, "ir"))) == 0:
        print("Dataset not found. Generating synthetic training pairs...")
        generate_synthetic_data(train_dir, num_images=160)
        
    full_dataset = SatelliteDataset(train_dir, is_train=True)
    
    # Split train/val
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Validation dataset should be non-augmented
    val_dataset.dataset.is_train = False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # 2. Initialize Models
    netG = DualBranchUNet(n_channels=1).to(args.device)
    netD = PatchGANDiscriminator(input_nc=1, target_nc=3).to(args.device)

    # 3. Loss & Optimizers
    criterionGAN = GANLoss(use_lsgan=True).to(args.device)
    criterionL1 = nn.L1Loss().to(args.device)
    criterionPerceptual = VGGPerceptualLoss().to(args.device) if args.lambda_percep > 0.0 else None

    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Linear decay scheduler
    def lambda_rule(epoch):
        # Decay to 0 linearly in the second half of training
        decay_start_epoch = args.epochs // 2
        if epoch < decay_start_epoch:
            return 1.0
        return 1.0 - float(epoch - decay_start_epoch) / (args.epochs - decay_start_epoch)
        
    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda_rule)
    schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda_rule)

    print("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        netG.train()
        netD.train()
        
        running_d_loss = 0.0
        running_g_loss = 0.0
        running_g_gan = 0.0
        running_g_l1 = 0.0
        running_g_percep = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for ir, enh_real, rgb_real in loop:
            ir = ir.to(args.device)
            enh_real = enh_real.to(args.device)
            rgb_real = rgb_real.to(args.device)
            
            # ---------------------------------------------------------
            # Train Discriminator
            # ---------------------------------------------------------
            optimizerD.zero_grad()
            
            # Generate fake images
            enh_fake, col_fake = netG(ir)
            
            # Real inputs
            pred_real = netD(ir, rgb_real)
            loss_d_real = criterionGAN(pred_real, True)
            
            # Fake inputs
            pred_fake = netD(ir, col_fake.detach())
            loss_d_fake = criterionGAN(pred_fake, False)
            
            # Total D loss
            loss_D = (loss_d_real + loss_d_fake) * 0.5
            loss_D.backward()
            optimizerD.step()
            
            running_d_loss += loss_D.item()
            
            # ---------------------------------------------------------
            # Train Generator
            # ---------------------------------------------------------
            optimizerG.zero_grad()
            
            # Adversarial loss (G wants to fool D)
            pred_fake_g = netD(ir, col_fake)
            loss_G_GAN = criterionGAN(pred_fake_g, True)
            
            # Reconstruction Losses
            loss_G_L1_enh = criterionL1(enh_fake, enh_real)
            loss_G_L1_col = criterionL1(col_fake, rgb_real)
            loss_G_L1 = loss_G_L1_enh + loss_G_L1_col
            
            # Perceptual Loss
            loss_G_percep = criterionPerceptual(col_fake, rgb_real) if criterionPerceptual is not None else torch.tensor(0.0, device=args.device)
            
            # Total G loss
            loss_G = loss_G_GAN + (args.lambda_recon * loss_G_L1) + (args.lambda_percep * loss_G_percep)
            loss_G.backward()
            optimizerG.step()
            
            running_g_loss += loss_G.item()
            running_g_gan += loss_G_GAN.item()
            running_g_l1 += loss_G_L1.item()
            running_g_percep += loss_G_percep.item()
            
            # Update tqdm progress bar
            loop.set_postfix(
                D_loss=loss_D.item(), 
                G_loss=loss_G.item(),
                L1=loss_G_L1.item()
            )

        # Decay LR
        schedulerG.step()
        schedulerD.step()
        
        # ---------------------------------------------------------
        # Validation & Logging
        # ---------------------------------------------------------
        netG.eval()
        avg_psnr = 0.0
        avg_ssim = 0.0
        
        with torch.no_grad():
            for idx, (ir, enh_real, rgb_real) in enumerate(val_loader):
                ir = ir.to(args.device)
                enh_real = enh_real.to(args.device)
                rgb_real = rgb_real.to(args.device)
                
                enh_fake, col_fake = netG(ir)
                
                # Metrics
                avg_psnr += calculate_psnr(col_fake[0], rgb_real[0])
                avg_ssim += calculate_ssim(col_fake[0], rgb_real[0])
                
                # Save first validation sample grid
                if idx == 0:
                    sample_path = os.path.join(args.samples_dir, f"epoch_{epoch:03d}.png")
                    save_sample_grid(ir[0], enh_real[0], enh_fake[0], rgb_real[0], col_fake[0], sample_path)
        
        avg_psnr /= len(val_loader)
        avg_ssim /= len(val_loader)
        
        print(f"\n--- Epoch {epoch} Metrics ---")
        print(f"D Loss: {running_d_loss/len(train_loader):.4f} | G Loss: {running_g_loss/len(train_loader):.4f}")
        print(f"GAN Loss: {running_g_gan/len(train_loader):.4f} | L1 Loss: {running_g_l1/len(train_loader):.4f} | Percep Loss: {running_g_percep/len(train_loader):.4f}")
        print(f"Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f}")
        
        # Save checkpoints
        if epoch == args.epochs or epoch % 10 == 0:
            torch.save(netG.state_dict(), os.path.join(args.checkpoint_dir, f"netG_epoch_{epoch}.pth"))
            torch.save(netD.state_dict(), os.path.join(args.checkpoint_dir, f"netD_epoch_{epoch}.pth"))
            
    # Save final models
    torch.save(netG.state_dict(), os.path.join(args.checkpoint_dir, "netG_final.pth"))
    print("Training finished! Saved final generator model netG_final.pth")

if __name__ == "__main__":
    main()
