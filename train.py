# train.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml
from data.datasets import ShadowRemovalDataset
from models.SSVANet import SSVA_Net
from losses.losses import CombinedLoss
from utils.utils import calculate_psnr_ssim

def train():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    image_size = config['image_size']  # Read image size from config
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Create datasets and dataloaders
    train_dataset = ShadowRemovalDataset(
        config['train_gt_dir'], config['train_lq_dir'], transform=transform, augment=True)
    val_dataset = ShadowRemovalDataset(
        config['val_gt_dir'], config['val_lq_dir'], transform=transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Initialize model
    model = SSVA_Net(**config['model'])
    model.to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-8)
    criterion = CombinedLoss(device=device)

    best_psnr = 0.0

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{config["num_epochs"]}', unit='batch') as pbar:
            for real_shadow, real_free in train_loader:
                real_shadow, real_free = real_shadow.to(device), real_free.to(device)

                optimizer.zero_grad()
                pred_free = model(real_shadow)
                loss = criterion(pred_free, real_free)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}] - Average Loss: {avg_loss:.4f}")
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_psnr = 0.0
            val_ssim = 0.0
            val_loss = 0.0
            for real_shadow, real_free in val_loader:
                real_shadow, real_free = real_shadow.to(device), real_free.to(device)
                pred_free = model(real_shadow)
                loss = criterion(pred_free, real_free)
                val_loss += loss.item()
                psnr_value, ssim_value = calculate_psnr_ssim(pred_free, real_free)
                val_psnr += psnr_value
                val_ssim += ssim_value

            avg_val_loss = val_loss / len(val_loader)
            avg_val_psnr = val_psnr / len(val_loader)
            avg_val_ssim = val_ssim / len(val_loader)
            print(f"Validation - Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.4f}, SSIM: {avg_val_ssim:.4f}")

            # Save the best model
            if avg_val_psnr > best_psnr:
                best_psnr = avg_val_psnr
                os.makedirs(config['save_dir'], exist_ok=True)
                torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
                print(f"New best PSNR: {best_psnr:.4f}. Model saved.")

if __name__ == "__main__":
    train()
