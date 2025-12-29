import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
# IMPORT THE SCHEDULER MODULE
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Simple U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(UNet, self).__init__()

        # Smaller U-Net: reduce channel widths to lower memory & compute
        # Encoder channel widths (reduced)
        self.enc1 = self.conv_block(in_channels, 8)
        self.enc2 = self.conv_block(8, 16)
        self.enc3 = self.conv_block(16, 32)
        self.enc4 = self.conv_block(32, 64)

        # Bottleneck (reduced)
        self.bottleneck = self.conv_block(64, 128)

        # Decoder (mirrors encoder)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(64 + 64, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(32 + 32, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(16 + 16, 16)

        self.up1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(8 + 8, 8)

        # Output layer
        self.out = nn.Conv2d(8, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output with sigmoid to get [0, 1] range
        out = torch.sigmoid(self.out(dec1))
        return out


# Dataset class
class MRIDataset(Dataset):
    def __init__(self, t1_dir, t2_dir, gt_dir, transform=None):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.gt_dir = gt_dir
        self.transform = transform

        # Get all image filenames
        self.image_files = sorted([f for f in os.listdir(t1_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # Load T1 and T2 images
        t1_path = os.path.join(self.t1_dir, img_name)
        t2_path = os.path.join(self.t2_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)

        # Read images in grayscale
        t1_img = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
        t2_img = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Resize to 512x512 for training
        target_size = (512, 512)
        t1_img = cv2.resize(t1_img, target_size, interpolation=cv2.INTER_LINEAR)
        t2_img = cv2.resize(t2_img, target_size, interpolation=cv2.INTER_LINEAR)
        # Use nearest for masks to preserve labels
        gt_img = cv2.resize(gt_img, target_size, interpolation=cv2.INTER_NEAREST)

        # Normalize to [0, 1]
        t1_img = t1_img.astype(np.float32) / 255.0
        t2_img = t2_img.astype(np.float32) / 255.0
        gt_img = gt_img.astype(np.float32) / 255.0

        # Stack T1 and T2 as 2-channel input
        input_img = np.stack([t1_img, t2_img], axis=0)  # Shape: (2, H, W)
        gt_img = np.expand_dims(gt_img, axis=0)  # Shape: (1, H, W)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_img)
        gt_tensor = torch.from_numpy(gt_img)

        return input_tensor, gt_tensor



# ---------------------------------------------------------
# Updated Training Function
# ---------------------------------------------------------
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50, val_loader=None, scheduler=None, writer=None):
    """
    Train the model with an optional Learning Rate Scheduler.
    """

    # Helper for Dice calculation
    def compute_dice(outputs, targets, threshold=0.2):
        if threshold is not None:
            th = torch.full_like(outputs, fill_value=threshold)
            cond = outputs > th
            preds = torch.where(cond, torch.ones_like(outputs), torch.zeros_like(outputs))
        else:
            preds = outputs

        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (preds_flat * targets_flat).sum(dim=1)
        sums = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection) / sums
        return dice.mean().item()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += compute_dice(outputs, targets)

        avg_loss = running_loss / max(1, len(train_loader))
        avg_dice = running_dice / max(1, len(train_loader))

        # Initialize val metrics
        avg_val_loss = 0.0
        avg_val_dice = 0.0

        # Scheduler Step
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()

        # Validation Step
        val_str = ""
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            with torch.no_grad():
                for v_inputs, v_targets in val_loader:
                    v_inputs = v_inputs.to(device)
                    v_targets = v_targets.to(device)
                    v_outputs = model(v_inputs)
                    v_loss = criterion(v_outputs, v_targets)
                    val_loss += v_loss.item()
                    val_dice += compute_dice(v_outputs, v_targets)

            avg_val_loss = val_loss / max(1, len(val_loader))
            avg_val_dice = val_dice / max(1, len(val_loader))
            val_str = f', Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}'

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}, Train Loss: {avg_loss:.4f}, Train Dice: {avg_dice:.4f}{val_str}')

        # --- FIX: Log metrics to the SAME chart ---
        # Naming them "Loss/Train" and "Loss/Validation" puts them on one graph
        if writer is not None:
            writer.add_scalars('Loss', {
                'Train': avg_loss,
                'Validation': avg_val_loss
            }, epoch + 1)

            writer.add_scalars('Dice', {
                'Train': avg_dice,
                'Validation': avg_val_dice
            }, epoch + 1)

            writer.add_scalar('Learning_Rate', current_lr, epoch + 1)

    return model


# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- SETUP ---
    batch_size = 4
    learning_rate = 0.005
    num_epochs = 150
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    models_config = [
        {'name': 'CT', 'gt_dir': 'MRIsample/CT'},
        {'name': 'FT', 'gt_dir': 'MRIsample/FT'},
        {'name': 'MN', 'gt_dir': 'MRIsample/MN'}
    ]
    t1_dir = 'MRIsample/T1'
    t2_dir = 'MRIsample/T2'

    for config in models_config:
        # Reset seed for reproducibility
        seed = 319
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_name = config['name']
        gt_dir = config['gt_dir']

        print(f'\n{"=" * 50}')
        print(f'Training model for {model_name}')
        print(f'{"=" * 50}')

        dataset = MRIDataset(t1_dir, t2_dir, gt_dir)
        if len(dataset) == 0:
            raise ValueError(f"Dataset empty for {model_name}")

        val_size = max(1, int(round(len(dataset) * 0.1)))
        train_size = len(dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = UNet(in_channels=2, out_channels=1).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir_path = os.path.join('runs', f"{model_name}_{timestamp}")

        # Initialize the writer with this unique path
        writer = SummaryWriter(log_dir=log_dir_path)

        model = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_epochs,
            val_loader=val_loader,
            scheduler=scheduler,
            writer=writer
        )

        writer.close()

        model_path = f'models/unet_{model_name.lower()}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'\nModel saved to {model_path}')


if __name__ == '__main__':
    main()