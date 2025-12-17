import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# Simple U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(UNet, self).__init__()

        # Reduced-size U-Net: fewer channels to lower memory & compute
        # Encoder channel widths
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder (mirrors encoder)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(128 + 128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(64 + 64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(32 + 32, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(16 + 16, 16)

        # Output layer
        self.out = nn.Conv2d(16, out_channels, kernel_size=1)

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


# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50, val_loader=None):
    """
    Train the model and optionally evaluate on a validation set each epoch.

    Args:
        model: torch.nn.Module
        train_loader: DataLoader for training
        criterion: loss function
        optimizer: optimizer
        device: torch.device
        num_epochs: number of epochs
        val_loader: optional DataLoader for validation

    Returns:
        model (trained)
    """
    # Cross-entropy metric function
    def compute_cross_entropy(outputs, targets, epsilon=1e-7):
        """Compute binary cross-entropy metric"""
        # Clamp outputs to avoid log(0)
        outputs = torch.clamp(outputs, epsilon, 1 - epsilon)
        ce = -torch.mean(targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))
        return ce.item()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_ce = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Compute cross-entropy metric
            running_ce += compute_cross_entropy(outputs, targets)

        avg_loss = running_loss / max(1, len(train_loader))
        avg_ce = running_ce / max(1, len(train_loader))
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train CE: {avg_ce:.4f}')

        # Validation step
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_ce = 0.0
            with torch.no_grad():
                for v_inputs, v_targets in val_loader:
                    v_inputs = v_inputs.to(device)
                    v_targets = v_targets.to(device)
                    v_outputs = model(v_inputs)
                    v_loss = criterion(v_outputs, v_targets)
                    val_loss += v_loss.item()
                    # Compute cross-entropy metric
                    val_ce += compute_cross_entropy(v_outputs, v_targets)

            avg_val_loss = val_loss / max(1, len(val_loader))
            avg_val_ce = val_ce / max(1, len(val_loader))
            print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val CE: {avg_val_ce:.4f}')

    return model


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    batch_size = 2
    learning_rate = 0.001
    num_epochs = 50

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Define the three models and their ground truth directories
    models_config = [
        {'name': 'CT', 'gt_dir': 'MRIsample/CT'},
        {'name': 'FT', 'gt_dir': 'MRIsample/FT'},
        {'name': 'MN', 'gt_dir': 'MRIsample/MN'}
    ]

    t1_dir = 'MRIsample/T1'
    t2_dir = 'MRIsample/T2'

    # Train each model
    for config in models_config:
        model_name = config['name']
        gt_dir = config['gt_dir']

        print(f'\n{"="*50}')
        print(f'Training model for {model_name}')
        print(f'{"="*50}')

        # Create dataset and split into train/validation (90/10)
        dataset = MRIDataset(t1_dir, t2_dir, gt_dir)
        dataset_size = len(dataset)
        if dataset_size == 0:
            raise ValueError(f"Dataset for {model_name} is empty. Check paths: {t1_dir}, {t2_dir}, {gt_dir}")

        val_size = max(1, int(round(dataset_size * 0.1)))  # at least 1 sample for val
        train_size = dataset_size - val_size

        # Use torch.utils.data.random_split for reproducible splits if needed
        train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = UNet(in_channels=2, out_channels=1).to(device)

        # Loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross Entropy for [0, 1] output
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model with validation
        model = train_model(model, train_loader, criterion, optimizer, device, num_epochs, val_loader=val_loader)

        # Save the trained model
        model_path = f'models/unet_{model_name.lower()}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'\nModel saved to {model_path}')

    print(f'\n{"="*50}')
    print('All models trained and saved successfully!')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
