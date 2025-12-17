import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F


# Simple U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

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
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

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

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return model


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    batch_size = 2
    learning_rate = 0.001
    num_epochs = 10

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Define the three models and their ground truth directories
    models_config = [
        {'name': 'CT', 'gt_dir': 'MRIsample/CT'},
        {'name': 'FT', 'gt_dir': 'MRIsample/FT'},
        {'name': 'GT', 'gt_dir': 'MRIsample/GT'}
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

        # Create dataset and dataloader
        dataset = MRIDataset(t1_dir, t2_dir, gt_dir)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        model = UNet(in_channels=2, out_channels=1).to(device)

        # Loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross Entropy for [0, 1] output
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        model = train_model(model, train_loader, criterion, optimizer, device, num_epochs)

        # Save the trained model
        model_path = f'models/unet_{model_name.lower()}.pth'
        torch.save(model.state_dict(), model_path)
        print(f'\nModel saved to {model_path}')

    print(f'\n{"="*50}')
    print('All models trained and saved successfully!')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()

