import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import segmentation_models_pytorch as smp


# --- MODEL, DATASET, AND LOSS CLASSES REMAIN THE SAME ---
# (I am keeping your existing imports and classes as they are correct)

# Dataset class
class MRIDataset(Dataset):
    def __init__(self, t1_dir, t2_dir, gt_dir, augment=True, gaussian_kernel=(5, 5)):
        self.t1_dir = t1_dir
        self.t2_dir = t2_dir
        self.gt_dir = gt_dir
        self.augment = augment

        self.image_files = sorted([f for f in os.listdir(t1_dir) if f.endswith('.jpg')])

        self.rotation_range = 30
        self.shift_frac = 0.1
        self.gaussian_kernel = gaussian_kernel
        self.contrast_range = (0.8, 1.2)
        self.gamma_range = (0.8, 1.2)
        self.intensity_aug_prob = 0.5

    def __len__(self):
        return len(self.image_files)

    def random_adjust_intensity(self, img):
        if np.random.rand() > self.intensity_aug_prob:
            return img

        alpha = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        img = img.astype(np.float32) * alpha
        img = np.clip(img, 0, 255)

        gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
        img = img / 255.0
        img = np.power(img, gamma)
        img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        t1_path = os.path.join(self.t1_dir, img_name)
        t2_path = os.path.join(self.t2_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)

        t1_img = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
        t2_img = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        target_size = (512, 512)
        t1_img = cv2.resize(t1_img, target_size, interpolation=cv2.INTER_LINEAR)
        t2_img = cv2.resize(t2_img, target_size, interpolation=cv2.INTER_LINEAR)
        gt_img = cv2.resize(gt_img, target_size, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if self.gaussian_kernel is not None:
                t1_img = cv2.GaussianBlur(t1_img, self.gaussian_kernel, 0)
                t2_img = cv2.GaussianBlur(t2_img, self.gaussian_kernel, 0)

            t1_img = self.random_adjust_intensity(t1_img)
            t2_img = self.random_adjust_intensity(t2_img)

            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            h, w = t1_img.shape[:2]
            max_tx = int(self.shift_frac * w)
            max_ty = int(self.shift_frac * h)
            tx = int(np.random.uniform(-max_tx, max_tx))
            ty = int(np.random.uniform(-max_ty, max_ty))

            center = (w // 2, h // 2)
            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            M_rot[0, 2] += tx
            M_rot[1, 2] += ty

            t1_img = cv2.warpAffine(t1_img, M_rot, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0,))
            t2_img = cv2.warpAffine(t2_img, M_rot, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0,))
            gt_img = cv2.warpAffine(gt_img, M_rot, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0,))

        t1_img = t1_img.astype(np.float32) / 255.0
        t2_img = t2_img.astype(np.float32) / 255.0
        gt_img = gt_img.astype(np.float32) / 255.0

        input_img = np.stack([t1_img, t2_img], axis=0)
        gt_img = np.expand_dims(gt_img, axis=0)

        return torch.from_numpy(input_img), torch.from_numpy(gt_img)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


# ---------------------------------------------------------
# UPDATED Training Function
# ---------------------------------------------------------
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50, val_loader=None, scheduler=None,
                writer=None, save_path=None):
    """
    Train the model and save the version with the highest Validation Dice Score.
    """

    # Initialize best score tracker
    best_val_dice = -1.0

    def compute_dice(outputs, targets):
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (preds_flat * targets_flat).sum(dim=1)
        sums = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection) / (sums + 1e-8)
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

        # Scheduler Step
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()

        # Validation Step
        avg_val_loss = 0.0
        avg_val_dice = 0.0
        val_str = ""
        save_msg = ""

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            with torch.no_grad():
                for v_inputs, v_targets in val_loader:
                    v_inputs = v_inputs.to(device)
                    v_targets = v_targets.to(device)
                    v_outputs = model(v_inputs)
                    val_loss += criterion(v_outputs, v_targets).item()
                    val_dice += compute_dice(v_outputs, v_targets)

            avg_val_loss = val_loss / max(1, len(val_loader))
            avg_val_dice = val_dice / max(1, len(val_loader))
            val_str = f', Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}'

            # --- SAVE BEST MODEL LOGIC ---
            if save_path is not None:
                if avg_val_dice > best_val_dice:
                    best_val_dice = avg_val_dice
                    torch.save(model.state_dict(), save_path)
                    save_msg = f" --> Best Model Saved! (Dice: {best_val_dice:.4f})"

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}, Train Loss: {avg_loss:.4f}, Train Dice: {avg_dice:.4f}{val_str}{save_msg}')

        if writer is not None:
            writer.add_scalars('Loss', {'Train': avg_loss, 'Validation': avg_val_loss}, epoch + 1)
            writer.add_scalars('Dice', {'Train': avg_dice, 'Validation': avg_val_dice}, epoch + 1)
            writer.add_scalar('Learning_Rate', current_lr, epoch + 1)

    print(f"Training Complete. Best Validation Dice: {best_val_dice:.4f}")
    return model


# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # --- SETUP ---
    batch_size = 5
    learning_rate = 0.002
    num_epochs = 100

    # Ensure directories exist
    os.makedirs('models', exist_ok=True)  # Changed from models_resnet for consistency with your save path
    os.makedirs('runs', exist_ok=True)

    models_config = [
        {'name': 'CT', 'gt_dir': 'carpalTunnel_Sorted/CT', 'v_gt_dir': 'MRIsample/CT'},
        {'name': 'FT', 'gt_dir': 'carpalTunnel_Sorted/FT', 'v_gt_dir': 'MRIsample/FT'},
        {'name': 'MN', 'gt_dir': 'carpalTunnel_Sorted/MN', 'v_gt_dir': 'MRIsample/MN'}
    ]
    t1_dir = 'carpalTunnel_Sorted/T1'
    t2_dir = 'carpalTunnel_Sorted/T2'
    val_t1_dir = 'MRIsample/T1'
    val_t2_dir = 'MRIsample/T2'

    for config in models_config:
        seed = 319
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model_name = config['name']
        gt_dir = config['gt_dir']
        v_t_dir = config['v_gt_dir']
        print(f'\n{"=" * 50}')
        print(f'Training model for {model_name}')
        print(f'{"=" * 50}')

        gaussian_kernel = None if model_name == 'MN' else (5, 5)
        train_dataset = MRIDataset(t1_dir, t2_dir, gt_dir, gaussian_kernel=gaussian_kernel)
        val_dataset = MRIDataset(val_t1_dir, val_t2_dir, v_t_dir, augment=False, gaussian_kernel=None)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=2,
            classes=1,
        ).to(device)

        criterion = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(log_dir=os.path.join('runs', f"{model_name}_{timestamp}"))

        # Define where to save the best model
        best_model_path = f'models/{model_name.lower()}_best.pth'

        train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_epochs,
            val_loader=val_loader,
            scheduler=scheduler,
            writer=writer,
            save_path=best_model_path  # Pass the path here
        )

        writer.close()
        print(f'Best model for {model_name} saved to {best_model_path}')


if __name__ == '__main__':
    main()