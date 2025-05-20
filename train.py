# train.py

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
from torchvision import datasets
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import numpy as np
from PIL import Image


from model import get_model_by_name

def save_predictions(epoch, model, device, images, masks, save_dir):
    model.eval()
    with torch.no_grad():
        outputs,_ = model(images.to(device))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)

    for idx in range(len(images)):
        img = images[idx].transpose(1, 2, 0)
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        mask = masks[idx]
        pred = preds[idx]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        axes[1].imshow(mask, cmap='jet', vmin=0, vmax=20)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        axes[2].imshow(pred, cmap='jet', vmin=0, vmax=20)
        axes[2].set_title('Model Prediction')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_sample_{idx}.png'))
        plt.close()


def visualize_masks_with_ignored_pixels(dataset, save_dir, num_samples=5):
    """
    Visualize and save images, masks, and ignored pixels (255) from the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to visualize.
        save_dir (str): Directory to save the visualizations.
        num_samples (int): Number of samples to visualize.
    """
    import matplotlib.colors as mcolors
    import os

    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

    # Define colors for each label including 255 (ignored pixels)
    cmap = mcolors.ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'cyan',
                                   'magenta', 'purple', 'orange', 'pink', 'gray',
                                   'brown', 'lime', 'navy', 'gold', 'teal',
                                   'coral', 'maroon', 'olive', 'indigo', 'white', 'silver'])
    bounds = list(range(22))  # Labels 0-20 and 255
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for i in range(num_samples):
        # Get a sample from the dataset
        img, mask = dataset[i]
        img_np = img.permute(1, 2, 0).numpy()  # Convert to HWC format
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize to [0, 1]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Input Image
        axs[0].imshow(img_np)
        axs[0].set_title("Input Image")
        axs[0].axis('off')

        # Ground Truth Mask
        axs[1].imshow(mask, cmap=cmap, norm=norm)
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis('off')

        # Highlight Ignored Pixels (255)
        ignored_pixels = (mask == 255).float()
        axs[2].imshow(img_np)
        axs[2].imshow(ignored_pixels, cmap='hot', alpha=0.5)  # Overlay ignored pixels
        axs[2].set_title("Ignored Pixels (255)")
        axs[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{i + 1}.png")
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        plt.close(fig)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path to save the model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Early stopping triggered.')
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Custom Segmentation Models on PASCAL VOC 2012')
    parser.add_argument('--model_name', type=str, required=True, choices=['best','light'],
                        help='Name of the model to train: "simple" or "ultralight or best"')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model weights and metrics')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    parser.add_argument('--root_dir', type=str, default='./data', help='Root directory of the dataset')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience (default: 10)')
    parser.add_argument('--delta', type=float, default=0.0, help='Minimum change in validation loss to qualify as improvement (default: 0.0)')
    return parser.parse_args()

def get_dataloaders(root_dir, batch_size):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # Augmentation for the training images
    train_transform_image = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure all images are resized

        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Random color jitter
        transforms.ToTensor(),                                     # Convert PIL Image to Tensor
        transforms.Normalize(mean=mean, std=std)                  # Normalize
    ])

    # Augmentation for the training masks
    train_transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure all images are resized

        transforms.PILToTensor(),                                  # Convert PIL Mask to Tensor
        transforms.Lambda(lambda x: x.squeeze().long())            # Convert to LongTensor
    ])

    # Validation transforms (no augmentation)
    val_transform_image = transforms.Compose([
        transforms.Resize((256, 256)),                             # Resize to 256x256
        transforms.ToTensor(),                                     # Convert PIL Image to Tensor
        transforms.Normalize(mean=mean, std=std)                  # Normalize
    ])

    val_transform_mask = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),  # Resize with 
        transforms.PILToTensor(),                                  # Convert PIL Mask to Tensor
        transforms.Lambda(lambda x: x.squeeze().long())            # Convert to LongTensor
    ])

    # Load the training dataset
    train_dataset = datasets.VOCSegmentation(
        root=root_dir,
        year='2012',
        image_set='train',
        download=True,
        transform=train_transform_image,
        target_transform=train_transform_mask
    )

    # Load the validation dataset
    val_dataset = datasets.VOCSegmentation(
        root=root_dir,
        year='2012',
        image_set='val',
        download=True,
        transform=val_transform_image,
        target_transform=val_transform_mask
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader

def get_model(model_name, num_classes=21):

    if model_name.lower() == 'light':
        model = get_model_by_name('light',num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'simple' or 'ultralight'.")
    return model

def initialize_metrics(num_classes, device):
    miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(device)
    return miou

def plot_metrics(train_losses, val_losses, train_mious, val_mious, model_name, save_dir):
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Plot mIoU
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mious, 'b-', label='Training mIoU')
    plt.plot(epochs, val_mious, 'r-', label='Validation mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('mIoU over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_metrics.png'))
    plt.show()

def plot_predictions(images, masks, preds, save_dir, model_name):
    num_samples = len(images)
    plt.figure(figsize=(15, num_samples * 5))
    for i in range(num_samples):
        # Input Image
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(images[i])
        plt.title("Input Image")
        plt.axis('off')

        # Ground Truth Mask
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(masks[i], cmap='jet', vmin=0, vmax=20)  # 21 classes
        plt.title("Ground Truth Mask")
        plt.axis('off')

        # Predicted Mask
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(preds[i], cmap='jet', vmin=0, vmax=20)
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_predictions.png'))
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, patience, delta, save_dir, model_name):
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []
    visualize_images = []
    visualize_masks = []
    visualize_preds = []

    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta, path=os.path.join(save_dir, f'{model_name}_best.pth'))

    miou_metric = initialize_metrics(num_classes=21, device=device)
     # Prepare fixed batch for visualization
    fixed_images, fixed_masks = next(iter(val_loader))
    fixed_images, fixed_masks = fixed_images[:16], fixed_masks[:16]  # Take first 16 samples
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')

        # Training Phase
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc='Training', leave=False):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs,_ = model(images)  # [B, 21, H, W]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            miou_metric.update(preds, masks)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_miou = miou_metric.compute().item()
        train_losses.append(epoch_loss)
        train_mious.append(epoch_miou)

        print(f'Train Loss: {epoch_loss:.4f} mIoU: {epoch_miou:.4f}')

        miou_metric.reset()

        # Validation Phase
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation', leave=False):
                images = images.to(device)
                masks = masks.to(device)

                outputs,_ = model(images)
                loss = criterion(outputs, masks)

                running_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                miou_metric.update(preds, masks)

                # Save first few predictions for visualization
                if len(visualize_images) < 5:
                    for i in range(images.size(0)):
                        if len(visualize_images) >= 5:
                            break
                        img = images[i].cpu()
                        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                        img = img.permute(1, 2, 0).numpy()
                        img = np.clip(img, 0, 1)
                        visualize_images.append(img)

                        mask = masks[i].cpu().numpy()
                        preds_i = preds[i].cpu().numpy()
                        visualize_masks.append(mask)
                        visualize_preds.append(preds_i)

        epoch_loss_val = running_loss / len(val_loader.dataset)
        epoch_miou_val = miou_metric.compute().item()
        val_losses.append(epoch_loss_val)
        val_mious.append(epoch_miou_val)

        print(f'Validation Loss: {epoch_loss_val:.4f} mIoU: {epoch_miou_val:.4f}')

        miou_metric.reset()
        # Save predictions every 10 epochs
        if (epoch + 1) % 10 == 0:
            predictions_dir = os.path.join(save_dir, 'predictions')
            save_predictions(epoch + 1, model, device, fixed_images, fixed_masks, predictions_dir)
        # Scheduler Step
        scheduler.step(epoch_loss_val)

        # Early Stopping Check
        early_stopping(epoch_loss_val, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(save_dir, f'{model_name}_best.pth')))
    print(f'Best model loaded from {os.path.join(save_dir, f"{model_name}_best.pth")}')

    final_model_path = os.path.join(save_dir, f'{model_name}_final_normaltrain.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')

    # Plotting Loss and mIoU
    plot_metrics(train_losses, val_losses, train_mious, val_mious, model_name, save_dir)

    # Plotting Predictions
    plot_predictions(visualize_images, visualize_masks, visualize_preds, save_dir, model_name)

    return model, train_losses, val_losses, train_mious, val_mious

def verify_mask_labels(dataset, num_classes):
    import torch
    from collections import Counter
    label_counter = Counter()
    for _, mask in tqdm(dataset, desc='Verifying Labels', leave=False):
        unique_labels = torch.unique(mask)
        for label in unique_labels:
            label_counter[label.item()] += 1
    print("Label distribution in the dataset:")
    for label, count in sorted(label_counter.items()):
        print(f"Label {label}: {count} pixels")
    # Check if any label >= num_classes and not equal to 255
    for label in label_counter:
        if label >= num_classes and label != 255:
            print(f"Warning: Label {label} is outside the range [0, {num_classes-1}] and not equal to 255.")

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Set device
    import torch

    # Log file path
    log_file = "gpu_info.txt"
        
# Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Assuming you're using the first GPU
        gpu_name = torch.cuda.get_device_name(device)
        log_message = f"Training on GPU: {gpu_name}"
    else:
        log_message = "CUDA is not available. Training on CPU."

# Print the message to the console
    print(log_message)

# Write the log message to a .txt file
    with open(log_file, "w") as file:
        file.write(log_message)

    print(f"GPU information logged into {log_file}")


    # Get DataLoaders
    train_loader, val_loader = get_dataloaders(args.root_dir, args.batch_size)

    # Verify mask labels
    print("Verifying training dataset labels:")
    #verify_mask_labels(train_loader.dataset, num_classes=21)
    print("\nVerifying validation dataset labels:")
    #verify_mask_labels(val_loader.dataset, num_classes=21)
    # Visualize masks with ignored pixels
    print("\nVisualizing ignored pixels and label distributions in the training set:")
    save_dir = os.path.join(args.save_dir, "visualizations")
    visualize_masks_with_ignored_pixels(train_loader.dataset, save_dir=save_dir, num_samples=5)
    # Initialize model
    model = get_model(args.model_name, num_classes=21)
    model = model.to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Define Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # Train the model
    trained_model, train_losses, val_losses, train_mious, val_mious = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        patience=args.patience,
        delta=args.delta,
        save_dir=args.save_dir,
        model_name=args.model_name
    )