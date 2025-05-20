import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Dataloader for PASCAL VOC 2012 Semantic Segmentation")
    parser.add_argument('--root_dir', type=str, default='./data', help='Root directory for the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for the DataLoader')
    parser.add_argument('--visualize', action='store_true', help='Visualize images and masks')
    return parser.parse_args()

# Named function for mask transformation to avoid lambda issues
def squeeze_and_convert(x):
    return x.squeeze().long()

def get_dataloaders(args):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # Image transformations: Minimal for evaluation
    transform_image = transforms.Compose([
        transforms.Resize((256, 256)),               # Resize to 256x256
        transforms.ToTensor(),                       # Convert PIL Image to Tensor
        transforms.Normalize(mean=mean, std=std)     # Normalize as per VOC
    ])

    # Mask transformations: Resize and convert to LongTensor
    transform_mask = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(squeeze_and_convert)
    ])

    # Load the official training set
    train_dataset = datasets.VOCSegmentation(
        root=args.root_dir,
        year='2012',
        image_set='train',
        download=True,
        transform=transform_image,
        target_transform=transform_mask
    )

    # Load the official validation set
    val_dataset = datasets.VOCSegmentation(
        root=args.root_dir,
        year='2012',
        image_set='val',
        download=True,
        transform=transform_image,
        target_transform=transform_mask
    )

    # DataLoaders with num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

def visualize_samples(loader, mean, std, num_samples=4):
    images, masks = next(iter(loader))
    # Denormalize images for visualization
    images = images * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)
    images = images.permute(0, 2, 3, 1).numpy()

    plt.figure(figsize=(12, num_samples * 2))
    for i in range(num_samples):
        # Show the image
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(np.clip(images[i], 0, 1))
        plt.title("Image")
        plt.axis('off')

        # Show the mask
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(masks[i].squeeze().numpy(), cmap='jet')
        plt.title("Mask")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

from PIL import Image

def inspect_and_visualize_dataset(dataset, mean, std, num_samples=4):
    """
    Visualize images and masks before and after resizing.
    """
    plt.figure(figsize=(12, num_samples * 3))
    for i in range(num_samples):
        # Get original image and mask
        original_image, original_mask = dataset[i]
        
        # Check if the dataset provides PIL images directly
        if isinstance(original_image, torch.Tensor):
            # Original size for tensors
            original_size = (original_image.shape[1], original_image.shape[2])  # H, W
        else:
            # Original size for PIL images
            original_size = original_image.size  # W, H

        # Show original image
        plt.subplot(num_samples, 4, i * 4 + 1)
        if isinstance(original_image, torch.Tensor):
            original_image_np = original_image.permute(1, 2, 0).numpy() * np.array(std) + np.array(mean)
            original_image_np = np.clip(original_image_np, 0, 1)
            plt.imshow(original_image_np)
        else:
            plt.imshow(original_image)
        plt.title(f"Original Image ({original_size})")
        plt.axis("off")

        # Show original mask
        plt.subplot(num_samples, 4, i * 4 + 2)
        if isinstance(original_mask, torch.Tensor):
            plt.imshow(original_mask.numpy(), cmap='jet')
        else:
            plt.imshow(original_mask, cmap='jet')
        plt.title("Original Mask")
        plt.axis("off")

        # If resizing is needed, apply to PIL images
        if not isinstance(original_image, torch.Tensor):
            resized_image = dataset.transform(original_image)  # Transform image
            resized_mask = dataset.target_transform(original_mask)  # Transform mask
        else:
            resized_image = original_image  # Already resized
            resized_mask = original_mask  # Already resized

        # Show resized image
        plt.subplot(num_samples, 4, i * 4 + 3)
        resized_image_np = resized_image.permute(1, 2, 0).numpy() * np.array(std) + np.array(mean)
        resized_image_np = np.clip(resized_image_np, 0, 1)
        plt.imshow(resized_image_np)
        plt.title(f"Resized Image (256x256)")
        plt.axis("off")

        # Show resized mask
        plt.subplot(num_samples, 4, i * 4 + 4)
        plt.imshow(resized_mask.numpy(), cmap='jet')
        plt.title(f"Resized Mask (256x256)")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    args = parse_args()
    train_loader, val_loader = get_dataloaders(args)

    # Visualize and inspect dataset
    print("Visualizing images and masks before and after resizing...")
    inspect_and_visualize_dataset(
        dataset=train_loader.dataset,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        num_samples=5
    )

