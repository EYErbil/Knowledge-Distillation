import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
import time
import csv

# Import your custom models from model.py
from model import LightweightResNet

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test different models on the validation set")
    parser.add_argument('--training_mode', type=str, required=True,
                        choices=['regular', 'response', 'feature', 'teacher'],
                        help='Training mode to test: regular, response, feature, teacher')
    return parser.parse_args()

def get_model(training_mode, device):
    if training_mode == 'teacher':
        # Load teacher model (ResNet50 pretrained on COCO)
        model = models.segmentation.fcn_resnet50(pretrained=True)
    else:
        # For 'regular', 'response', 'feature', load the model definition
        model = LightweightResNet(num_classes=21)
        # Load the state dict
        model_path = os.path.join('checkpoints', {
            'regular': 'light_best.pth',
            'response': 'response_kd_best.pth',
            'feature': 'feature_kd_best.pth'
        }[training_mode])

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model '{training_mode}' loaded from '{model_path}'.")
        else:
            print(f"Model file '{model_path}' does not exist.")
            sys.exit(1)
    model = model.to(device)
    model.eval()
    return model

def decode_segmap(image, nc=21):
    # Define the PASCAL VOC color map
    label_colors = np.array([(0, 0, 0),
                             (128, 0, 0), (0, 128, 0), (128, 128, 0),
                             (0, 0, 128), (128, 0, 128), (0, 128, 128),
                             (128, 128, 128), (64, 0, 0), (192, 0, 0),
                             (64, 128, 0), (192, 128, 0), (64, 0, 128),
                             (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             (0, 64, 0), (128, 64, 0), (0, 192, 0),
                             (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def visualize_predictions(images, masks, preds, batch_idx, training_mode):
    """
    Visualizes predictions alongside images and ground truth masks.
    Saves the visualizations to files.
    """
    images = images.cpu()
    masks = masks.cpu()
    preds = preds.cpu()
    batch_size = images.shape[0]

    for i in range(batch_size):
        image = images[i]
        mask = masks[i].numpy()
        pred = preds[i].numpy()

        # Unnormalize the image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image = image * std + mean
        image = image.permute(1,2,0).numpy()
        image = np.clip(image, 0, 1)

        # Decode segmentation maps
        mask_rgb = decode_segmap(mask)
        pred_rgb = decode_segmap(pred)

        # Create a figure with subplots
        fig, axs = plt.subplots(1,3, figsize=(12,4))
        axs[0].imshow(image)
        axs[0].set_title('Image')
        axs[0].axis('off')

        axs[1].imshow(mask_rgb)
        axs[1].set_title('Ground Truth')
        axs[1].axis('off')

        axs[2].imshow(pred_rgb)
        axs[2].set_title('Prediction')
        axs[2].axis('off')

        plt.tight_layout()
        # Save the figure
        save_path = f'visualizations/{training_mode}_batch{batch_idx}_img{i}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

        # Save the raw predictions (guesses) and ground truth masks
        pred_save_path = f'predictions/{training_mode}_batch{batch_idx}_img{i}_pred.png'
        mask_save_path = f'predictions/{training_mode}_batch{batch_idx}_img{i}_mask.png'
        os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
        plt.imsave(pred_save_path, pred, cmap='gray')
        plt.imsave(mask_save_path, mask, cmap='gray')

def main():
    args = parse_arguments()
    training_mode = args.training_mode

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the appropriate model
    model = get_model(training_mode, device)

    # Define mean and std for normalization (PASCAL VOC)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Define image transformations
    transform_image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Define mask transformations
    def squeeze_and_convert(x):
        return x.squeeze().long()

    transform_mask = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(squeeze_and_convert)
    ])

    # Load the PASCAL VOC 2012 validation dataset
    val_dataset = datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',
        download=True,
        transform=transform_image,
        target_transform=transform_mask
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # Initialize mIoU metrics
    num_classes = 21  # 20 classes + background
    miou_with_ignore = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(device)
    miou_without_ignore = MulticlassJaccardIndex(num_classes=num_classes).to(device)

    # Number of batches to visualize
    num_visualizations = 5
    vis_count = 0

    # Variables to measure inference time
    total_time = 0.0
    total_images = 0

    # Evaluation loop
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating")):
            images = images.to(device)
            masks = masks.to(device)
            batch_size = images.size(0)
            total_images += batch_size

            # Measure inference time
            start_time = time.perf_counter()
            outputs = model(images)
            inference_time = time.perf_counter() - start_time
            total_time += inference_time

            if isinstance(outputs, tuple):
                outputs, _ = outputs  # For custom models returning (out, features)
            elif isinstance(outputs, dict):
                outputs = outputs['out']  # For torchvision models

            # Get predictions
            preds = torch.argmax(outputs, dim=1)  # Shape: [batch_size, H, W]

            # Clamp masks to valid range
            masks = torch.clamp(masks, min=0, max=num_classes - 1)

            # Update mIoU metrics
            miou_with_ignore.update(preds, masks)
            miou_without_ignore.update(preds, masks)

            # Visualize predictions
            if vis_count < num_visualizations:
                visualize_predictions(images, masks, preds, idx, training_mode)
                vis_count += 1

    # Compute final mIoUs
    final_miou_with_ignore = miou_with_ignore.compute().item()
    final_miou_without_ignore = miou_without_ignore.compute().item()

    # Compute average inference time per image in milliseconds
    avg_inference_time_per_image = (total_time / total_images) * 1000  # Convert to milliseconds

    # Display results
    print(f"Mean Intersection over Union (mIoU) with ignore_index=255: {final_miou_with_ignore:.4f}")
    print(f"Mean Intersection over Union (mIoU) without ignore_index: {final_miou_without_ignore:.4f}")
    print(f"Average inference time per image: {avg_inference_time_per_image:.2f} ms")

    # Save results to a CSV file for comparison
    results_file = 'results.csv'
    file_exists = os.path.isfile(results_file)
    with open(results_file, 'a', newline='') as csvfile:
        fieldnames = ['training_mode', 'mIoU_with_ignore', 'mIoU_without_ignore', 'avg_inference_time_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow({
            'training_mode': training_mode,
            'mIoU_with_ignore': final_miou_with_ignore,
            'mIoU_without_ignore': final_miou_without_ignore,
            'avg_inference_time_ms': avg_inference_time_per_image
        })

if __name__ == "__main__":
    main()
