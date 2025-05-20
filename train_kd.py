import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchmetrics import JaccardIndex  # Correct metric for segmentation
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Import your custom models
from model import TeacherModelWithFeatures, get_model_by_name
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
        plt.imshow(masks[i], cmap='tab20', vmin=0, vmax=20)
        plt.title("Ground Truth Mask")
        plt.axis('off')

        # Predicted Mask
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(preds[i], cmap='tab20', vmin=0, vmax=20)
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_predictions.png'))
    plt.show()
def save_predictions(epoch, model, device, images, masks, save_dir):
    model.eval()
    with torch.no_grad():
        outputs, _ = model(images.to(device))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)

    # Define the colormap for segmentation masks
    cmap = plt.get_cmap('tab20')
    num_classes = 21

    for idx in range(len(images)):
        img = images[idx].transpose(1, 2, 0)
        # Denormalize the image
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        mask = masks[idx]
        pred = preds[idx]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Input Image
        axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # Ground Truth Mask
        axes[1].imshow(mask, cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')

        # Predicted Mask
        axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[2].set_title('Model Prediction')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_sample_{idx}.png'))
        plt.close()
class EarlyStopping:
    """Early stops the training if validation mIoU doesn't improve after a given patience."""
    def __init__(self, patience=30, verbose=False, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_miou = None
        self.early_stop = False
        self.path = path
        self.delta = delta  # Ensure delta is initialized here

    def __call__(self, val_miou, model):
        if self.best_miou is None:
            self.best_miou = val_miou
            self.save_checkpoint(val_miou, model)
        elif val_miou < self.best_miou + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print('Early stopping triggered.')
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation mIoU increased ({self.best_miou:.6f} --> {val_miou:.6f}).  Saving model ...')
            self.best_miou = val_miou
            self.save_checkpoint(val_miou, model)
            self.counter = 0

    def save_checkpoint(self, val_miou, model):
        torch.save(model.state_dict(), self.path)



def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Student Model with Knowledge Distillation on PASCAL VOC 2012')
    parser.add_argument('--model_name', type=str, required=True, choices=['best','light'],
                        help='Name of the student model to train: "best" or "light"')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model weights and metrics')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--root_dir', type=str, default='./data', help='Root directory of the dataset')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--delta', type=float, default=0.0, help='Minimum change in validation mIoU to qualify as improvement')
    parser.add_argument('--kd_method', type=str, required=True, choices=['response', 'feature'],
                        help='Knowledge distillation method: "response" or "feature"')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for the distillation loss')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for distillation')
    return parser.parse_args()


def get_dataloaders(root_dir, batch_size):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    transform_image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.squeeze().long())
    ])

    train_dataset = datasets.VOCSegmentation(
        root=root_dir,
        year='2012',
        image_set='train',
        download=True,
        transform=transform_image,
        target_transform=transform_mask
    )

    val_dataset = datasets.VOCSegmentation(
        root=root_dir,
        year='2012',
        image_set='val',
        download=True,
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        target_transform=transform_mask
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader


def train_kd(teacher, student, train_loader, val_loader, epochs, learning_rate, alpha, temperature, kd_method, device, save_dir):
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5, verbose=True)

    miou_metric = JaccardIndex(task='multiclass', num_classes=21, ignore_index=255).to(device)
    early_stopping = EarlyStopping(patience=30, verbose=True, path=os.path.join(save_dir, f'{kd_method}_kd_best.pth'))

    # Initialize metrics lists
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []
    visualize_images = []
    visualize_masks = []
    visualize_preds = []

    # Prepare fixed batch for visualization
    fixed_images, fixed_masks = next(iter(val_loader))
    fixed_images, fixed_masks = fixed_images[:16], fixed_masks[:16]  # Take first 16 samples
    fixed_images_train, fixed_masks_train = next(iter(train_loader))
    fixed_images_train, fixed_masks_train = fixed_images_train[:16], fixed_masks_train[:16]  # Take first 16 samples



    teacher.eval()
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits, teacher_features = teacher(inputs)

            if kd_method == "response":
                student_logits, _ = student(inputs)
                # Adjusted temperature scaling
                student_logits_t = student_logits / temperature
                teacher_logits_t = teacher_logits / temperature

                student_log_probs = F.log_softmax(student_logits_t, dim=1)
                teacher_probs = F.softmax(teacher_logits_t, dim=1)

                kd_loss = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs) * (temperature ** 2)
                label_loss = ce_loss(student_logits, labels)
                loss = alpha * kd_loss + (1 - alpha) * label_loss

            elif kd_method == "feature":
                student_logits, student_features = student(inputs)
                hidden_loss = 0.0
                for idx, (tf, sf) in enumerate(zip(teacher_features, student_features)):
                    sf_resized = F.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
                    sf_aligned = student.align_layers[idx](sf_resized)
                    hidden_loss += (1 - F.cosine_similarity(sf_aligned, tf.detach(), dim=1)).mean()
                label_loss = ce_loss(student_logits, labels)
                loss = alpha * hidden_loss + (1 - alpha) * label_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)  # Multiply by batch size

            preds = torch.argmax(student_logits, dim=1)
            miou_metric.update(preds, labels)

        train_miou = miou_metric.compute().item()
        miou_metric.reset()

        train_loss_avg = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss_avg)
        train_mious.append(train_miou)

        # Validation
        val_loss = 0.0
        student.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                student_logits, _ = student(inputs)
                loss = ce_loss(student_logits, labels)
                val_loss += loss.item() * inputs.size(0)

                preds = torch.argmax(student_logits, dim=1)
                miou_metric.update(preds, labels)

                # Save first few predictions for visualization
                if len(visualize_images) < 5:
                    for i in range(inputs.size(0)):
                        if len(visualize_images) >= 5:
                            break
                        img = inputs[i].cpu()
                        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                        img = img.permute(1, 2, 0).numpy()
                        img = np.clip(img, 0, 1)
                        visualize_images.append(img)

                        mask = labels[i].cpu().numpy()
                        preds_i = preds[i].cpu().numpy()
                        visualize_masks.append(mask)
                        visualize_preds.append(preds_i)

        val_miou = miou_metric.compute().item()
        miou_metric.reset()

        val_loss_avg = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss_avg)
        val_mious.append(val_miou)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss_avg:.4f}, Train mIoU: {train_miou:.4f}, Val Loss: {val_loss_avg:.4f}, Val mIoU: {val_miou:.4f}")

        if (epoch + 1) % 10 == 0:
            predictions_dir = os.path.join(save_dir, 'predictions_val')
            save_predictions(epoch + 1, student, device, fixed_images, fixed_masks, predictions_dir)
            predictions_dir_train=os.path.join(save_dir,'predictions_train')
            save_predictions(epoch + 1, student, device, fixed_images_train, fixed_masks_train, predictions_dir_train)


        scheduler.step(val_miou)
        early_stopping(val_miou, student)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model
    student.load_state_dict(torch.load(os.path.join(save_dir, f'{kd_method}_kd_best.pth')))
    print(f'Best model loaded from {os.path.join(save_dir, f"{kd_method}_kd_best.pth")}')

    # Save the final model
    final_model_path = os.path.join(save_dir, f'{kd_method}_kd_final.pth')
    torch.save(student.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')

    # Plotting Loss and mIoU
    # Assuming you have plot_metrics function
    plot_metrics(train_losses, val_losses, train_mious, val_mious, kd_method, save_dir)

    # Plotting Predictions
    # Assuming you have plot_predictions function
    plot_predictions(visualize_images, visualize_masks, visualize_preds, save_dir, kd_method)

def main():
    args = parse_arguments()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(args.root_dir, args.batch_size)

    weights = models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    teacher_model = models.segmentation.fcn_resnet50(weights=weights).to(device)
    teacher = TeacherModelWithFeatures(teacher_model).to(device)
    teacher.eval()

    student = get_model_by_name(args.model_name, num_classes=21).to(device)

    # Ensure optimizer includes alignment layers if using feature distillation
    if args.kd_method == "feature":
        optimizer = optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_kd(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        temperature=args.temperature,
        kd_method=args.kd_method,
        device=device,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()