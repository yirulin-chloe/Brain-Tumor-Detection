import os
import torch
from model import BaselineCNN, resnet18
from utils import list_files, remove_duplicates, copy_split, train_model, evaluate_model, visualize_errors
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preparation
    os.system("rm -rf BrainTumor")
    os.system("apt-get update && apt-get install -y p7zip-full")
    os.system("7z x data/BrainTumor.zip -oBrainTumor")
    hash_dict = {}
    list_files(hash_dict)
    remove_duplicates(hash_dict)

    ROOT = Path("BrainTumor")
    NEW_ROOT = Path("BrainTumor_80_10_10")   # new clean dataset

    # Classes (exact folder names in dataset)
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Collect all image paths + labels
    data = []

    for split in ['Training', 'Testing']:
        for cls in classes:
            folder = ROOT / split / cls
            for img_file in folder.glob("*.jpg"):
                data.append({'path': str(img_file), 'label': cls})

    df = pd.DataFrame(data)
    print(f"Total images found: {len(df)}")   # should print 6726

    # Stratified 80/10/10 split
    X = df['path']
    y = df['label']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, stratify=y_train_val, random_state=42)
    
    copy_split(X_train, y_train, "train", classes, NEW_ROOT)
    copy_split(X_val,   y_val,   "val", classes, NEW_ROOT)
    copy_split(X_test,  y_test,  "test", classes, NEW_ROOT)

    # Print final counts
    print("\nFinal 80/10/10 split:")
    print("Train :", len(X_train))
    print("Val   :", len(X_val))
    print("Test  :", len(X_test))

    # Load data
    IMG_SIZE = 224  # 224x224
    SEED = 42
    BATCH_SIZE = 32

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # ensure 3 channels
        transforms.Resize((IMG_SIZE, IMG_SIZE)),       # resize all images
        transforms.RandomHorizontalFlip(p=0.5),       # randomly flip horizontally
        transforms.RandomRotation(degrees=10),        # rotate randomly ± degrees
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
        transforms.ToTensor(),                         # convert to tensor
        normalize
    ])

    # For validation, test, we don't do data augmentation
    test_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),  # ensure 3 channels
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    # Loading data into memory
    train_path = "BrainTumor_80_10_10/train"
    val_path = "BrainTumor_80_10_10/val"
    test_path  = "BrainTumor_80_10_10/test"

    train_set = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_set = datasets.ImageFolder(root=val_path, transform=test_transform)
    test_set = datasets.ImageFolder(root=test_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print("Class to index mapping:", train_set.class_to_idx)
    print(f"Training samples  : {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")
    print(f"Testing samples   : {len(test_set)}")
    
    # Models
    # Setup Baseline
    model_baseline = BaselineCNN(num_classes=4).to(device)
    optimizer_base = torch.optim.Adam(model_baseline.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler_base = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_base, mode='max', patience=3, factor=0.5)

    # Setup ResNet18
    model_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 4)
    model_resnet = model_resnet.to(device)

    optimizer_res = torch.optim.Adam(model_resnet.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler_res = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_res, mode='max', patience=3, factor=0.5)

    # 3. Train Both
    # Using the same criterion for fair comparison
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history_baseline = train_model(
        model_baseline, "BaselineCNN",
        train_loader, val_loader,
        criterion, optimizer_base, scheduler_base, device
    )

    history_resnet = train_model(
        model_resnet, "ResNet18",
        train_loader, val_loader,
        criterion, optimizer_res, scheduler_res, device
    )

    # Plot Training Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_baseline['train_acc'], label='Baseline Train Acc')
    plt.plot(history_resnet['train_acc'], label='ResNet18 Train Acc')
    plt.title("Training Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Plot Training Loss
    plt.subplot(1, 2, 2)
    plt.plot(history_baseline['train_loss'], label='Baseline Train Loss')
    plt.plot(history_resnet['train_loss'], label='ResNet18 Train Loss')
    plt.title("Training Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.show()

    plt.figure(figsize=(12, 5))

    # Plot Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_baseline['val_acc'], label='Baseline Val Acc')
    plt.plot(history_resnet['val_acc'], label='ResNet18 Val Acc')
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Plot Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history_baseline['val_loss'], label='Baseline Val Loss')
    plt.plot(history_resnet['val_loss'], label='ResNet18 Val Loss')
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.show()

    # Evaluate
    # Get class names
    class_names = train_set.classes

    # Baseline CNN
    model_baseline.load_state_dict(torch.load('BaselineCNN_best.pth'))
    evaluate_model(model_baseline, test_loader, device, class_names, model_name="Baseline CNN")

    # ResNet18
    model_resnet.load_state_dict(torch.load('ResNet18_best.pth'))
    evaluate_model(model_resnet, test_loader, device, class_names, model_name="ResNet18")
    
    # Visualize
    # Baseline CNN Errors
    visualize_errors(model_baseline, test_loader, device, class_names, model_name="Baseline CNN")
    # ResNet18 Errors
    visualize_errors(model_resnet, test_loader, device, class_names, model_name="ResNet18")


if __name__ == "__main__":
    main()