import os
import shutil
import zipfile
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# Import custom modules
from model import BaselineCNN
from utils import list_files, remove_duplicates, copy_split, train_model, evaluate_model, visualize_errors

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Setup Directories ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # src/
    PROJECT_ROOT = os.path.dirname(BASE_DIR)              # Project Root
    
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # --- 2. Data Preparation ---
    zip_path = os.path.join(DATA_DIR, "BrainTumor.zip")
    extract_to = os.path.join(DATA_DIR, "BrainTumor")         # Raw extracted data
    final_dataset_dir = os.path.join(DATA_DIR, "BrainTumor_Split") # Final Clean Split

    # A. Extract Logic
    if os.path.exists(zip_path):
        if not os.path.exists(extract_to):
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print("Extraction complete.")
    elif os.path.exists(extract_to):
        print(f"Zip not found, but extracted folder exists at {extract_to}. Proceeding...")
    else:
        print(f"CRITICAL ERROR: Neither {zip_path} nor {extract_to} found.")
        print("Please place 'BrainTumor.zip' inside the 'data/' folder.")
        return

    # B. Cleaning Logic (Duplicates)
    print("Checking for duplicates...")
    hash_dict = {}
    list_files(hash_dict, extract_to) # Pass the extracted folder path
    remove_duplicates(hash_dict)

    # C. Splitting Logic
    # Check if we already created the split dataset to save time
    if not os.path.exists(final_dataset_dir) or not os.listdir(final_dataset_dir):
        print("Creating Train/Val/Test split...")
        ROOT = Path(extract_to)
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        # Collect paths
        data = []
        for root, dirs, files in os.walk(ROOT):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Infer label from parent folder name
                    label = os.path.basename(root)
                    if label in classes:
                        data.append({'path': os.path.join(root, file), 'label': label})

        if not data:
            print("Error: No images found in extracted folder. Check directory structure.")
            return

        df = pd.DataFrame(data)
        print(f"Total images found: {len(df)}")

        X = df['path']
        y = df['label']

        # Split 80/10/10
        # 1. Split off Test (10%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.10, stratify=y, random_state=42
        )
        # 2. Split Train/Val 
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.1111, stratify=y_train_val, random_state=42
        )

        copy_split(X_train, y_train, "train", classes, final_dataset_dir)
        copy_split(X_val,   y_val,   "val", classes, final_dataset_dir)
        copy_split(X_test,  y_test,  "test", classes, final_dataset_dir)
    else:
        print(f"Split dataset already exists at {final_dataset_dir}")

    # --- 3. Load Data ---
    IMG_SIZE = 224
    BATCH_SIZE = 32

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    train_path = os.path.join(final_dataset_dir, "train")
    val_path = os.path.join(final_dataset_dir, "val")
    test_path = os.path.join(final_dataset_dir, "test")

    train_set = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_set = datasets.ImageFolder(root=val_path, transform=test_transform)
    test_set = datasets.ImageFolder(root=test_path, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_set)} | Validation: {len(val_set)} | Testing: {len(test_set)}")

    # --- 4. Setup Models ---
    print("\nInitializing Models...")
    
    # Baseline
    model_baseline = BaselineCNN(num_classes=4).to(device)
    opt_base = torch.optim.Adam(model_baseline.parameters(), lr=0.001, weight_decay=1e-3)
    sched_base = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_base, mode='max', patience=3, factor=0.5)

    # ResNet18
    model_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 4)
    model_resnet = model_resnet.to(device)
    
    opt_res = torch.optim.Adam(model_resnet.parameters(), lr=0.001, weight_decay=1e-3)
    sched_res = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_res, mode='max', patience=3, factor=0.5)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- 5. Train Models ---
    
    # Train Baseline
    print("\n--- Training BaselineCNN ---")
    hist_base = train_model(
        model_baseline, "BaselineCNN",
        train_loader, val_loader,
        criterion, opt_base, sched_base, device
    )
    # Move saved weights to checkpoints
    if os.path.exists("BaselineCNN_best.pth"):
        shutil.move("BaselineCNN_best.pth", os.path.join(CHECKPOINT_DIR, "BaselineCNN_best.pth"))

    # Train ResNet
    print("\n--- Training ResNet18 ---")
    hist_res = train_model(
        model_resnet, "ResNet18",
        train_loader, val_loader,
        criterion, opt_res, sched_res, device
    )
    if os.path.exists("ResNet18_best.pth"):
        shutil.move("ResNet18_best.pth", os.path.join(CHECKPOINT_DIR, "ResNet18_best.pth"))

    # --- 6. Plotting Results ---
    print("Saving comparison plots to results/...")
    
    plt.figure(figsize=(12, 10))
    
    # Train Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(hist_base['train_acc'], label='Baseline Train')
    plt.plot(hist_res['train_acc'], label='ResNet Train')
    plt.title("Training Accuracy")
    plt.legend(); plt.grid(True)

    # Train Loss
    plt.subplot(2, 2, 2)
    plt.plot(hist_base['train_loss'], label='Baseline Train')
    plt.plot(hist_res['train_loss'], label='ResNet Train')
    plt.title("Training Loss")
    plt.legend(); plt.grid(True)

    # Val Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(hist_base['val_acc'], label='Baseline Val')
    plt.plot(hist_res['val_acc'], label='ResNet Val')
    plt.title("Validation Accuracy")
    plt.legend(); plt.grid(True)
    
    # Val Loss
    plt.subplot(2, 2, 4)
    plt.plot(hist_base['val_loss'], label='Baseline Val')
    plt.plot(hist_res['val_loss'], label='ResNet Val')
    plt.title("Validation Loss")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_comparison.png"))
    print("Plots saved.")

    # --- 7. Evaluation ---
    class_names = train_set.classes
    
    # Evaluate Baseline
    print("\n--- Evaluating BaselineCNN ---")
    base_path = os.path.join(CHECKPOINT_DIR, 'BaselineCNN_best.pth')
    if os.path.exists(base_path):
        model_baseline.load_state_dict(torch.load(base_path, map_location=device))
        evaluate_model(model_baseline, test_loader, device, class_names, model_name="Baseline CNN")
        visualize_errors(model_baseline, test_loader, device, class_names, model_name="Baseline CNN")
    else:
        print(f"Error: {base_path} not found.")

    # Evaluate ResNet
    print("\n--- Evaluating ResNet18 ---")
    res_path = os.path.join(CHECKPOINT_DIR, 'ResNet18_best.pth')
    if os.path.exists(res_path):
        model_resnet.load_state_dict(torch.load(res_path, map_location=device))
        evaluate_model(model_resnet, test_loader, device, class_names, model_name="ResNet18")
        visualize_errors(model_resnet, test_loader, device, class_names, model_name="ResNet18")
    else:
        print(f"Error: {res_path} not found.")

if __name__ == "__main__":
    main()