import os
import hashlib
import shutil
from pathlib import Path
import copy
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ['glioma', 'meningioma','notumor', 'pituitary']
extract_path = 'BrainTumor'

def compute_hash(file):
    hasher = hashlib.md5()
    with open(file, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def list_files(hash_dict):
    for data_type in ['Training', 'Testing']:
        for label in LABELS:
            folder_path = os.path.join(extract_path, data_type, label)
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(".jpg"):
                        file_path = os.path.join(root, file)
                        file_hash = compute_hash(file_path)
                        hash_dict.setdefault(file_hash, []).append(file_path)

def remove_duplicates(hash_dict):
    duplicate_count = 0
    for hash_value, paths in hash_dict.items():
        if len(paths) > 1:
            for p in paths[1:]:  # keep first, delete others
                os.remove(p)
                duplicate_count += 1
    print("Total duplicates removed:", duplicate_count)

def copy_split(paths, labels, split_name, classes, NEW_ROOT):
    split_folder = NEW_ROOT / split_name
    for cls in classes:
        (split_folder / cls).mkdir(parents=True, exist_ok=True)

    for p, label in zip(paths, labels):
        dest = split_folder / label / Path(p).name
        shutil.copy2(p, dest)

def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, patience=7):
    print(f"\n{'='*10} Training {model_name} {'='*10}")

    # Track metrics
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    trigger_times = 0

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds)

        # Update History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Scheduler Step
        if scheduler:
            scheduler.step(val_acc)

        # Logging
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping & Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    print(f"Best Val Acc for {model_name}: {best_val_acc:.4f}")
    return history

def evaluate_model(model, test_loader, device, class_names, model_name="Model"):
    print(f"\n{'='*10} Evaluating {model_name} {'='*10}")

    # 1. Set to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []

    # 2. Iterate over test data
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Calculate Accuracy
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"FINAL TEST ACCURACY for {model_name}: {test_acc:.4f}")

    # 4. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix (Acc: {test_acc:.4f})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    # 5. Print Classification Report
    print(f"\nClassification Report ({model_name}):")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return test_acc

def show_image(tensor):
    """Helper to convert tensor to valid image for plotting"""
    img = tensor.numpy().transpose(1, 2, 0)
    # Normalize to [0, 1] range for display
    img = (img - img.min()) / (img.max() - img.min())
    return img

def visualize_errors(model, test_loader, device, class_names, model_name="Model"):
    print(f"\n{'='*10} Analyzing Misclassifications: {model_name} {'='*10}")

    # 1. Setup container for errors
    misclassified = {cls: [] for cls in class_names}

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            # Loop over batch to find errors
            for img, true_label, pred_label, prob_vec in zip(inputs, labels, preds, probs):
                if true_label != pred_label:
                    pred_prob = prob_vec[pred_label].item()
                    true_prob = prob_vec[true_label].item()
                    wrongness = pred_prob - true_prob   # larger = confident but wrong

                    cls_name = class_names[true_label.cpu().item()]
                    misclassified[cls_name].append({
                        "img": img.cpu(),
                        "true": true_label.cpu().item(),
                        "pred": pred_label.cpu().item(),
                        "pred_prob": pred_prob,
                        "true_prob": true_prob,
                        "wrongness": wrongness
                    })

    # 2. Sort errors by "wrongness"
    for cls in misclassified:
        misclassified[cls] = sorted(
            misclassified[cls],
            key=lambda x: x["wrongness"],
            reverse=True
        )

    # 3. Plot top 3 errors per class
    for cls in class_names:
        errors = misclassified[cls][:3]  # top 3 worst mistakes

        if len(errors) == 0:
            print(f"Good news! No misclassified images for class: {cls}")
            continue

        # Create a figure for this class
        plt.figure(figsize=(10, 3))
        plt.suptitle(f"[{model_name}] Worst Errors for Class: {cls}", fontsize=14, y=1.05)

        for i, err in enumerate(errors):
            plt.subplot(1, 3, i+1)
            plt.imshow(show_image(err["img"]))
            plt.axis("off")

            true_name = class_names[err['true']]
            pred_name = class_names[err['pred']]

            plt.title(
                f"True: {true_name}\n"
                f"Pred: {pred_name}\n"
                f"Conf: {err['pred_prob']:.2f} vs {err['true_prob']:.2f}",
                fontsize=9
            )

        plt.tight_layout()
        plt.show()

