import os
import sys
import torch
import gdown
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
import torch.nn as nn

# --- 1. Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Add src to path
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

# Map Model Names to Drive IDs
MODELS_CONFIG = {
    'ResNet18': '1sIzWNmRUYPVwcXj6UimuyV_FeYRG2whj',     
    'Baseline_CNN': '1uty1EA7wIoS4mL3ZSnGpAv0u0x8UaDnl'  
}

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- 2. Helper Functions ---
def get_true_label(filename):
    """
    Extracts the 'True Type' from the filename if it exists.
    Example: 'glioma_45.jpg' -> 'glioma'
    """
    filename_lower = filename.lower()
    for cls in CLASSES:
        if cls in filename_lower:
            return cls
    return "Unknown"

def load_model_instance(model_name, device):
    """Loads the architecture and weights."""
    model = None
    if model_name == 'Baseline_CNN':
        try:
            from model import BaselineCNN
            model = BaselineCNN(num_classes=len(CLASSES))
        except ImportError:
            print("Error: Could not import BaselineCNN from src/model.py")
            return None
    elif model_name == 'ResNet18':
        model = models.resnet18(weights=None) 
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
        
    weights_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.pth")
    if not os.path.exists(weights_path):
        print(f"Weights missing: {weights_path}")
        return None
        
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception:
        print(f"Error loading weights for {model_name}")
        return None

    model.to(device)
    model.eval()
    return model

def download_models():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for name, file_id in MODELS_CONFIG.items():
        path = os.path.join(CHECKPOINT_DIR, f"{name}.pth")
        if not os.path.exists(path):
            print(f"Downloading {name}...")
            gdown.download(f'https://drive.google.com/uc?id={file_id}', path, quiet=False)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    download_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    images = [f for f in os.listdir(CURRENT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print("No images found in demo/ folder.")
        return

    # Output file
    output_path = os.path.join(RESULTS_DIR, 'predictions.txt')
    
    with open(output_path, 'w') as f:
        # Table Header
        header = f"{'Model':<15} | {'Image Name':<20} | {'True Type':<12} | {'Predict':<12} | {'Conf':<6}"
        print("\n" + header)
        print("-" * 75)
        f.write(header + "\n")

        # Run for each model
        for model_name in MODELS_CONFIG.keys():
            model = load_model_instance(model_name, device)
            if not model: continue
            
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            for img_file in images:
                try:
                    img_path = os.path.join(CURRENT_DIR, img_file)
                    true_label = get_true_label(img_file)
                    
                    # Inference
                    pil_img = Image.open(img_path).convert('RGB')
                    tensor = preprocess(pil_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        out = model(tensor)
                        probs = torch.nn.functional.softmax(out, dim=1)
                        conf, idx = torch.max(probs, 1)
                    
                    pred_label = CLASSES[idx.item()]
                    conf_val = conf.item() * 100
                    
                    # 1. Console & Text File Output
                    # Format: Image "type", Predict: "type" (Conf: 99.9%)
                    row = f"{model_name:<15} | {img_file:<20} | {true_label:<12} | {pred_label:<12} | {conf_val:.1f}%"
                    print(row)
                    f.write(row + "\n")
                    
                    # 2. Generate Labeled Image
                    # Draw text on image: "Pred: glioma (98%)"
                    draw = ImageDraw.Draw(pil_img)
                    
                    # Text Settings
                    text_str = f"Model: {model_name}\nTrue: {true_label}\nPred: {pred_label} ({conf_val:.1f}%)"
                    
                    # Draw Red if wrong, Green if correct (if True Label is known)
                    color = "green" if (true_label == "Unknown" or true_label == pred_label) else "red"
                    
                    # Draw a black background rectangle for text visibility
                    draw.rectangle([(0, 0), (160, 50)], fill="black")
                    draw.text((5, 5), text_str, fill=color)
                    
                    # Save
                    save_name = f"{model_name}_{img_file}"
                    pil_img.save(os.path.join(RESULTS_DIR, save_name))
                    
                except Exception as e:
                    print(f"Error: {e}")
            
            del model
            torch.cuda.empty_cache()
            print("-" * 75)

    print(f"\n Predictions saved to {output_path}")
    print(f" Labeled images saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()