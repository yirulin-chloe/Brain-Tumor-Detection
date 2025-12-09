# Load the pre-trained model from google drive.
# Run on sample inputs (e.g., from the demo/ folder).
# Produce output in the results/ folder (e.g., generated images, predictions, etc.).

import os
import sys
import torch
import gdown
from PIL import Image
from torchvision import transforms

# --- Setup Paths ---
# Get the absolute path of the demo folder and the project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'predictions.txt')

# Add 'src' to system path so we can import the model
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

# Try importing the model architecture
try:
    from model import BrainTumorClassifier
except ImportError:
    print("Error: Could not import 'BrainTumorClassifier' from src/model.py.")
    print("Make sure your project structure is correct and src/model.py exists.")
    sys.exit(1)

# --- Configuration ---
# Google Drive File ID 
# Link 1: 1sIzWNmRUYPVwcXj6UimuyV_FeYRG2whj
# Link 2: 1uty1EA7wIoS4mL3ZSnGpAv0u0x8UaDnl
FILE_ID = '1sIzWNmRUYPVwcXj6UimuyV_FeYRG2whj' 
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def download_model():
    """Downloads the pretrained model from Google Drive if not present."""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return

    print(f"Downloading model from Google Drive (ID: {FILE_ID})...")
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download model: {e}")
        sys.exit(1)

def get_demo_images():
    """Retrieves all .jpg/.png images from the demo folder."""
    valid_extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(CURRENT_DIR) if f.lower().endswith(valid_extensions)]
    return [os.path.join(CURRENT_DIR, img) for img in images]

def run_predictions():
    """Loads model, runs inference on demo images, and saves results."""
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 2. Load Model Architecture
    try:
        model = BrainTumorClassifier(num_classes=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Tip: Ensure src/model.py matches the architecture used to train this .pth file.")
        sys.exit(1)

    # 3. Define Transforms (Must match training transforms)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Process Images
    image_paths = get_demo_images()
    if not image_paths:
        print("No images found in the demo/ folder. Please add some .jpg or .png files.")
        return

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"Processing {len(image_paths)} images...")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("Filename\tPredicted Class\tConfidence\n")
        f.write("-" * 50 + "\n")
        
        for img_path in image_paths:
            try:
                # Preprocess
                img = Image.open(img_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)

                # Result
                file_name = os.path.basename(img_path)
                class_name = CLASSES[predicted_idx.item()]
                conf_score = confidence.item()

                # Write to file and print
                result_str = f"{file_name}\t{class_name}\t{conf_score:.4f}"
                print(result_str)
                f.write(result_str + "\n")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"\nPredictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    download_model()
    run_predictions()