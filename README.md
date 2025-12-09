# Brain-Tumor-Detection
Project Overview: Brain tumors have been seriously affecting people’s lives. In the US alone, nearly seven hundred thousand of people are living with a brain tumor.
If the tumors are detected at early stage, patients survival rate can jump from 5% to 90%.
The goal of this project is to train models with over 90% accuracy that can help save lives and reduce healthcare costs.

This project detects brain tumors from MRI images using CNN models. Data is sourced from Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data.

## Repository Structure
- **src/**: Core Python code.
  - `main.py`: Run this to prepare data, train, and evaluate models.
  - `utils.py`: Helper functions for data handling, training, etc.
  - `model.py`: Model definitions.
- **data/**: Download data from Kaggle and extract to this folder.
- **checkpoints/**: Saved model weights.
- **demo/**: Jupyter notebook for demo/inference.
- **results/**: Generated plots and results.
- ├── requirements.txt: required depencies 

Setup Instructions:
Step-by-step instructions to set up the environment, including how to install dependencies from requirements.txt.

How to Run:
Instructions to execute a simple demo script (e.g., python demo.py or demo.ipynb), showcasing the core functionality of the project.

Expected Output: A description or example of what users should see after running the demo.

Pre-trained Model Link: Google Drive 
1.
2.

Acknowledgments: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)

## Running the Demo
1. Install dependencies:
   `pip install -r requirements.txt`

2. Run the demo:
   `python demo/demo.py`

The script will:
- download pretrained model from Google Drive
- run predictions on sample MRI images
- save outputs in results/predictions.txt

## To train the model 
1. Install dependencies: `pip install -r requirements.txt`
2. Download the Kaggle dataset ZIP and place it in `data/`.
3. Train models, run `python src/main.py` for full execution.

