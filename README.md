# Material Classification Vision System

An end-to-end feature-based vision system for classifying materials into seven categories using traditional machine learning approaches.

## Project Overview

This project implements a complete pipeline for material classification:
- **Data Augmentation**: Increases training data by 30%+ and balances classes to ~500 samples each
- **Feature Extraction**: Converts raw images to fixed-size numerical feature vectors
- **Classifiers**: Implements both SVM and k-NN classifiers
- **Real-time Deployment**: Live camera feed classification application

## Material Classes

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | Glass | Amorphous solid materials (bottles, jars) |
| 1 | Paper | Thin cellulose-based materials (newspapers, office paper) |
| 2 | Cardboard | Multi-layer cellulose fiber materials |
| 3 | Plastic | High-molecular-weight organic compounds |
| 4 | Metal | Elemental or compound metallic substances |
| 5 | Trash | Non-recyclable or contaminated waste |
| 6 | Unknown | Out-of-distribution or uncertain items |

## Project Structure

```
ML_Project/
├── config/              # Configuration files
├── data/               # Dataset directories
│   ├── raw/           # Original dataset
│   ├── processed/     # Preprocessed images
│   └── augmented/     # Augmented images
├── src/               # Source code
│   ├── classifiers/   # SVM and k-NN implementations
│   ├── training/      # Training scripts
│   ├── evaluation/    # Evaluation utilities
│   └── deployment/    # Real-time application
├── models/            # Trained model files
├── scripts/           # Utility scripts
└── utils/             # Helper functions
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ML_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start Guide

### Option 1: Run Complete Pipeline (Recommended)
Run all steps automatically:
```bash
python run_project.py
```
This will execute:
1. Data preparation
2. Model training
3. Model evaluation

### Option 2: Run Steps Manually

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Prepare Data
Loads raw images, applies data augmentation, and prepares training/validation sets.
```bash
python scripts/prepare_data.py
```
**Expected Output:**
- Augmented training data: ~3000 samples (500 per class)
- Validation data: ~373 samples
- Files saved to `data/processed/`

#### Step 3: Train Models
Trains both SVM and k-NN classifiers with optimized hyperparameters.
```bash
python scripts/train_models.py
```
**Expected Output:**
- Trained SVM model saved to `models/svm_model_*.pkl`
- Trained k-NN model saved to `models/knn_model_*.pkl`
- Best model (SVM or k-NN) saved to `models/best_model_*.pkl`
- Training takes 5-10 minutes depending on your system

#### Step 4: Evaluate Models
Evaluates trained models and generates performance reports.
```bash
python scripts/evaluate_models.py
```
**Expected Output:**
- Detailed accuracy comparison between SVM and k-NN
- Confusion matrices saved to `results/`
- Classification reports printed to console
- Best model automatically selected based on primary classes accuracy

#### Step 5: Run Real-time Application (Optional)
Launches live camera feed for real-time material classification.
```bash
python src/deployment/real_time_app.py
```
**Note:** Requires a webcam/camera connected to your system. Press 'q' to quit.

## Cleanup

To remove temporary files, test scripts, and optimization scripts:
```bash
python scripts/cleanup.py
```

This removes:
- Temporary config files
- Test scripts
- Optimization scripts
- Old model files
- Python cache files (`__pycache__`)

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Deliverables

- ✅ Source code repository
- ✅ Trained model weights (SVM and k-NN)
- 📄 Technical report (to be generated)

