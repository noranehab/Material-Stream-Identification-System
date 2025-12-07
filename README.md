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

## Usage

### 1. Data Preparation
```bash
python scripts/prepare_data.py
```

### 2. Train Models
```bash
python scripts/train_models.py
```

### 3. Evaluate Models
```bash
python scripts/evaluate_models.py
```

### 4. Run Real-time Application
```bash
python src/deployment/real_time_app.py
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Deliverables

- ✅ Source code repository
- ✅ Trained model weights (SVM and k-NN)
- 📄 Technical report (to be generated)

