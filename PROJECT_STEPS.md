# Project Execution Steps

## Complete Workflow

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data
```bash
python scripts/prepare_data.py
```
**What it does:**
- Loads raw images from `data/raw/dataset/`
- Splits into train (80%) and validation (20%)
- Applies data augmentation to balance classes (~500 samples each)
- Saves processed data to `data/processed/`

**Output:**
- `train_augmented.pkl` - Augmented training images
- `val_data.pkl` - Validation images

### Step 3: Train Models
```bash
python scripts/train_models.py
```
**What it does:**
- Extracts features from all images (LBP + Color Histogram)
- Applies PCA for dimensionality reduction
- Trains SVM classifier
- Trains k-NN classifier
- Compares both models and saves the best one

**Output:**
- `models/svm_model_*.pkl` - SVM model files
- `models/knn_model_*.pkl` - k-NN model files
- `models/best_model_*.pkl` - Best performing model (SVM or k-NN)

### Step 4: Evaluate Models
```bash
python scripts/evaluate_models.py
```
**What it does:**
- Loads trained models
- Evaluates on validation set
- Generates confusion matrices
- Prints detailed performance metrics

**Output:**
- `results/svm_confusion_matrix.png`
- `results/knn_confusion_matrix.png`
- `results/model_comparison.png`
- `results/evaluation_results.json`

### Step 5: Run Real-time Application (Optional)
```bash
python src/deployment/real_time_app.py
```
**What it does:**
- Loads best trained model
- Opens camera feed
- Classifies materials in real-time
- Displays results on screen

**Controls:**
- Press 'q' to quit

## Quick Run (All Steps)
```bash
python run_project.py
```

## Cleanup
Remove unnecessary files:
```bash
python scripts/cleanup.py
```

## Expected Results

After training, you should see:
- **SVM Accuracy**: ~68-72% (primary classes)
- **k-NN Accuracy**: ~61-64% (primary classes)
- **Best Model**: Automatically selected (usually SVM)

## Troubleshooting

**"Data not found" error:**
- Ensure dataset is in `data/raw/dataset/` with class folders (glass, paper, cardboard, etc.)

**"Model not found" error:**
- Run `python scripts/train_models.py` first

**Camera not working:**
- Check camera permissions
- Try changing `camera_index` in `config/config.yaml`

