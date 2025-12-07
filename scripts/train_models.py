"""
Model Training Script

Trains both SVM and k-NN classifiers.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.training.train import train_models


def main():
    """
    Main training function.
    """
    print("Starting model training...")
    train_models("config/config.yaml")


if __name__ == "__main__":
    main()

