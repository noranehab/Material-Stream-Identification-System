"""
Model Evaluation Script

Evaluates trained models and generates reports.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.evaluate import evaluate_models


def main():
    """
    Main evaluation function.
    """
    print("Starting model evaluation...")
    evaluate_models("config/config.yaml")


if __name__ == "__main__":
    main()

