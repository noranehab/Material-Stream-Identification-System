"""
Main script to run the complete project pipeline.
Executes all steps in sequence: data preparation, training, and evaluation.
"""

import sys
from pathlib import Path
import subprocess

def run_command(command, description):
    """Run a command and handle errors."""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {description} failed!")
        print(f"Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n\nINTERRUPTED: {description} was cancelled by user.")
        return False

def main():
    """Run the complete project pipeline."""
    print("="*70)
    print("MATERIAL CLASSIFICATION PROJECT - COMPLETE PIPELINE")
    print("="*70)
    
    steps = [
        ("python scripts/prepare_data.py", "Data Preparation"),
        ("python scripts/train_models.py", "Model Training"),
        ("python scripts/evaluate_models.py", "Model Evaluation"),
    ]
    
    print("\nThis will run the complete pipeline:")
    for i, (cmd, desc) in enumerate(steps, 1):
        print(f"  {i}. {desc}")
    
    print("\nNote: Real-time application can be run separately with:")
    print("  python src/deployment/real_time_app.py")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Run each step
    for i, (command, description) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] Running: {description}")
        
        success = run_command(command, description)
        
        if not success:
            print(f"\n{'='*70}")
            print("PIPELINE FAILED")
            print(f"{'='*70}")
            print(f"Failed at step: {description}")
            print("Please check the error messages above.")
            sys.exit(1)
    
    # Success message
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Check results in 'results/' directory")
    print("  2. Review model performance in the output above")
    print("  3. Run real-time app: python src/deployment/real_time_app.py")
    print("="*70)

if __name__ == "__main__":
    main()

