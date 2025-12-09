"""
Cleanup script to remove unnecessary files from the project.
"""

import os
from pathlib import Path

def cleanup_unnecessary_files():
    """Remove temporary and test files."""
    project_root = Path(__file__).parent.parent
    
    files_to_remove = [
        # Temporary config files
        "config/temp_config.yaml",
        "config/temp_optimize.yaml",
        "config/optimized_config.yaml",  # Keep if you want, but not needed for main run
        
        # Test scripts
        "scripts/test_data_loading.py",
        "scripts/test_feature_methods.py",
        "scripts/test_training.py",
        
        # Optimization scripts (optional - keep if you want to re-optimize)
        "scripts/optimize_accuracy.py",
        "scripts/optimize_to_85.py",
        "scripts/advanced_optimization.py",
        
        # Planning/documentation files
        "NEXT_STEPS.md",
        
        # Old optimized models (keep best models only)
        "models/optimized_svm_model.pkl",
        "models/optimized_svm_scaler.pkl",
        "models/optimized_knn_model.pkl",
        "models/optimized_knn_scaler.pkl",
        "models/optimized_pca.pkl",
        
        # Ensemble model files (not needed)
        "models/best_model_ensemble_svm.pkl",
        "models/best_model_ensemble_knn.pkl",
        "models/best_model_ensemble_svm_scaler.pkl",
        "models/best_model_ensemble_knn_scaler.pkl",
        "models/best_model_ensemble_pca.pkl",
    ]
    
    removed = []
    not_found = []
    
    print("="*60)
    print("CLEANUP: Removing Unnecessary Files")
    print("="*60)
    
    for file_path in files_to_remove:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                full_path.unlink()
                removed.append(file_path)
                print(f"  [REMOVED] {file_path}")
            except Exception as e:
                print(f"  [ERROR] Could not remove {file_path}: {e}")
        else:
            not_found.append(file_path)
    
    print("\n" + "="*60)
    print(f"Summary: {len(removed)} files removed")
    if not_found:
        print(f"Note: {len(not_found)} files not found (already removed or don't exist)")
    print("="*60)
    
    # Clean up __pycache__ directories
    print("\nCleaning up __pycache__ directories...")
    pycache_dirs = list(project_root.rglob("__pycache__"))
    for pycache_dir in pycache_dirs:
        try:
            import shutil
            shutil.rmtree(pycache_dir)
            print(f"  [REMOVED] {pycache_dir.relative_to(project_root)}")
        except Exception as e:
            print(f"  [ERROR] Could not remove {pycache_dir}: {e}")
    
    print("\nCleanup completed!")

if __name__ == "__main__":
    cleanup_unnecessary_files()

