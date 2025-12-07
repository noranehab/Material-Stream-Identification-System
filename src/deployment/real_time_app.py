"""
Real-Time Camera Application

Live camera feed processing and material classification.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.feature_extraction import FeatureExtractor
from src.classifiers.svm_classifier import SVMClassifier
from src.classifiers.knn_classifier import KNNClassifier


def run_real_time_app(config_path: str = "config/config.yaml", model_type: str = "best"):
    """
    Run real-time classification application.
    
    Args:
        config_path: Path to configuration file
        model_type: Type of model to use ('svm', 'knn', or 'best')
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    deploy_config = config['deployment']
    classes = config['classes']
    class_names = [classes[i] for i in sorted(classes.keys())]
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(config_path)
    
    # Load model
    print(f"Loading {model_type} model...")
    if model_type == "svm":
        classifier = SVMClassifier(config_path)
        classifier.load("models/svm_model")
    elif model_type == "knn":
        classifier = KNNClassifier(config_path)
        classifier.load("models/knn_model")
    else:  # best model
        # TODO: Load best model based on evaluation results
        classifier = SVMClassifier(config_path)
        classifier.load("models/best_model")
    
    # Initialize camera
    cap = cv2.VideoCapture(deploy_config['camera_index'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, deploy_config['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, deploy_config['frame_height'])
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera initialized. Press 'q' to quit.")
    
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # TODO: Implement real-time processing
        # 1. Preprocess frame (resize, normalize)
        # 2. Extract features
        # 3. Predict class
        # 4. Apply rejection mechanism
        # 5. Display results on frame
        
        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb,
            tuple(config['dataset']['image_size'])
        )
        
        # Extract features
        features = feature_extractor.extract_features(frame_resized)
        features = features.reshape(1, -1)
        
        # Predict
        if hasattr(classifier, 'predict_with_rejection'):
            prediction, confidence, rejected = classifier.predict_with_rejection(features)
            pred_class = prediction[0]
            conf = confidence[0] if confidence is not None else 0.0
            is_rejected = rejected[0] if rejected is not None else False
        else:
            prediction = classifier.predict(features)
            pred_class = prediction[0]
            conf = 1.0
            is_rejected = False
        
        # Display results
        class_name = class_names[pred_class] if pred_class < len(class_names) else "Unknown"
        
        # Draw prediction on frame
        text = f"Class: {class_name}"
        if deploy_config['display_confidence']:
            text += f" | Confidence: {conf:.2f}"
        if is_rejected:
            text += " [REJECTED]"
        
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if not is_rejected else (0, 0, 255),
            2
        )
        
        # Display FPS
        if deploy_config['display_fps']:
            fps_counter += 1
            if fps_counter >= 30:
                fps = 30 / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Show frame
        cv2.imshow('Material Classification', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    run_real_time_app(model_type="best")

