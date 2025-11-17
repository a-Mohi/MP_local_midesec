"""
Inference and evaluation script for mitotic cell detection
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_model(model_path):
    """Load trained model."""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    return model


def preprocess_image(image_path, image_size=(512, 512)):
    """
    Load and preprocess image for inference.

    Args:
        image_path: Path to image
        image_size: Target size

    Returns:
        Preprocessed image tensor and original image
    """
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    original_image = image.numpy()

    # Preprocess
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    image = tf.expand_dims(image, 0)  # Add batch dimension

    return image, original_image


def predict(model, image_path, image_size=(512, 512), confidence_threshold=0.5):
    """
    Run inference on a single image.

    Args:
        model: Trained model
        image_path: Path to image
        image_size: Input image size
        confidence_threshold: Confidence threshold for detection

    Returns:
        Dictionary with predictions
    """
    # Preprocess image
    image, original_image = preprocess_image(image_path, image_size)

    # Predict
    predictions = model.predict(image, verbose=0)

    # Parse predictions
    class_probs = predictions['class'][0]
    bbox_pred = predictions['bbox'][0]

    # Check if object is detected
    has_object = class_probs[1] > confidence_threshold

    result = {
        'image_path': image_path,
        'has_object': bool(has_object),
        'confidence': float(class_probs[1]),
        'bbox': bbox_pred.tolist() if has_object else None,
        'class_probs': class_probs.tolist(),
        'original_image': original_image
    }

    return result


def visualize_prediction(result, save_path=None, show=True):
    """
    Visualize prediction with bounding box.

    Args:
        result: Prediction result dictionary
        save_path: Path to save visualization
        show: Whether to display the plot
    """
    image = result['original_image']
    h, w = image.shape[:2]

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    if result['has_object'] and result['bbox'] is not None:
        bbox = result['bbox']
        xmin, ymin, xmax, ymax = bbox

        # Convert normalized coords to pixel coords
        xmin_px = int(xmin * w)
        ymin_px = int(ymin * h)
        xmax_px = int(xmax * w)
        ymax_px = int(ymax * h)

        # Draw bounding box
        rect = patches.Rectangle(
            (xmin_px, ymin_px),
            xmax_px - xmin_px,
            ymax_px - ymin_px,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        label = f"Mitotic Cell: {result['confidence']:.2f}"
        ax.text(
            xmin_px, ymin_px - 10,
            label,
            color='white',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7)
        )

    ax.axis('off')
    ax.set_title(f"Detection Result - {os.path.basename(result['image_path'])}")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def evaluate_on_dataset(model, annotations_path, image_size=(512, 512), confidence_threshold=0.5, output_dir='results'):
    """
    Evaluate model on entire dataset.

    Args:
        model: Trained model
        annotations_path: Path to annotations JSON
        image_size: Input image size
        confidence_threshold: Confidence threshold
        output_dir: Directory to save results

    Returns:
        Evaluation metrics
    """
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    all_results = []

    print(f"\nEvaluating on {len(annotations)} images...")
    print("=" * 80)

    for i, annotation in enumerate(annotations):
        image_path = annotation['image_path']
        has_ground_truth = annotation['num_objects'] > 0

        # Predict
        result = predict(model, image_path, image_size, confidence_threshold)

        # Compare with ground truth
        predicted_object = result['has_object']

        if has_ground_truth and predicted_object:
            true_positives += 1
            status = "TP"
        elif not has_ground_truth and not predicted_object:
            true_negatives += 1
            status = "TN"
        elif has_ground_truth and not predicted_object:
            false_negatives += 1
            status = "FN"
        else:  # not has_ground_truth and predicted_object
            false_positives += 1
            status = "FP"

        result['ground_truth'] = has_ground_truth
        result['status'] = status
        all_results.append(result)

        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == len(annotations):
            print(f"Processed {i + 1}/{len(annotations)} images")

        # Save visualization for first 20 images or misclassifications
        if i < 20 or status in ['FP', 'FN']:
            vis_path = os.path.join(vis_dir, f"{os.path.basename(image_path).replace('.jpg', '')}_{status}.png")
            visualize_prediction(result, save_path=vis_path, show=False)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(annotations)

    metrics = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'num_images': len(annotations)
    }

    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'predictions': [
                {
                    'image_path': r['image_path'],
                    'has_object': r['has_object'],
                    'confidence': r['confidence'],
                    'ground_truth': r['ground_truth'],
                    'status': r['status']
                }
                for r in all_results
            ]
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total images: {len(annotations)}")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"True Negatives: {true_negatives}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print("=" * 80)

    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {vis_dir}")

    return metrics, all_results


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inference.py <model_path> [image_path]")
        print("\nExamples:")
        print("  # Evaluate on test set")
        print("  python inference.py models/run_*/best_model.h5")
        print("\n  # Predict on single image")
        print("  python inference.py models/run_*/best_model.h5 data/test_images/P00_00.jpg")
        sys.exit(1)

    model_path = sys.argv[1]

    # Load model
    model = load_model(model_path)

    if len(sys.argv) == 3:
        # Single image prediction
        image_path = sys.argv[2]
        print(f"\nRunning inference on: {image_path}")

        result = predict(model, image_path, confidence_threshold=0.5)

        print("\nPrediction:")
        print(f"  Has object: {result['has_object']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        if result['bbox']:
            print(f"  BBox: {result['bbox']}")

        visualize_prediction(result)

    else:
        # Evaluate on test set
        print("\nEvaluating on test set...")
        metrics, results = evaluate_on_dataset(
            model,
            'data/test_annotations.json',
            image_size=(512, 512),
            confidence_threshold=0.5,
            output_dir='results'
        )
