"""
Training script for mitotic cell detection
"""

import tensorflow as tf
from tensorflow import keras
import json
import numpy as np
import os
from datetime import datetime

from model import build_detection_model, compile_model


def create_simple_dataset(annotations_path, batch_size=8, image_size=(512, 512), augment=True):
    """
    Create a simpler dataset that works with the detection model.
    Returns (image, {'class': class_label, 'bbox': bbox_coords})
    """
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Extract image paths and prepare simple data
    image_paths = []
    has_objects = []
    first_bboxes = []

    for ann in annotations:
        image_paths.append(ann['image_path'])
        if len(ann['bboxes']) > 0:
            has_objects.append(1)  # Has object
            first_bboxes.append(ann['bboxes'][0]['bbox'])
        else:
            has_objects.append(0)  # No object
            first_bboxes.append([0.0, 0.0, 0.0, 0.0])

    def load_and_process(image_path, has_object, bbox):
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, image_size)

        # Class label (one-hot)
        if has_object > 0:
            class_label = tf.constant([0.0, 1.0], dtype=tf.float32)
        else:
            class_label = tf.constant([1.0, 0.0], dtype=tf.float32)

        # Simple augmentation
        if augment:
            if tf.random.uniform([]) > 0.5:
                image = tf.image.flip_left_right(image)
                if has_object > 0:
                    xmin, ymin, xmax, ymax = tf.unstack(bbox)
                    bbox = tf.stack([1.0 - xmax, ymin, 1.0 - xmin, ymax])

            if tf.random.uniform([]) > 0.5:
                image = tf.image.flip_up_down(image)
                if has_object > 0:
                    xmin, ymin, xmax, ymax = tf.unstack(bbox)
                    bbox = tf.stack([xmin, 1.0 - ymax, xmax, 1.0 - ymin])

            # Color augmentation
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.clip_by_value(image, 0.0, 1.0)

        return image, class_label, bbox

    # Create dataset from lists
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, has_objects, first_bboxes))

    dataset = dataset.map(
        lambda path, has_obj, bbox: load_and_process(path, has_obj, bbox),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Restructure to match model output format
    dataset = dataset.map(lambda img, cls, bbox: (img, {'class': cls, 'bbox': bbox}))

    if augment:
        dataset = dataset.shuffle(len(annotations))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(annotations) // batch_size


def train_model(
    epochs=50,
    batch_size=8,
    image_size=(512, 512),
    learning_rate=1e-4,
    model_dir='models'
):
    """
    Train the mitotic cell detection model.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        image_size: Input image size
        learning_rate: Learning rate
        model_dir: Directory to save models
    """
    print("=" * 80)
    print("MITOTIC CELL DETECTION - TRAINING")
    print("=" * 80)

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(model_dir, f'run_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Create datasets
    print("\nCreating datasets...")
    train_ds, train_steps = create_simple_dataset(
        'data/train_annotations.json',
        batch_size=batch_size,
        image_size=image_size,
        augment=True
    )

    val_ds, val_steps = create_simple_dataset(
        'data/test_annotations.json',
        batch_size=batch_size,
        image_size=image_size,
        augment=False
    )

    print(f"Training samples: {train_steps * batch_size}")
    print(f"Validation samples: {val_steps * batch_size}")

    # Create model
    print("\nBuilding model...")
    model = build_detection_model(
        input_shape=(*image_size, 3),
        num_classes=1,
        model_type='efficientnet'
    )

    # Compile model
    print("Compiling model...")
    model = compile_model(model, learning_rate=learning_rate)

    # Print model summary
    print("\nModel summary:")
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(save_dir, 'training_log.csv')
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_dir, 'logs')
        )
    ]

    # Train model
    print("\nStarting training...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Image size: {image_size}")
    print(f"Model will be saved to: {save_dir}")
    print("=" * 80)

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Save training history
    history_path = os.path.join(save_dir, 'history.json')
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # Print final metrics
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    if 'class_accuracy' in history.history:
        print(f"Final training class accuracy: {history.history['class_accuracy'][-1]:.4f}")
        print(f"Final validation class accuracy: {history.history['val_class_accuracy'][-1]:.4f}")

    return model, history, save_dir


if __name__ == '__main__':
    # Configuration
    CONFIG = {
        'epochs': 100,
        'batch_size': 8,
        'image_size': (512, 512),
        'learning_rate': 1e-4,
        'model_dir': 'models'
    }

    print("Training configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()

    # Train model
    model, history, save_dir = train_model(**CONFIG)

    print(f"\nAll files saved to: {save_dir}")
    print("To monitor training with TensorBoard:")
    print(f"  tensorboard --logdir {os.path.join(save_dir, 'logs')}")
