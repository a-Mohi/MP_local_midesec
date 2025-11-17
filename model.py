"""
YOLO-style object detection model for mitotic cell detection
Using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_yolo_model(input_shape=(512, 512, 3), num_classes=1):
    """
    Create a simplified YOLO-style detection model.
    This is a custom implementation suitable for small object detection.

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of object classes (1 for mitotic cells)

    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_shape)

    # Backbone: Feature extraction using MobileNetV2
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Use intermediate layers for multi-scale detection
    # Get features at different scales
    x = backbone(inputs)

    # Detection head
    # Grid size: 16x16 for 512x512 input
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Output: [batch, grid_h, grid_w, 5 + num_classes]
    # 5 = [x, y, w, h, objectness]
    num_outputs = 5 + num_classes
    outputs = layers.Conv2D(num_outputs, 1, padding='same')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='yolo_mitotic_detector')

    return model


def create_simple_detector(input_shape=(512, 512, 3), num_classes=1):
    """
    Create a simpler detection model using Faster R-CNN style approach.
    This uses a CNN backbone + detection heads.

    For simplicity, we'll use a pre-trained feature extractor and add detection layers.
    """
    inputs = keras.Input(shape=input_shape)

    # Feature extractor - using EfficientNetB0 for better accuracy
    try:
        backbone = keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet',
            pooling=None
        )
        print("Loaded EfficientNetB0 with ImageNet weights")
    except Exception as e:
        print(f"Warning: Could not load ImageNet weights ({e})")
        print("Initializing EfficientNetB0 with random weights")
        backbone = keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=None,
            pooling=None
        )

    # Make backbone trainable
    backbone.trainable = True

    # Extract features
    features = backbone(inputs, training=True)

    # Add detection layers
    x = layers.Conv2D(256, 3, padding='same')(features)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Global average pooling + dense layers for classification
    gap = layers.GlobalAveragePooling2D()(x)

    # Classification head
    class_output = layers.Dense(128, activation='relu')(gap)
    class_output = layers.Dropout(0.3)(class_output)
    class_output = layers.Dense(num_classes + 1, activation='softmax', name='class_output')(class_output)

    # Bounding box regression head
    bbox_output = layers.Dense(128, activation='relu')(gap)
    bbox_output = layers.Dropout(0.3)(bbox_output)
    bbox_output = layers.Dense(4, activation='sigmoid', name='bbox_output')(bbox_output)  # [xmin, ymin, xmax, ymax]

    model = keras.Model(inputs=inputs, outputs={'class': class_output, 'bbox': bbox_output})

    return model


def build_detection_model(
    input_shape=(512, 512, 3),
    num_classes=1,
    model_type='efficientnet'
):
    """
    Build object detection model.

    Args:
        input_shape: Input image shape
        num_classes: Number of classes (excluding background)
        model_type: 'efficientnet' or 'mobilenet'

    Returns:
        Compiled Keras model
    """
    if model_type == 'efficientnet':
        model = create_simple_detector(input_shape, num_classes)
    elif model_type == 'mobilenet':
        model = create_yolo_model(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


class DetectionLoss(keras.losses.Loss):
    """Custom loss for object detection."""

    def __init__(self, num_classes=1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.classification_loss = keras.losses.CategoricalCrossentropy()
        self.bbox_loss = keras.losses.Huber()

    def call(self, y_true, y_pred):
        """
        Calculate detection loss.

        Args:
            y_true: Dict with 'class' and 'bbox' ground truth
            y_pred: Dict with 'class' and 'bbox' predictions
        """
        # Classification loss
        class_loss = self.classification_loss(y_true['class'], y_pred['class'])

        # Bounding box loss (only for positive samples)
        # Mask out negative samples
        obj_mask = tf.reduce_max(y_true['class'][:, 1:], axis=1)  # Has object if any class > 0
        obj_mask = tf.cast(obj_mask > 0, tf.float32)

        bbox_loss = self.bbox_loss(y_true['bbox'], y_pred['bbox'])
        bbox_loss = bbox_loss * obj_mask  # Only compute for images with objects
        bbox_loss = tf.reduce_mean(bbox_loss)

        # Total loss
        total_loss = class_loss + bbox_loss

        return total_loss


def compile_model(model, learning_rate=1e-4):
    """
    Compile the detection model with optimizer and losses.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Define losses for each output
    losses = {
        'class': keras.losses.CategoricalCrossentropy(),
        'bbox': keras.losses.Huber()
    }

    # Loss weights
    loss_weights = {
        'class': 1.0,
        'bbox': 1.0
    }

    # Metrics
    metrics = {
        'class': ['accuracy'],
        'bbox': [keras.metrics.MeanAbsoluteError()]
    }

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )

    return model


if __name__ == '__main__':
    # Test model creation
    print("Creating detection model...")
    model = build_detection_model(
        input_shape=(512, 512, 3),
        num_classes=1,
        model_type='efficientnet'
    )

    print("\nModel architecture:")
    model.summary()

    print("\nCompiling model...")
    model = compile_model(model, learning_rate=1e-4)

    print("\nModel compiled successfully!")

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = tf.random.normal((2, 512, 512, 3))
    output = model(dummy_input, training=False)
    print(f"Class output shape: {output['class'].shape}")
    print(f"BBox output shape: {output['bbox'].shape}")
