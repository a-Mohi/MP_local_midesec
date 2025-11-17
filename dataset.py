"""
TensorFlow dataset pipeline for mitotic cell detection
"""

import tensorflow as tf
import json
import os
from typing import Dict, Tuple, List


class MitoticCellDataset:
    """Dataset loader for mitotic cell detection."""

    def __init__(
        self,
        annotations_path: str,
        image_size: Tuple[int, int] = (512, 512),
        batch_size: int = 4,
        augment: bool = True
    ):
        self.annotations_path = annotations_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment

        # Load annotations
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

        print(f"Loaded {len(self.annotations)} images from {annotations_path}")

    def _parse_annotation(self, annotation: Dict) -> Tuple[str, List, int]:
        """Parse annotation dict to get image path, bboxes, and count."""
        image_path = annotation['image_path']
        bboxes = annotation['bboxes']

        # Extract bbox coordinates [ymin, xmin, ymax, xmax] format for TF
        bbox_list = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox['bbox']
            # Convert to [ymin, xmin, ymax, xmax]
            bbox_list.append([ymin, xmin, ymax, xmax])

        return image_path, bbox_list, len(bboxes)

    def _load_image(self, image_path: str) -> tf.Tensor:
        """Load and preprocess image."""
        # Read image file
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        # Convert to float32 and normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # Resize to target size
        image = tf.image.resize(image, self.image_size)

        return image

    def _augment_image(self, image: tf.Tensor, bboxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply data augmentation."""
        # Random horizontal flip
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
            # Flip bboxes: [ymin, xmin, ymax, xmax] -> [ymin, 1-xmax, ymax, 1-xmin]
            if tf.shape(bboxes)[0] > 0:
                ymin, xmin, ymax, xmax = tf.unstack(bboxes, axis=1)
                bboxes = tf.stack([ymin, 1.0 - xmax, ymax, 1.0 - xmin], axis=1)

        # Random vertical flip
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_up_down(image)
            # Flip bboxes: [ymin, xmin, ymax, xmax] -> [1-ymax, xmin, 1-ymin, xmax]
            if tf.shape(bboxes)[0] > 0:
                ymin, xmin, ymax, xmax = tf.unstack(bboxes, axis=1)
                bboxes = tf.stack([1.0 - ymax, xmin, 1.0 - ymin, xmax], axis=1)

        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.2)

        # Random contrast
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # Random saturation
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

        # Random hue
        image = tf.image.random_hue(image, max_delta=0.1)

        # Clip values to [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, bboxes

    def _process_example(self, annotation_dict):
        """Process a single example."""
        # Parse annotation
        image_path = annotation_dict['image_path']
        bboxes = annotation_dict['bboxes']

        # Load image
        image = self._load_image(image_path)

        # Convert bboxes to tensor
        if len(bboxes) > 0:
            bbox_coords = []
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox['bbox']
                bbox_coords.append([ymin, xmin, ymax, xmax])
            bboxes_tensor = tf.constant(bbox_coords, dtype=tf.float32)
        else:
            bboxes_tensor = tf.zeros((0, 4), dtype=tf.float32)

        # Apply augmentation if enabled
        if self.augment:
            image, bboxes_tensor = self._augment_image(image, bboxes_tensor)

        # Prepare labels (classes are all 0 for mitotic cells)
        num_boxes = tf.shape(bboxes_tensor)[0]
        classes = tf.zeros((num_boxes,), dtype=tf.int32)

        return {
            'image': image,
            'objects': {
                'bbox': bboxes_tensor,
                'label': classes,
                'num_objects': num_boxes
            }
        }

    def build_dataset(self) -> tf.data.Dataset:
        """Build TensorFlow dataset."""
        # Create dataset from annotations
        dataset = tf.data.Dataset.from_tensor_slices(self.annotations)

        # Map processing function
        dataset = dataset.map(
            lambda x: tf.py_function(
                func=self._process_example,
                inp=[x],
                Tout={
                    'image': tf.float32,
                    'objects': {
                        'bbox': tf.float32,
                        'label': tf.int32,
                        'num_objects': tf.int32
                    }
                }
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Filter out images with no objects (optional)
        # dataset = dataset.filter(lambda x: x['objects']['num_objects'] > 0)

        # Shuffle and batch
        if self.augment:
            dataset = dataset.shuffle(buffer_size=len(self.annotations))

        # Padded batch for variable number of boxes
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes={
                'image': [*self.image_size, 3],
                'objects': {
                    'bbox': [None, 4],
                    'label': [None],
                    'num_objects': []
                }
            },
            padding_values={
                'image': 0.0,
                'objects': {
                    'bbox': 0.0,
                    'label': -1,
                    'num_objects': 0
                }
            }
        )

        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_steps_per_epoch(self) -> int:
        """Calculate steps per epoch."""
        return len(self.annotations) // self.batch_size


def create_train_dataset(
    batch_size: int = 4,
    image_size: Tuple[int, int] = (512, 512)
) -> tf.data.Dataset:
    """Create training dataset."""
    loader = MitoticCellDataset(
        annotations_path='data/train_annotations.json',
        image_size=image_size,
        batch_size=batch_size,
        augment=True
    )
    return loader.build_dataset(), loader.get_steps_per_epoch()


def create_val_dataset(
    batch_size: int = 4,
    image_size: Tuple[int, int] = (512, 512)
) -> tf.data.Dataset:
    """Create validation dataset."""
    loader = MitoticCellDataset(
        annotations_path='data/test_annotations.json',
        image_size=image_size,
        batch_size=batch_size,
        augment=False
    )
    return loader.build_dataset(), loader.get_steps_per_epoch()


if __name__ == '__main__':
    # Test dataset creation
    print("Creating training dataset...")
    train_ds, train_steps = create_train_dataset(batch_size=2, image_size=(512, 512))

    print("\nTesting training dataset...")
    for batch in train_ds.take(1):
        print(f"Image shape: {batch['image'].shape}")
        print(f"BBox shape: {batch['objects']['bbox'].shape}")
        print(f"Label shape: {batch['objects']['label'].shape}")
        print(f"Num objects: {batch['objects']['num_objects']}")
        print(f"BBox example: {batch['objects']['bbox'][0][:3]}")  # First 3 boxes

    print("\nCreating validation dataset...")
    val_ds, val_steps = create_val_dataset(batch_size=2, image_size=(512, 512))

    print(f"\nDataset created successfully!")
    print(f"Train steps per epoch: {train_steps}")
    print(f"Val steps per epoch: {val_steps}")
