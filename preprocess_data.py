"""
Data preprocessing script for mitotic cell detection
Converts polygon annotations from CSV to bounding boxes
"""

import os
import csv
import numpy as np
from PIL import Image
import json


def parse_polygon_csv(csv_path):
    """
    Parse CSV file containing polygon coordinates.
    Each row contains x,y coordinate pairs for one polygon.
    Returns list of polygons, where each polygon is list of (x,y) tuples.
    """
    polygons = []

    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split by comma and parse coordinate pairs
            coords = line.split(',')
            coords = [c.strip() for c in coords if c.strip()]

            if len(coords) < 6:  # Need at least 3 points (6 values)
                continue

            # Group into (x, y) pairs
            polygon = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    try:
                        x = int(coords[i])
                        y = int(coords[i + 1])
                        polygon.append((x, y))
                    except ValueError:
                        break

            if len(polygon) >= 3:  # Valid polygon needs at least 3 points
                polygons.append(polygon)

    return polygons


def polygon_to_bbox(polygon):
    """
    Convert polygon to bounding box.
    Args:
        polygon: List of (x, y) tuples
    Returns:
        (xmin, ymin, xmax, ymax)
    """
    if not polygon:
        return None

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    return (xmin, ymin, xmax, ymax)


def normalize_bbox(bbox, image_width, image_height):
    """
    Normalize bounding box coordinates to [0, 1].
    Args:
        bbox: (xmin, ymin, xmax, ymax) in pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
    Returns:
        (xmin, ymin, xmax, ymax) normalized to [0, 1]
    """
    xmin, ymin, xmax, ymax = bbox
    return (
        xmin / image_width,
        ymin / image_height,
        xmax / image_width,
        ymax / image_height
    )


def process_dataset(data_dir, split='train'):
    """
    Process entire dataset split.
    Args:
        data_dir: Path to data directory
        split: 'train' or 'test'
    Returns:
        List of dicts with 'image_path', 'bboxes', 'image_width', 'image_height'
    """
    if split == 'train':
        images_dir = os.path.join(data_dir, 'train images')
        pattern = 'P0_'
    else:
        images_dir = os.path.join(data_dir, 'test images')
        pattern = 'P00_'

    annotations = []

    # Get all jpg files
    jpg_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') and pattern in f]
    jpg_files.sort()

    for jpg_file in jpg_files:
        base_name = jpg_file.replace('.jpg', '')
        image_path = os.path.join(images_dir, jpg_file)
        csv_path = os.path.join(images_dir, f'{base_name}.csv')

        if not os.path.exists(csv_path):
            continue

        # Get image dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Parse polygons and convert to bounding boxes
        polygons = parse_polygon_csv(csv_path)
        bboxes = []

        for polygon in polygons:
            bbox = polygon_to_bbox(polygon)
            if bbox:
                # Normalize coordinates
                norm_bbox = normalize_bbox(bbox, image_width, image_height)
                bboxes.append({
                    'bbox': norm_bbox,
                    'class': 0,  # Single class: mitotic cell
                    'class_name': 'mitotic_cell'
                })

        annotations.append({
            'image_path': image_path,
            'image_id': base_name,
            'bboxes': bboxes,
            'image_width': image_width,
            'image_height': image_height,
            'num_objects': len(bboxes)
        })

    return annotations


def save_annotations(annotations, output_path):
    """Save annotations to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved {len(annotations)} annotations to {output_path}")


def print_dataset_stats(annotations, split_name):
    """Print statistics about the dataset."""
    print(f"\n{split_name} Dataset Statistics:")
    print(f"  Total images: {len(annotations)}")

    total_objects = sum(ann['num_objects'] for ann in annotations)
    print(f"  Total objects: {total_objects}")
    print(f"  Average objects per image: {total_objects / len(annotations):.2f}")

    objects_per_image = [ann['num_objects'] for ann in annotations]
    print(f"  Min objects per image: {min(objects_per_image)}")
    print(f"  Max objects per image: {max(objects_per_image)}")

    # Get image size (assuming all same size)
    if annotations:
        print(f"  Image size: {annotations[0]['image_width']}x{annotations[0]['image_height']}")


if __name__ == '__main__':
    # Process training data
    train_annotations = process_dataset('data', split='train')
    save_annotations(train_annotations, 'data/train_annotations.json')
    print_dataset_stats(train_annotations, 'Training')

    # Process test data
    test_annotations = process_dataset('data', split='test')
    save_annotations(test_annotations, 'data/test_annotations.json')
    print_dataset_stats(test_annotations, 'Test')

    print("\nPreprocessing complete!")
    print("Annotation files saved:")
    print("  - data/train_annotations.json")
    print("  - data/test_annotations.json")
