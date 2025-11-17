"""
Script to clear Keras model cache if weights are corrupted
Run this if you get shape mismatch errors when loading pre-trained models
"""

import os
import shutil
from pathlib import Path

def clear_keras_cache():
    """Clear Keras application model cache."""

    # Find Keras cache directory
    cache_dir = Path.home() / '.keras'
    models_dir = cache_dir / 'models'
    datasets_dir = cache_dir / 'datasets'

    print("Keras Cache Cleaner")
    print("=" * 60)
    print(f"Cache directory: {cache_dir}")

    if not cache_dir.exists():
        print("No Keras cache directory found. Nothing to clean.")
        return

    # Show what will be deleted
    total_size = 0
    files_to_delete = []

    if models_dir.exists():
        for file in models_dir.rglob('*'):
            if file.is_file():
                size = file.stat().st_size
                total_size += size
                files_to_delete.append((file, size))

    if not files_to_delete:
        print("No cached model files found.")
        return

    print(f"\nFound {len(files_to_delete)} cached model files")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    print("\nFiles to be deleted:")
    for file, size in files_to_delete[:10]:  # Show first 10
        print(f"  - {file.name} ({size / (1024*1024):.2f} MB)")
    if len(files_to_delete) > 10:
        print(f"  ... and {len(files_to_delete) - 10} more files")

    # Ask for confirmation
    response = input("\nDo you want to delete these files? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        # Delete models directory
        if models_dir.exists():
            shutil.rmtree(models_dir)
            print(f"\n✓ Deleted {models_dir}")

        # Recreate empty directory
        models_dir.mkdir(parents=True, exist_ok=True)
        print("✓ Cache cleared successfully!")
        print("\nNext time you run training, Keras will download fresh weights.")
    else:
        print("Cancelled. No files were deleted.")


if __name__ == '__main__':
    clear_keras_cache()
