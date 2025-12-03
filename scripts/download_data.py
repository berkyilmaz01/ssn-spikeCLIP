"""
Download and prepare N-Caltech101 dataset for SpikeCLIP training

This script downloads the N-Caltech101 event dataset and corresponding
Caltech101 image dataset, then prepares them for training.

Note: N-Caltech101 dataset may require manual download due to Baidu restrictions.
This script provides instructions and helps organize the data.
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dataset URLs (these may need to be updated)
N_CALTECH101_URL = "https://www.garrickorchard.com/datasets/n-caltech101"
CALTECH101_IMAGES_URL = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"

# Default dataset directory
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets"


def print_instructions():
    """Print manual download instructions."""
    print("=" * 80)
    print("N-Caltech101 Dataset Download Instructions")
    print("=" * 80)
    print()
    print("Due to Baidu restrictions, N-Caltech101 may require manual download.")
    print()
    print("STEP 1: Download N-Caltech101 Event Data")
    print("-" * 80)
    print("1. Visit: https://www.garrickorchard.com/datasets/n-caltech101")
    print("2. Download the dataset (may require Baidu account)")
    print("3. Extract to: datasets/N-Caltech101/Caltech101/Caltech101/")
    print("   Expected structure:")
    print("   datasets/N-Caltech101/Caltech101/Caltech101/")
    print("   ├── accordion/")
    print("   │   ├── image_0001.bin")
    print("   │   └── ...")
    print("   ├── airplanes/")
    print("   └── ...")
    print()
    print("STEP 2: Download Caltech101 Image Data")
    print("-" * 80)
    print("1. Visit: http://www.vision.caltech.edu/Image_Datasets/Caltech101/")
    print("2. Download: 101_ObjectCategories.tar.gz")
    print("3. Extract to: datasets/101_ObjectCategories/")
    print("   Expected structure:")
    print("   datasets/101_ObjectCategories/101_ObjectCategories/")
    print("   ├── accordion/")
    print("   │   ├── image_0001.jpg")
    print("   │   └── ...")
    print("   ├── airplanes/")
    print("   └── ...")
    print()
    print("STEP 3: Update .env file")
    print("-" * 80)
    print("Update your .env file with the correct paths:")
    print("EVENT_PATH=path/to/datasets/N-Caltech101/Caltech101/Caltech101")
    print("IMAGE_PATH=path/to/datasets/101_ObjectCategories/101_ObjectCategories")
    print()
    print("=" * 80)


def download_caltech101_images(output_dir):
    """
    Attempt to download Caltech101 images automatically.
    Falls back to instructions if download fails.
    """
    print("\n[1/2] Downloading Caltech101 Images...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = output_dir / "101_ObjectCategories.tar.gz"
    extract_path = output_dir / "101_ObjectCategories"
    
    # Check if already extracted
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"  [OK] Caltech101 images already exist at {extract_path}")
        return str(extract_path)
    
    # Try to download
    try:
        print(f"  Downloading from {CALTECH101_IMAGES_URL}...")
        print("  This may take several minutes...")
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(
            CALTECH101_IMAGES_URL,
            tar_path,
            show_progress
        )
        print("\n  [OK] Download complete")
        
        # Extract
        print("  Extracting...")
        with zipfile.ZipFile(tar_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Clean up
        tar_path.unlink()
        print(f"  [OK] Extracted to {extract_path}")
        
        return str(extract_path)
        
    except Exception as e:
        print(f"\n  [ERROR] Automatic download failed: {e}")
        print("  Please download manually (see instructions above)")
        return None


def verify_dataset_structure(event_path, image_path):
    """
    Verify that datasets are correctly structured.
    
    Returns:
        (event_ok, image_ok): Tuple of booleans indicating if paths are valid
    """
    event_path = Path(event_path) if event_path else None
    image_path = Path(image_path) if image_path else None
    
    event_ok = False
    image_ok = False
    
    if event_path and event_path.exists():
        # Check for .bin files
        bin_files = list(event_path.rglob("*.bin"))
        if bin_files:
            event_ok = True
            print(f"  [OK] Found {len(bin_files)} event files")
        else:
            print(f"  [ERROR] No .bin files found in {event_path}")
    else:
        print(f"  [ERROR] Event path does not exist: {event_path}")
    
    if image_path and image_path.exists():
        # Check for image files
        image_files = list(image_path.rglob("*.jpg")) + list(image_path.rglob("*.png"))
        if image_files:
            image_ok = True
            print(f"  [OK] Found {len(image_files)} image files")
        else:
            print(f"  [ERROR] No image files found in {image_path}")
    else:
        print(f"  [ERROR] Image path does not exist: {image_path}")
    
    return event_ok, image_ok


def main():
    """Main function to download and verify datasets."""
    print("=" * 80)
    print("SpikeCLIP Dataset Setup")
    print("=" * 80)
    print()
    
    # Get paths from environment or use defaults
    event_path = os.getenv("EVENT_PATH")
    image_path = os.getenv("IMAGE_PATH")
    
    if not event_path or not image_path:
        print("[WARNING] Environment variables EVENT_PATH and IMAGE_PATH not set.")
        print("   Using default paths from .env.example")
        print()
        print_instructions()
        return
    
    print("[1/3] Checking dataset paths...")
    print(f"  EVENT_PATH: {event_path}")
    print(f"  IMAGE_PATH: {image_path}")
    print()
    
    # Verify existing datasets
    print("[2/3] Verifying dataset structure...")
    event_ok, image_ok = verify_dataset_structure(event_path, image_path)
    print()
    
    if event_ok and image_ok:
        print("=" * 80)
        print("[OK] Datasets are ready!")
        print("=" * 80)
        print(f"  Event data: {event_path}")
        print(f"  Image data: {image_path}")
        print()
        print("You can now run training scripts:")
        print("  python spikeclip_snn/train_spikeclip.py  # Stage 1")
        print("  python scripts/train_stage2.py           # Stage 2")
        print("  python scripts/train_stage3.py           # Stage 3")
        return
    
    # Try to download images if missing
    if not image_ok:
        print("[3/3] Attempting to download Caltech101 images...")
        downloaded_path = download_caltech101_images(DEFAULT_DATASET_DIR)
        if downloaded_path:
            print(f"\n[OK] Images downloaded to: {downloaded_path}")
            print("  Update IMAGE_PATH in your .env file if needed")
    
    # Event data requires manual download
    if not event_ok:
        print("\n[3/3] N-Caltech101 event data requires manual download.")
        print_instructions()
    
    print()
    print("=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print("1. Download N-Caltech101 event data (see instructions above)")
    print("2. Download Caltech101 image data (if not already done)")
    print("3. Update .env file with correct paths")
    print("4. Run this script again to verify")
    print("=" * 80)


if __name__ == "__main__":
    main()

