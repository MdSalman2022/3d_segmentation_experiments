"""
KiTS Imaging Downloader
=======================
Downloads CT imaging files for the KiTS dataset.
Run this if your dataset only has segmentation masks.
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Configuration
KITS_BASE_URL = "https://kits19.sfo2.digitaloceanspaces.com"
DATASET_DIR = Path("./kits23/dataset")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, str(output_path), reporthook=t.update_to)


def get_case_ids(dataset_dir: Path):
    """Get all case IDs from the dataset directory."""
    case_ids = []
    for item in sorted(dataset_dir.iterdir()):
        if item.is_dir() and item.name.startswith("case_"):
            case_ids.append(item.name)
    return case_ids


def download_imaging(dataset_dir: Path = DATASET_DIR, overwrite: bool = False):
    """Download imaging files for all cases."""
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    case_ids = get_case_ids(dataset_dir)
    print(f"Found {len(case_ids)} cases in {dataset_dir}")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for case_id in case_ids:
        case_dir = dataset_dir / case_id
        imaging_path = case_dir / "imaging.nii.gz"
        
        if imaging_path.exists() and not overwrite:
            skipped += 1
            continue
        
        # KiTS uses sequential numbering
        case_num = int(case_id.split("_")[1])
        url = f"{KITS_BASE_URL}/{case_id}/imaging.nii.gz"
        
        print(f"\nDownloading {case_id}...")
        try:
            download_file(url, imaging_path)
            downloaded += 1
        except Exception as e:
            print(f"  Failed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download KiTS imaging files")
    parser.add_argument("--dataset_dir", type=str, default="./kits23/dataset",
                        help="Path to dataset directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing imaging files")
    args = parser.parse_args()
    
    download_imaging(Path(args.dataset_dir), args.overwrite)
