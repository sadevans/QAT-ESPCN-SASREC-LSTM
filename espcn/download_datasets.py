"""Script to download and extract datasets for ESPCN training and evaluation."""

import os
import sys
import zipfile
import tarfile
import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm


def download_and_extract_figshare(data_root: Path) -> None:
    """
    Download and extract Set5 and Set14 benchmark datasets from figshare.
    
    Args:
        data_root: Root directory for storing datasets.
    """
    archive_path = data_root / "sr_benchmarks.zip"
    set5_target = data_root / "Set5"
    set14_target = data_root / "Set14"

    if not archive_path.exists():
        print("📥 Downloading SR benchmarks from figshare...")
        url = "https://figshare.com/ndownloader/articles/21586188/versions/1"
        download_url(url, archive_path)

    temp_dir = data_root / "_temp_sr_benchmarks"

    if not set5_target.exists() or not set14_target.exists():
        print("📦 Extracting inner ZIP archives...")
        extract_archive(archive_path, temp_dir)
        if (temp_dir / "Set5.zip").exists() and not set5_target.exists():
            extract_archive(temp_dir / "Set5.zip", data_root)
        else:
            print("⚠️  Set5.zip not found or already extracted.")
        if (temp_dir / "Set14.zip").exists() and not set14_target.exists():
            extract_archive(temp_dir / "Set14.zip", data_root)
        else:
            print("⚠️  Set14.zip not found or already extracted.")
        import shutil
        shutil.rmtree(temp_dir)

    archive_path.unlink(missing_ok=True)

class DownloadProgressBar(tqdm):
    """
    Progress bar for file downloads with byte tracking.
    
    Extends tqdm to show download progress in bytes.
    """
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update progress bar with downloaded bytes.
        
        Args:
            b: Number of blocks downloaded (default: 1).
            bsize: Size of each block in bytes (default: 1).
            tsize: Total size of file in bytes (optional).
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path) -> None:
    """
    Download a file from URL to output path with progress bar.
    
    Args:
        url: URL of the file to download.
        output_path: Path where the downloaded file should be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """
    Extract archive file (ZIP or TAR.GZ) to a directory.
    
    Args:
        archive_path: Path to the archive file.
        extract_to: Directory where files should be extracted.
        
    Raises:
        ValueError: If archive format is not supported (ZIP or TAR.GZ).
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data", help="Root directory for datasets")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(exist_ok=True)

    div2k_dir = data_root / "DIV2K_train_HR"
    if not div2k_dir.exists() or not any(div2k_dir.iterdir()):
        print("Downloading DIV2K_train_HR...")
        url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
        zip_path = data_root / "DIV2K_train_HR.zip"
        if not zip_path.exists():
            download_url(url, zip_path)
        print("Extracting DIV2K...")
        extract_archive(zip_path, data_root)
        zip_path.unlink()  # remove zip after extraction
    else:
        print("DIV2K_train_HR already exists.")

    print("Downloading Set5&Set14...")
    download_and_extract_figshare(data_root)
    print("\nAll datasets downloaded and ready!")
    print(f"Location: {data_root.resolve()}")


if __name__ == "__main__":
    main()
