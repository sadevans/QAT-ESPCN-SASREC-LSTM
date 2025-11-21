import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, dest_path: Path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte

    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def download_movielens_1m(data_dir: str = "/ssd/a.gorokhova/datasets/itmo/ml-1m"):
    """Download and extract MovieLens 1M dataset."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = data_path / "ml-1m.zip"
    
    # Check if ratings.dat already exists in the target directory or subdir
    target_file = data_path / "ratings.dat"
    # The zip usually extracts to a folder 'ml-1m'
    extracted_folder_file = data_path / "ml-1m" / "ratings.dat"
    
    if target_file.exists() or extracted_folder_file.exists():
        print(f"Dataset already exists in {data_path}")
        return

    print(f"Downloading MovieLens 1M to {data_path}...")
    download_file(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path.parent) # Extract to parent so we get data/ml-1m/ml-1m structure or similar
        
    # Cleanup
    zip_path.unlink()
    
    # Move files if nested
    # The zip contains a folder 'ml-1m'. If we extracted to data/ml-1m, we might have data/ml-1m/ml-1m.
    # Let's normalize.
    # If the user asked for data/ml-1m, and we extracted 'ml-1m' folder there...
    # Let's just rely on the standard structure.
    
    print("Done!")

if __name__ == "__main__":
    download_movielens_1m()

