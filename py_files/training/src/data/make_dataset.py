import csv
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.config import Config
from botocore import UNSIGNED

from src.config import S3_ENDPOINT, S3_BUCKET, IMAGE_DATA_DIR, METADATA_PATH

def _setup_s3_client():
    """Create a boto3 S3 client configured for Hetzner's public bucket."""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id='',
        aws_secret_access_key='',
        config=Config(signature_version=UNSIGNED)
    )

def _download_file_with_boto3(s3_client, key, output_path):
    """Download a single file from S3 using boto3."""
    output_path = Path(output_path)
    if output_path.exists():
        return False
    try:
        s3_client.download_file(S3_BUCKET, key, str(output_path))
        return True
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        return False

def download_files_from_metadata(metadata_path, output_dir, prefix=""):
    """Download all files listed in a metadata csv from the object store."""
    s3_client = _setup_s3_client()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    downloaded_count, skipped_count, failed_count = 0, 0, 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for row in rows:
            filename = row["filename"]
            key = f"{prefix}{filename}"
            output_path = Path(output_dir) / filename
            future = executor.submit(_download_file_with_boto3, s3_client, key, output_path)
            futures[future] = (key, output_path)

        for future in as_completed(futures):
            key, output_path = futures[future]
            if future.result():
                downloaded_count += 1
            else:
                if output_path.exists():
                    skipped_count += 1
                else:
                    failed_count += 1

    print(f"Downloaded {downloaded_count} images, skipped {skipped_count} existing, failed {failed_count}")

def pull_data_from_dvc():
    """Pull the training and test images listed in the metadata from the object store."""
    download_files_from_metadata("data/images/metadata.csv", "data/images/raw", "images/raw/")
    download_files_from_metadata("data/test/metadata.csv", "data/test", "test/")

def load_image_dataset():
    """Load (image_path, label) pairs from the raw images and the metadata."""
    if not Path(METADATA_PATH).exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)
    image_dataset = []
    
    for _, row in metadata.iterrows():
        filename = row["filename"]
        label = int(row["class"])
        image_path = Path(IMAGE_DATA_DIR) / filename
        if image_path.exists():
            image_dataset.append((image_path, label))

    return image_dataset