"""
src/data/ingest.py

Load raw data from local filesystem or AWS S3.
Supports both local dev and cloud deployment.
"""

import os
import io
import logging
import pandas as pd
import boto3
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def load_local(filepath: str) -> pd.DataFrame:
    """Load CSV data from local filesystem."""
    logger.info(f"Loading data from local path: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def load_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """
    Load CSV data from an S3 bucket.

    Args:
        bucket: S3 bucket name (e.g. 'my-churn-analytics-bucket')
        key: S3 object key (e.g. 'data/raw/telco_churn.csv')

    Returns:
        pandas DataFrame
    """
    logger.info(f"Loading data from s3://{bucket}/{key}")
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    response = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(response["Body"].read()))
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns from S3")
    return df


def upload_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """
    Upload a DataFrame as CSV to S3.

    Args:
        df: DataFrame to upload
        bucket: S3 bucket name
        key: S3 object key
    """
    logger.info(f"Uploading {len(df):,} rows to s3://{bucket}/{key}")
    s3 = boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    logger.info("Upload complete.")


def load_data(use_s3: bool = False) -> pd.DataFrame:
    """
    Main entry point for data loading.
    Falls back to local if S3 env vars are not set.

    Args:
        use_s3: If True, attempt to load from S3

    Returns:
        Raw DataFrame
    """
    if use_s3 and os.getenv("S3_BUCKET_NAME"):
        bucket = os.getenv("S3_BUCKET_NAME")
        key = f"data/raw/{os.getenv('RAW_FILENAME', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')}"
        return load_from_s3(bucket, key)
    else:
        local_path = os.getenv("LOCAL_DATA_PATH", "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        return load_local(local_path)
