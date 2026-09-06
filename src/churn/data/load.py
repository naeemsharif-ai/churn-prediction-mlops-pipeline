"""Data acquisition: download raw Telco churn CSV to data/raw/."""
import logging
from pathlib import Path

import requests

from churn.config import settings

logger = logging.getLogger(__name__)

TELCO_CHURN_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

def download_raw_data(url: str = TELCO_CHURN_URL, dest: Path | None = None) -> Path:
    """Download raw CSV if not already present. Returns local path."""
    dest = dest or settings.data_raw_dir / "telco_churn_raw.csv"
    if dest.exists():
        logger.info("Raw data already present at %s, skipping download.", dest)
        return dest

    logger.info("Downloading raw data from %s", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    logger.info("Saved %d bytes to %s", len(response.content), dest)
    return dest

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_raw_data()