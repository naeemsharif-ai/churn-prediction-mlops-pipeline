from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[2]
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "churn-prediction"
    random_seed: int = 42
    target_col: str = "Churn"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()