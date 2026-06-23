from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    db_engine: str = "sqlite"
    sqlite_path: str = "./data/finance_ledger.db"
    mssql_connection_string: str = ""

    model_path: str = "./models"
    output_path: str = "./output"
    sample_csv_path: str = "./data/sample_transactions.csv"
    random_state: int = 42
    isolation_forest_contamination: float = 0.05
    anomaly_score_threshold: float = 0.65
    validation_precision_target: float = 0.88
    run_month: str = "auto"
    log_level: str = "INFO"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent


@lru_cache
def get_settings() -> Settings:
    return Settings()
