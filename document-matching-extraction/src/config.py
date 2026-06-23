from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-02-15-preview"

    db_engine: str = "sqlite"
    sqlite_path: str = "./data/documents.db"
    mssql_connection_string: str = ""

    aws_region: str = "ap-south-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket: str = "finance-documents"
    s3_incoming_prefix: str = "incoming/"
    ses_sender: str = "noreply@financeops.example.com"
    ses_recipients: str = "finance-ops@example.com"

    faiss_index_path: str = "./data/faiss_index"
    incoming_docs_path: str = "./data/incoming_documents"
    reference_csv_path: str = "./data/reference_records/reference_master.csv"
    confidence_threshold: float = 0.75
    log_level: str = "INFO"

    @property
    def use_azure_openai(self) -> bool:
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    @property
    def notification_emails(self) -> list[str]:
        return [e.strip() for e in self.ses_recipients.split(",") if e.strip()]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent


@lru_cache
def get_settings() -> Settings:
    return Settings()
