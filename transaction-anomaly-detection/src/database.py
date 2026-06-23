import pandas as pd
from sqlalchemy import Column, Date, Float, Integer, String, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import get_settings

Base = declarative_base()


class LedgerTransaction(Base):
    __tablename__ = "ledger_transactions"
    transaction_id = Column(String(30), primary_key=True)
    account_id = Column(String(20), nullable=False)
    vendor_id = Column(String(20), nullable=False)
    cost_center = Column(String(30), nullable=False)
    amount = Column(Float, nullable=False)
    posting_date = Column(Date, nullable=False)
    is_weekend = Column(Integer, default=0)
    analyst_label = Column(Integer, nullable=True)


class AnomalyRun(Base):
    __tablename__ = "anomaly_runs"
    run_id = Column(Integer, primary_key=True, autoincrement=True)
    run_month = Column(String(7), nullable=False)
    total_transactions = Column(Integer)
    flagged_count = Column(Integer)
    precision_score = Column(Float)
    output_file = Column(String(500))
    created_at = Column(String(30), server_default=func.now())


def get_engine():
    settings = get_settings()
    if settings.db_engine == "mssql" and settings.mssql_connection_string:
        return create_engine(f"mssql+pyodbc:///?odbc_connect={settings.mssql_connection_string}")
    return create_engine(f"sqlite:///{settings.sqlite_path}")


engine = get_engine()
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def load_transactions_from_db() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM ledger_transactions", engine)


def load_transactions_from_csv() -> pd.DataFrame:
    settings = get_settings()
    path = settings.project_root / settings.sample_csv_path
    return pd.read_csv(path, parse_dates=["posting_date"])


def seed_from_csv() -> None:
    init_db()
    df = load_transactions_from_csv()
    df.to_sql("ledger_transactions", engine, if_exists="replace", index=False)
