from contextlib import contextmanager
from datetime import date
from typing import Generator

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config import get_settings

Base = declarative_base()


class Account(Base):
    __tablename__ = "accounts"
    account_id = Column(String(20), primary_key=True)
    customer_name = Column(String(100), nullable=False)
    status = Column(String(20), default="ACTIVE")
    created_at = Column(DateTime, server_default=func.now())


class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id = Column(String(30), primary_key=True)
    account_id = Column(String(20), nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    posting_date = Column(Date, nullable=False)
    description = Column(String(255))
    status = Column(String(20), default="POSTED")


class Complaint(Base):
    __tablename__ = "complaints"
    complaint_id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(20))
    transaction_id = Column(String(30))
    intent = Column(String(50))
    message = Column(Text)
    bot_response = Column(Text)
    escalated = Column(Integer, default=0)
    case_summary = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


def _build_engine():
    settings = get_settings()
    if settings.db_engine == "mssql" and settings.mssql_connection_string:
        return create_engine(f"mssql+pyodbc:///?odbc_connect={settings.mssql_connection_string}")
    return create_engine(f"sqlite:///{settings.sqlite_path}")


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def seed_sample_data() -> None:
    init_db()
    with get_session() as session:
        if session.query(Account).count() > 0:
            return
        accounts = [
            Account(account_id="ACC-10001", customer_name="Acme Corp", status="ACTIVE"),
            Account(account_id="ACC-10002", customer_name="Globex Ltd", status="ACTIVE"),
        ]
        transactions = [
            Transaction(
                transaction_id="TXN-900001",
                account_id="ACC-10001",
                amount=12500.00,
                posting_date=date(2025, 11, 15),
                description="Vendor payment - Invoice INV-4421",
            ),
            Transaction(
                transaction_id="TXN-900002",
                account_id="ACC-10001",
                amount=-450.00,
                posting_date=date(2025, 12, 1),
                description="Billing adjustment",
            ),
            Transaction(
                transaction_id="TXN-900003",
                account_id="ACC-10002",
                amount=78000.00,
                posting_date=date(2025, 12, 10),
                description="Quarterly settlement",
            ),
        ]
        session.add_all(accounts + transactions)
