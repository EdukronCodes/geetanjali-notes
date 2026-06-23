import json
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config import get_settings

Base = declarative_base()


class VendorMaster(Base):
    __tablename__ = "vendor_master"
    vendor_id = Column(Integer, primary_key=True, autoincrement=True)
    vendor_name = Column(String(150), nullable=False, unique=True)
    cost_center = Column(String(30))


class PORegister(Base):
    __tablename__ = "po_register"
    po_number = Column(String(30), primary_key=True)
    vendor_name = Column(String(150), nullable=False)
    amount = Column(Float, nullable=False)
    cost_center = Column(String(30))
    due_date = Column(String(20))


class DocumentAuditLog(Base):
    __tablename__ = "document_audit_log"
    audit_id = Column(Integer, primary_key=True, autoincrement=True)
    document_path = Column(String(500))
    extracted_json = Column(Text)
    match_reference_id = Column(String(50))
    match_score = Column(Float)
    validation_passed = Column(Integer, default=0)
    summary = Column(Text)
    status = Column(String(30), default="PROCESSED")
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


def seed_from_csv() -> None:
    import pandas as pd

    settings = get_settings()
    init_db()
    csv_path = Path(settings.reference_csv_path)
    if not csv_path.is_absolute():
        csv_path = settings.project_root / csv_path
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    with get_session() as session:
        if session.query(VendorMaster).count() > 0:
            return
        for _, row in df.iterrows():
            vendor = session.query(VendorMaster).filter_by(vendor_name=row["vendor_name"]).first()
            if not vendor:
                session.add(VendorMaster(vendor_name=row["vendor_name"], cost_center=row["cost_center"]))
            if row["document_type"] == "PO":
                existing = session.query(PORegister).filter_by(po_number=row["reference_id"]).first()
                if not existing:
                    session.add(
                        PORegister(
                            po_number=row["reference_id"],
                            vendor_name=row["vendor_name"],
                            amount=float(row["amount"]),
                            cost_center=row["cost_center"],
                            due_date=str(row["due_date"]),
                        )
                    )


def read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    return path.read_text(encoding="utf-8", errors="ignore")


def rule_based_extract(text: str) -> dict:
    """Fallback extraction when Azure OpenAI is unavailable."""
    patterns = {
        "vendor": r"(?:Vendor|Supplier)[:\s-]+([A-Za-z0-9 &\.]+)",
        "invoice_number": r"(?:Invoice Number|Invoice No)[:\s-]+([A-Z0-9-]+)",
        "po_number": r"(?:Purchase Order|PO)[:\s-]+([A-Z0-9-]+)",
        "contract_id": r"(?:Contract ID|Contract)[:\s-]+([A-Z0-9-]+)",
        "amount": r"(?:Amount|Annual Value|Total)[:\s-]+(?:USD\s?)?([\d,]+\.?\d*)",
        "due_date": r"(?:Due Date|Expiry Date)[:\s-]+(\d{4}-\d{2}-\d{2})",
        "cost_center": r"(?:Cost Center)[:\s-]+([A-Z0-9-]+)",
    }
    result = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            val = match.group(1).strip()
            if field == "amount":
                result[field] = float(val.replace(",", ""))
            else:
                result[field] = val
    doc_type = "INVOICE" if "invoice" in text.lower() else "CONTRACT" if "contract" in text.lower() else "OTHER"
    result["document_type"] = doc_type
    result["confidence"] = 0.82 if len(result) >= 4 else 0.55
    return result
