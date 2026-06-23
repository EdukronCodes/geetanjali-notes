from pydantic import BaseModel, Field


class ExtractedFields(BaseModel):
    vendor: str | None = None
    invoice_number: str | None = None
    po_number: str | None = None
    contract_id: str | None = None
    amount: float | None = None
    due_date: str | None = None
    cost_center: str | None = None
    document_type: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MatchResult(BaseModel):
    reference_id: str
    vendor_name: str
    similarity_score: float
    matched: bool


class ValidationResult(BaseModel):
    passed: bool
    mismatches: list[str]
    requires_review: bool


class PipelineResult(BaseModel):
    document_path: str
    extracted: ExtractedFields
    match: MatchResult | None
    validation: ValidationResult
    summary: str
    audit_id: int | None = None
