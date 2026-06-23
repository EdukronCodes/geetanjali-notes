from sqlalchemy.orm import Session

from src.config import get_settings
from src.database import PORegister, VendorMaster
from src.models.schemas import ExtractedFields, MatchResult, ValidationResult


def cross_validate(session: Session, extracted: ExtractedFields, match: MatchResult | None) -> ValidationResult:
    settings = get_settings()
    mismatches: list[str] = []

    if not match or not match.matched:
        mismatches.append("No confident vector match found")
        return ValidationResult(passed=False, mismatches=mismatches, requires_review=True)

    ref_id = match.reference_id
    po = session.query(PORegister).filter(PORegister.po_number == ref_id).first()
    vendor = session.query(VendorMaster).filter(VendorMaster.vendor_name == match.vendor_name).first()

    if extracted.vendor and vendor and extracted.vendor.lower() not in vendor.vendor_name.lower():
        mismatches.append(f"Vendor mismatch: extracted '{extracted.vendor}' vs master '{vendor.vendor_name}'")

    if po and extracted.amount is not None and abs(po.amount - extracted.amount) > 0.01:
        mismatches.append(f"Amount mismatch: extracted {extracted.amount} vs PO {po.amount}")

    if po and extracted.cost_center and po.cost_center and extracted.cost_center != po.cost_center:
        mismatches.append(f"Cost center mismatch: {extracted.cost_center} vs {po.cost_center}")

    low_confidence = extracted.confidence < settings.confidence_threshold
    if low_confidence:
        mismatches.append(f"Low extraction confidence: {extracted.confidence}")

    passed = len(mismatches) == 0
    return ValidationResult(passed=passed, mismatches=mismatches, requires_review=not passed)
