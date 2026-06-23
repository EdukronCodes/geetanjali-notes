EXTRACTION_SYSTEM = """You extract structured finance fields from documents.
Return valid JSON only with keys: vendor, invoice_number, po_number, contract_id,
amount (number), due_date (YYYY-MM-DD), cost_center, document_type, confidence (0-1)."""


EXTRACTION_USER = """Extract fields from this finance document:

{document_text}
"""


SUMMARY_SYSTEM = """You summarize matched finance document pairs for operations review.
Highlight key terms, value differences, and open items. Be concise and factual."""


SUMMARY_USER = """Incoming document extraction:
{extracted}

Matched reference record:
{reference}

Validation notes:
{validation}

Generate a structured summary for finance ops email."""
