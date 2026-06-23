from src.config import get_settings
from src.extraction.prompts import SUMMARY_SYSTEM, SUMMARY_USER
from src.models.schemas import ExtractedFields, MatchResult, ValidationResult


class SummaryGenerator:
    def generate(
        self,
        extracted: ExtractedFields,
        match: MatchResult | None,
        validation: ValidationResult,
    ) -> str:
        settings = get_settings()
        reference = match.model_dump() if match else {"status": "no match"}
        validation_text = "; ".join(validation.mismatches) if validation.mismatches else "All checks passed"

        if not settings.use_azure_openai:
            return (
                f"Document Summary\n"
                f"Vendor: {extracted.vendor} | Amount: {extracted.amount} | PO/Ref: {match.reference_id if match else 'N/A'}\n"
                f"Match score: {match.similarity_score if match else 0} | Validation: {'PASS' if validation.passed else 'REVIEW'}\n"
                f"Notes: {validation_text}\n"
                f"Open items: {'None' if validation.passed else 'Manual review required'}"
            )

        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        response = client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {
                    "role": "user",
                    "content": SUMMARY_USER.format(
                        extracted=extracted.model_dump_json(),
                        reference=str(reference),
                        validation=validation_text,
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content or ""
