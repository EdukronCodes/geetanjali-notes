import json
import logging
from pathlib import Path

from src.config import get_settings
from src.database import DocumentAuditLog, get_session, init_db, read_document_text, seed_from_csv
from src.extraction.llm_extractor import LLMExtractor
from src.matching.vector_matcher import VectorMatcher
from src.models.schemas import PipelineResult
from src.notifications.ses_notifier import send_summary_email, upload_to_s3
from src.summary.summary_generator import SummaryGenerator
from src.validation.cross_validator import cross_validate

logger = logging.getLogger(__name__)


class DocumentPipeline:
    def __init__(self) -> None:
        self.extractor = LLMExtractor()
        self.matcher = VectorMatcher()
        self.summarizer = SummaryGenerator()

    def process_file(self, file_path: Path) -> PipelineResult:
        settings = get_settings()
        text = read_document_text(file_path)
        extracted = self.extractor.extract(text)
        match = self.matcher.match(extracted, text)

        with get_session() as session:
            validation = cross_validate(session, extracted, match)
            summary = self.summarizer.generate(extracted, match, validation)

            audit = DocumentAuditLog(
                document_path=str(file_path),
                extracted_json=extracted.model_dump_json(),
                match_reference_id=match.reference_id if match else None,
                match_score=match.similarity_score if match else None,
                validation_passed=1 if validation.passed else 0,
                summary=summary,
                status="REVIEW" if validation.requires_review else "MATCHED",
            )
            session.add(audit)
            session.flush()
            audit_id = audit.audit_id

        s3_key = f"{settings.s3_incoming_prefix}{file_path.name}"
        upload_to_s3(str(file_path), s3_key)
        send_summary_email(
            subject=f"[Doc Intelligence] {file_path.name} — {'PASS' if validation.passed else 'REVIEW'}",
            body=summary,
        )

        return PipelineResult(
            document_path=str(file_path),
            extracted=extracted,
            match=match,
            validation=validation,
            summary=summary,
            audit_id=audit_id,
        )

    def run_batch(self, input_dir: Path | None = None) -> list[PipelineResult]:
        settings = get_settings()
        init_db()
        seed_from_csv()
        directory = input_dir or (settings.project_root / settings.incoming_docs_path)
        results = []
        for file_path in sorted(directory.glob("*")):
            if file_path.suffix.lower() in {".txt", ".pdf", ".docx"}:
                logger.info("Processing %s", file_path.name)
                results.append(self.process_file(file_path))
        return results


def main() -> None:
    logging.basicConfig(level=get_settings().log_level)
    pipeline = DocumentPipeline()
    results = pipeline.run_batch()
    print(json.dumps([r.model_dump() for r in results], indent=2, default=str))


if __name__ == "__main__":
    main()
