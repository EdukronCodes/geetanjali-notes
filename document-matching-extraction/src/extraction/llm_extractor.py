import json
import re

from src.config import get_settings
from src.database import rule_based_extract
from src.extraction.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER
from src.models.schemas import ExtractedFields


class LLMExtractor:
    def extract(self, document_text: str) -> ExtractedFields:
        settings = get_settings()
        if not settings.use_azure_openai:
            data = rule_based_extract(document_text)
            return ExtractedFields(**data)

        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        response = client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": EXTRACTION_USER.format(document_text=document_text[:8000])},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        content = response.choices[0].message.content or "{}"
        match = re.search(r"\{.*\}", content, re.S)
        data = json.loads(match.group(0) if match else content)
        return ExtractedFields(**data)
