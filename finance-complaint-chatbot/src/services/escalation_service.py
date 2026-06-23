import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.config import get_settings

logger = logging.getLogger(__name__)


def send_escalation_email(subject: str, body: str) -> bool:
    settings = get_settings()
    if not settings.aws_access_key_id:
        logger.warning("AWS credentials not configured — escalation email logged only.")
        logger.info("ESCALATION EMAIL\nSubject: %s\n%s", subject, body)
        return False

    try:
        client = boto3.client(
            "ses",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        client.send_email(
            Source=settings.ses_sender,
            Destination={"ToAddresses": settings.escalation_emails},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
            },
        )
        return True
    except (BotoCoreError, ClientError) as exc:
        logger.error("SES send failed: %s", exc)
        return False
