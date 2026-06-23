import json
import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.config import get_settings

logger = logging.getLogger(__name__)


def upload_to_s3(local_path: str, s3_key: str) -> bool:
    settings = get_settings()
    if not settings.aws_access_key_id:
        logger.info("S3 mock upload: %s -> s3://%s/%s", local_path, settings.s3_bucket, s3_key)
        return False
    try:
        client = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        client.upload_file(local_path, settings.s3_bucket, s3_key)
        return True
    except (BotoCoreError, ClientError) as exc:
        logger.error("S3 upload failed: %s", exc)
        return False


def send_summary_email(subject: str, body: str) -> bool:
    settings = get_settings()
    if not settings.aws_access_key_id:
        logger.info("SES mock email\nSubject: %s\n%s", subject, body)
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
            Destination={"ToAddresses": settings.notification_emails},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
            },
        )
        return True
    except (BotoCoreError, ClientError) as exc:
        logger.error("SES send failed: %s", exc)
        return False
