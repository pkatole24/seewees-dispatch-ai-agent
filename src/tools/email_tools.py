from __future__ import annotations
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email_smtp(
    subject: str,
    html_body: str,
    to_email: str,
    attachments: list[dict[str, str]] | None = None,
) -> None:
    """
    Generic SMTP sender. Works with Zoho if env vars are set:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
    """
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "465"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_email

    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(html_body, "html"))
    msg.attach(alternative)

    for attachment in attachments or []:
        mime_type = attachment.get("mime_type", "text/plain")
        _, _, subtype = mime_type.partition("/")
        part = MIMEText(attachment.get("content", ""), subtype or "plain", "utf-8")
        part.add_header(
            "Content-Disposition",
            "attachment",
            filename=attachment.get("filename", "attachment.txt"),
        )
        msg.attach(part)

    if port == 465:
        with smtplib.SMTP_SSL(host, port) as server:
            server.login(user, password)
            server.sendmail(user, [to_email], msg.as_string())
    else:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_email], msg.as_string())
