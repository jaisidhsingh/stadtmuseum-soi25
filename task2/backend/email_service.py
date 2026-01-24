
import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from typing import List

logger = logging.getLogger("EmailService")

class EmailService:
    def __init__(self):
        self.smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.smtp_user = os.environ.get("SMTP_USER")
        self.smtp_password = os.environ.get("SMTP_PASSWORD")
        
    def send_email(self, to_email: str, image_paths: List[Path]):
        """
        Sends an email with the specified images as attachments.
        """
        if not self.smtp_user or not self.smtp_password:
            logger.warning("SMTP credentials not set. Skipping email send (Mock mode).")
            return "Mock: Email sent (credentials missing)"

        msg = MIMEMultipart()
        msg['Subject'] = 'Your Generated Silhouettes'
        msg['From'] = self.smtp_user
        msg['To'] = to_email

        body = MIMEText("Here are your generated silhouettes and compositions!", 'plain')
        msg.attach(body)

        for path in image_paths:
            if not path.exists():
                logger.warning(f"Attachment not found: {path}")
                continue
                
            try:
                with open(path, 'rb') as f:
                    img_data = f.read()
                    
                image = MIMEImage(img_data, name=path.name)
                msg.attach(image)
            except Exception as e:
                logger.error(f"Failed to attach image {path}: {e}")

        try:
            logger.info(f"Connecting to SMTP server {self.smtp_server}:{self.smtp_port}...")
            # Use SMTP_SSL if port 465, else starttls
            if self.smtp_port == 465:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
            
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
            server.quit()
            logger.info(f"Email sent successfully to {to_email}")
            return "Email sent successfully"
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise e
