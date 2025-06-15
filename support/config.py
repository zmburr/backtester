from dotenv import load_dotenv
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from email.mime.base import MIMEBase
from email import encoders
# dotenv loading from project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

# Centralized API keys
SERP_API_KEY = os.getenv("SERP_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
PPLX_API_KEY     = os.getenv("PPLX_API_KEY")

def send_email(to_email, subject, body, attachments=None, is_html: bool = False,
               from_email='zmburr@gmail.com', password=os.getenv('GMAIL_PASSWORD')):

    # Email server settings for Gmail
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Create message
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Add body to email
    subtype = 'html' if is_html else 'plain'
    msg.attach(MIMEText(body, _subtype=subtype))

    # Attach any files
    attachments = attachments or []
    for path in attachments:
        try:
            with open(path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}"')
            msg.attach(part)
        except Exception as e:
            print(f"Failed to attach {path}: {e}")

    try:
        # Connect to the server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable TLS

        # Login to your email account
        server.login(from_email, password)

        # Send email
        server.send_message(msg)
        print(f"Email successfully sent to {to_email}")

    except Exception as e:
        print(f"Failed to send email: {str(e)}")

    finally:
        # Close the connection
        server.quit()