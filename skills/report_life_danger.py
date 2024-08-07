import subprocess
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_notification(msg_content):
    """Send a notification to macOS Notification Center."""
    script = f'display notification "{msg_content}" with title "AGENTIC SLEEP MONITOR"'
    subprocess.run(["osascript", "-e", script])

def send_email(msg_content):
    sender_email = "matthews65537@outlook.com"
    sender_password = os.environ.get('GMAIL_PASSWORD')
    receiver_email = "matthews65537@gmail.com"
    smtp_server = "smtp.office365.com"  # This is for Gmail, change if using a different provider
    smtp_port = 587  # For TLS

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Agentic Sleep Monitor WARNING"

    # Add body to email
    message.attach(MIMEText(msg_content, "plain"))

    try:
        # Create a secure SSL/TLS connection
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        
        # Login to your account
        server.login(sender_email, sender_password)
        
        # Send the email
        server.send_message(message)
        # print("Email sent successfully!")

    except Exception as e:
        pass
        # print(f"An error occurred: {e}")

    finally:
        # Close the connection
        server.quit()

def report_danger(msg):
    send_email(msg)
    send_notification(msg)

if __name__ == "__main__":
    report_danger("DANGER DANGER")