#!/usr/bin/env python3
"""
SENTINEL Phase 5: Live Gmail Integration Test
==============================================

Real-time phishing detection on live Gmail messages using:
- Stream A: DistilBERT text analysis
- Stream B: XGBoost URL feature analysis  
- Stream C: Attachment structural analysis
- Ensemble: Weighted fusion with critical threat override
- XAI: SHAP explainability with threat descriptors

Author: Sentinel ML Team
Date: April 2026
Status: Production Testing
"""

import os
import sys
import json
import logging
import time
import re
import base64
import mimetypes
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from email.mime.text import MIMEText

# Gmail & Authentication
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Sentinel & ML
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_engine import SentinelEnsemble

# Text processing
import re
from bs4 import BeautifulSoup
import requests

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Sentinel_Gmail_Integration - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_logs/8_test_gmail_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify']
TEMP_ATTACHMENT_DIR = Path('Data/temp_gmail')
TOKEN_FILE = 'token.json'
CREDENTIALS_FILE = 'credentials.json'  # Downloaded from Google Cloud Console
POLL_INTERVAL = 10  # seconds
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024  # 10 MB


class GmailIntegration:
    """
    Gmail API interface with OAuth2 authentication and message processing.
    """

    def __init__(self):
        """Initialize Gmail service and authenticate."""
        self.service = None
        self.authenticate()
        self._create_temp_dir()

    def _create_temp_dir(self):
        """Create temporary attachment directory."""
        TEMP_ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"[INIT] Temporary attachment directory: {TEMP_ATTACHMENT_DIR}")

    def authenticate(self) -> None:
        """
        Authenticate with Gmail API using OAuth2.
        Saves credentials to token.json for reuse.
        """
        logger.info("[AUTH] Starting Gmail OAuth2 authentication...")

        creds = None

        # Load existing token if available
        if os.path.exists(TOKEN_FILE):
            logger.info(f"[AUTH] Loading existing credentials from {TOKEN_FILE}")
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, GMAIL_SCOPES)

        # If no valid credentials, perform OAuth2 flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("[AUTH] Refreshing expired credentials...")
                creds.refresh(Request())
            else:
                if not os.path.exists(CREDENTIALS_FILE):
                    logger.error(
                        f"\n[ERROR] Missing {CREDENTIALS_FILE}\n"
                        "Please download OAuth2 credentials from Google Cloud Console:\n"
                        "1. Go to https://console.cloud.google.com\n"
                        "2. Create a Desktop application\n"
                        "3. Download as JSON and save as 'credentials.json'\n"
                        "4. Grant Gmail API access (readonly + modify)\n"
                    )
                    sys.exit(1)

                logger.info(f"[AUTH] Performing OAuth2 flow using {CREDENTIALS_FILE}...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE,
                    GMAIL_SCOPES,
                    scopes=GMAIL_SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save credentials for future runs
            with open(TOKEN_FILE, 'w') as token_file:
                token_file.write(creds.to_json())
                logger.info(f"[AUTH] Credentials saved to {TOKEN_FILE}")

        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("[AUTH] ✅ Gmail API authenticated successfully!")

    def get_unread_messages(self, max_results: int = 1) -> List[Dict]:
        """
        Fetch unread messages from Gmail.

        Args:
            max_results: Maximum number of messages to retrieve

        Returns:
            List of message dictionaries
        """
        try:
            results = self.service.users().messages().list(
                userId='me',
                q='is:unread',
                maxResults=max_results
            ).execute()

            messages = results.get('messages', [])
            if messages:
                logger.info(f"[FETCH] Found {len(messages)} unread message(s)")
            return messages
        except HttpError as error:
            logger.error(f"[FETCH] Gmail API error: {error}")
            return []

    def get_message_details(self, message_id: str) -> Optional[Dict]:
        """
        Get full message details including headers and body.

        Args:
            message_id: Gmail message ID

        Returns:
            Message object with headers and payload
        """
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()
            return message
        except HttpError as error:
            logger.error(f"[GET_MESSAGE] Error retrieving message: {error}")
            return None

    def mark_as_read(self, message_id: str) -> bool:
        """
        Mark a message as read to avoid reprocessing.

        Args:
            message_id: Gmail message ID

        Returns:
            True if successful, False otherwise
        """
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            logger.info(f"[MARK_READ] Message {message_id} marked as read")
            return True
        except HttpError as error:
            logger.error(f"[MARK_READ] Error marking message as read: {error}")
            return False

    def extract_email_parts(self, message: Dict) -> Tuple[str, str, str, str, List[Tuple[str, bytes]]]:
        """
        Extract sender, subject, body, and attachments from message.

        Args:
            message: Full message object from Gmail API

        Returns:
            Tuple of (sender, subject, body, html_body, attachments)
            attachments: List of (filename, content) tuples
        """
        headers = message['payload'].get('headers', [])
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '[No Subject]')

        body = self._extract_body(message['payload'])
        html_body = self._extract_html_body(message['payload'])
        attachments = self._extract_attachments(message)

        logger.info(f"[EXTRACT] From: {sender}, Subject: {subject}")
        logger.info(f"[EXTRACT] Body length: {len(body)}, Attachments: {len(attachments)}")

        return sender, subject, body, html_body, attachments

    def _extract_body(self, payload: Dict) -> str:
        """Extract plain text body from message payload."""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data', '')
                    if data:
                        return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        else:
            data = payload['body'].get('data', '')
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        return ''

    def _extract_html_body(self, payload: Dict) -> str:
        """Extract HTML body from message payload."""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/html':
                    data = part['body'].get('data', '')
                    if data:
                        return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        else:
            if payload['mimeType'] == 'text/html':
                data = payload['body'].get('data', '')
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
        return ''

    def _extract_attachments(self, message: Dict) -> List[Tuple[str, bytes]]:
        """Extract attachments from message."""
        attachments = []
        parts = message['payload'].get('parts', [])

        for part in parts:
            if part.get('filename'):
                filename = part['filename']
                attachment_id = part['body'].get('attachmentId')

                if attachment_id:
                    try:
                        attachment = self.service.users().messages().attachments().get(
                            userId='me',
                            messageId=message['id'],
                            id=attachment_id
                        ).execute()

                        file_data = base64.urlsafe_b64decode(attachment['data'])

                        # Check file size
                        if len(file_data) <= MAX_ATTACHMENT_SIZE:
                            attachments.append((filename, file_data))
                            logger.info(f"[ATTACH] Downloaded: {filename} ({len(file_data)} bytes)")
                        else:
                            logger.warning(
                                f"[ATTACH] Skipped {filename} - exceeds max size "
                                f"({len(file_data)} > {MAX_ATTACHMENT_SIZE} bytes)"
                            )
                    except Exception as e:
                        logger.error(f"[ATTACH] Error downloading {filename}: {e}")

        return attachments

    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from email body text.

        Args:
            text: Email body text

        Returns:
            List of URLs found
        """
        # URL regex pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'

        urls = re.findall(url_pattern, text)
        urls = list(set(urls))  # Remove duplicates

        logger.info(f"[URLs] Found {len(urls)} unique URL(s)")
        for url in urls[:5]:  # Log first 5
            logger.info(f"       {url[:80]}")

        return urls

    def save_attachments(self, attachments: List[Tuple[str, bytes]]) -> List[str]:
        """
        Save attachments to temporary directory.

        Args:
            attachments: List of (filename, content) tuples

        Returns:
            List of file paths
        """
        file_paths = []
        for filename, content in attachments:
            # Sanitize filename
            safe_filename = re.sub(r'[^\w\s.-]', '_', filename)
            file_path = TEMP_ATTACHMENT_DIR / safe_filename

            try:
                with open(file_path, 'wb') as f:
                    f.write(content)
                file_paths.append(str(file_path))
                logger.info(f"[SAVE_ATTACH] Saved: {file_path}")
            except Exception as e:
                logger.error(f"[SAVE_ATTACH] Error saving {filename}: {e}")

        return file_paths

    def cleanup_temp_dir(self) -> None:
        """Clean up temporary attachment directory."""
        try:
            if TEMP_ATTACHMENT_DIR.exists():
                shutil.rmtree(TEMP_ATTACHMENT_DIR)
                TEMP_ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
                logger.info("[CLEANUP] Temporary directory cleaned")
        except Exception as e:
            logger.error(f"[CLEANUP] Error cleaning temp directory: {e}")


class SentinelGmailScanner:
    """
    Main scanner combining Gmail integration with Sentinel ML inference.
    """

    def __init__(self):
        """Initialize Gmail integration and Sentinel ensemble."""
        logger.info("[INIT] Initializing Sentinel Gmail Scanner...")

        self.gmail = GmailIntegration()
        self.ensemble = SentinelEnsemble()

        logger.info("[INIT] ✅ Scanner ready! Waiting for emails...")
        logger.info(f"[INIT] Polling interval: {POLL_INTERVAL} seconds")
        logger.info("[INIT] " + "=" * 100)

    def extract_stream_c_features(self, file_path: str) -> str:
        """
        Load attachment file for Stream C analysis.

        Args:
            file_path: Path to attachment file

        Returns:
            File path for processing
        """
        return file_path

    def scan_email(self, message_id: str) -> Optional[Dict]:
        """
        Complete scanning pipeline for a single email.

        Args:
            message_id: Gmail message ID

        Returns:
            Scan results dictionary
        """
        logger.info(f"\n[SCAN] Starting scan of message {message_id}...")

        # Get message details
        message = self.gmail.get_message_details(message_id)
        if not message:
            logger.error("[SCAN] Failed to retrieve message details")
            return None

        # Extract components
        sender, subject, body, html_body, attachments = self.gmail.extract_email_parts(message)

        # Extract URLs
        urls = self.gmail.extract_urls(body)

        # Save attachments
        attachment_paths = self.gmail.save_attachments(attachments)

        # Prepare inputs for ensemble
        text_input = body[:2000]  # First 2000 chars for efficiency
        url_list = urls if urls else None
        attachment_file = attachment_paths[0] if attachment_paths else None

        logger.info(
            f"[SCAN] Prepared inputs: "
            f"Text={len(text_input)}chars, URLs={len(urls) if urls else 0}, "
            f"Attachments={len(attachment_paths)}"
        )

        # === STREAM A: TEXT ANALYSIS ===
        logger.info("[STREAM_A] Running DistilBERT text analysis...")
        try:
            from ensemble_engine import StreamATextAnalyzer
            stream_a = StreamATextAnalyzer()
            stream_a_conf = stream_a.classify(text_input)
            logger.info(f"[STREAM_A] ✅ Confidence: {stream_a_conf:.4f}")
        except Exception as e:
            logger.warning(f"[STREAM_A] Failed - using default: {e}")
            stream_a_conf = None

        # === STREAM B: URL ANALYSIS ===
        logger.info("[STREAM_B] Running XGBoost URL analysis...")
        try:
            from ensemble_engine import StreamBUrlAnalyzer
            stream_b = StreamBUrlAnalyzer()
            if url_list:
                stream_b_conf = stream_b.classify(url_list)
            else:
                logger.info("[STREAM_B] No URLs found - setting to None")
                stream_b_conf = None
            logger.info(f"[STREAM_B] ✅ Confidence: {stream_b_conf:.4f if stream_b_conf else 'None'}")
        except Exception as e:
            logger.warning(f"[STREAM_B] Failed - using default: {e}")
            stream_b_conf = None

        # === STREAM C: ATTACHMENT ANALYSIS ===
        logger.info("[STREAM_C] Running attachment structural analysis...")
        try:
            from ensemble_engine import StreamCAttachmentAnalyzer
            stream_c = StreamCAttachmentAnalyzer()
            if attachment_file:
                stream_c_result = stream_c.analyze_structure(attachment_file)
                stream_c_conf = stream_c_result['confidence']
                threat_description = stream_c_result.get('threat_description', None)
            else:
                logger.info("[STREAM_C] No attachments - setting to None")
                stream_c_conf = None
                threat_description = None
            logger.info(f"[STREAM_C] ✅ Confidence: {stream_c_conf:.4f if stream_c_conf else 'None'}")
        except Exception as e:
            logger.warning(f"[STREAM_C] Failed - using default: {e}")
            stream_c_conf = None
            threat_description = None

        # === ENSEMBLE FUSION ===
        logger.info("[ENSEMBLE] Fusing stream predictions...")
        try:
            fusion_result = self.ensemble.fuse_predictions(
                stream_a_conf=stream_a_conf,
                stream_b_conf=stream_b_conf,
                stream_c_conf=stream_c_conf
            )
            ensemble_score = fusion_result['ensemble_score']
            ensemble_verdict = fusion_result['verdict']
            ensemble_confidence = fusion_result['confidence']

            logger.info(f"[ENSEMBLE] ✅ Score: {ensemble_score:.4f}, Verdict: {ensemble_verdict}")
        except Exception as e:
            logger.error(f"[ENSEMBLE] Fusion failed: {e}")
            return None

        # === SHAP EXPLANATION ===
        logger.info("[SHAP] Computing SHAP explanations...")
        try:
            shap_explanation = self.ensemble.explain_decision(
                stream_a_conf=stream_a_conf,
                stream_b_conf=stream_b_conf,
                stream_c_conf=stream_c_conf,
                c_threat_description=threat_description
            )
            logger.info("[SHAP] ✅ Explanation computed successfully")
        except Exception as e:
            logger.error(f"[SHAP] Failed: {e}")
            shap_explanation = None

        # Compile results
        results = {
            'message_id': message_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'sender': sender,
            'subject': subject,
            'text_length': len(body),
            'urls_found': len(urls),
            'attachments_found': len(attachment_paths),
            'stream_a_conf': stream_a_conf,
            'stream_b_conf': stream_b_conf,
            'stream_c_conf': stream_c_conf,
            'ensemble_score': ensemble_score,
            'ensemble_verdict': ensemble_verdict,
            'ensemble_confidence': ensemble_confidence,
            'threat_description': threat_description,
            'shap_explanation': shap_explanation
        }

        # Mark message as read
        self.gmail.mark_as_read(message_id)

        # Generate and print report
        self.print_scan_report(results)

        # Cleanup
        self.gmail.cleanup_temp_dir()

        return results

    def print_scan_report(self, results: Dict) -> None:
        """
        Print formatted scan report to terminal.

        Args:
            results: Scan results dictionary
        """
        print("\n" + "=" * 120)
        print("█" * 120)
        print("█  SENTINEL GMAIL INTEGRATION - REAL-TIME PHISHING DETECTION SCAN REPORT")
        print("█" * 120)
        print("=" * 120 + "\n")

        # Header Information
        print(f"╔{'═' * 118}╗")
        print(f"║ {'SCAN TIMESTAMP':<25} {results['timestamp']:<92} ║")
        print(f"║ {'MESSAGE ID':<25} {results['message_id']:<92} ║")
        print(f"╚{'═' * 118}╝\n")

        # Email Information
        print(f"┌─ EMAIL DETAILS {'─' * 100}┐")
        print(f"│ Sender:     {results['sender']:<110}")
        print(f"│ Subject:    {results['subject']:<110}")
        print(f"│ Text:       {results['text_length']:>6} chars  │  URLs: {results['urls_found']:>3}  │  Attachments: {results['attachments_found']:>2}")
        print(f"└{'─' * 118}┘\n")

        # Stream Scores
        print(f"┌─ ML STREAM CONFIDENCE SCORES {'─' * 87}┐")
        stream_a_score = (
            f"{results['stream_a_conf']:.4f}"
            if results['stream_a_conf'] is not None
            else "N/A (no text)"
        )
        stream_b_score = (
            f"{results['stream_b_conf']:.4f}"
            if results['stream_b_conf'] is not None
            else "N/A (no URLs)"
        )
        stream_c_score = (
            f"{results['stream_c_conf']:.4f}"
            if results['stream_c_conf'] is not None
            else "N/A (no attach)"
        )

        print(f"│")
        print(f"│  Stream A (Text):        {stream_a_score:>15}  │  DistilBERT email body analysis")
        print(f"│  Stream B (URLs):        {stream_b_score:>15}  │  XGBoost URL feature analysis")
        print(f"│  Stream C (Attachment):  {stream_c_score:>15}  │  Structural threat detection")
        print(f"│")
        print(f"└{'─' * 118}┘\n")

        # Threat Description
        if results['threat_description']:
            threat_icon = "⚠️  "
            print(f"┌─ THREAT DETECTED {'─' * 99}┐")
            print(f"│ {threat_icon}{results['threat_description']:<112}")
            print(f"└{'─' * 118}┘\n")

        # Final Verdict
        verdict = results['ensemble_verdict'].upper()
        confidence = results['ensemble_confidence']

        if verdict == "BENIGN":
            verdict_icon = "✅"
            verdict_color = "SAFE"
        else:
            verdict_icon = "🚨"
            verdict_color = "DANGEROUS"

        print(f"╔{'═' * 118}╗")
        print(f"║                                   {verdict_icon} FINAL VERDICT: {verdict_color:<90} ║")
        print(f"║                                   Confidence: {confidence:.1%} (threshold: 70%)                            ║")
        print(f"╚{'═' * 118}╝\n")

        # SHAP Explanation
        if results['shap_explanation']:
            exp = results['shap_explanation']
            print(f"┌─ EXPLAINABILITY (SHAP - Shapley Additive exPlanations) {'─' * 50}┐")
            if 'explanation' in exp:
                explanation_text = exp['explanation']
                # Print first 500 chars of explanation
                print(f"│")
                for line in explanation_text.split('\n')[:15]:  # First 15 lines
                    if line:
                        print(f"│  {line:<115}")
                print(f"│")
        else:
            print(f"┌─ EXPLAINABILITY {'─' * 99}┐")
            print(f"│  SHAP explanation unavailable")
            print(f"│")

        print(f"└{'─' * 118}┘\n")

        print("=" * 120)
        print("█" * 120)
        print("█  Scan complete. System continuing to monitor for new messages...")
        print("█" * 120)
        print("=" * 120 + "\n")

    def run_watchdog(self) -> None:
        """
        Main watchdog loop: Poll Gmail every 10 seconds for unread messages.
        """
        logger.info("\n[WATCHDOG] Starting Gmail watchdog loop...")
        processed_messages = set()

        try:
            while True:
                logger.info(f"[WATCHDOG] Polling Gmail (interval: {POLL_INTERVAL}s)...")

                messages = self.gmail.get_unread_messages(max_results=5)

                if messages:
                    logger.info(f"[WATCHDOG] Found {len(messages)} unread message(s)")

                    for message in messages:
                        msg_id = message['id']

                        # Skip if already processed
                        if msg_id in processed_messages:
                            logger.debug(f"[WATCHDOG] Skipping already-processed message {msg_id}")
                            continue

                        logger.info(f"[WATCHDOG] New message detected! Processing...")
                        result = self.scan_email(msg_id)

                        if result:
                            processed_messages.add(msg_id)
                else:
                    logger.info("[WATCHDOG] No unread messages found (waiting for new emails...)")

                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n[WATCHDOG] Received interrupt - shutting down gracefully...")
            self.gmail.cleanup_temp_dir()
            logger.info("[WATCHDOG] ✅ Cleanup complete. Goodbye!")
            sys.exit(0)


def main():
    """Main entry point."""
    print("\n" + "█" * 120)
    print("█" + " " * 118 + "█")
    print("█" + "  SENTINEL PHASE 5: LIVE GMAIL INTEGRATION TEST".center(118) + "█")
    print("█" + "  Real-Time Phishing Detection on Live Email".center(118) + "█")
    print("█" + " " * 118 + "█")
    print("█" * 120 + "\n")

    logger.info("=" * 120)
    logger.info("SENTINEL GMAIL INTEGRATION TEST - STARTING")
    logger.info("=" * 120)

    try:
        scanner = SentinelGmailScanner()
        scanner.run_watchdog()
    except Exception as e:
        logger.error(f"[FATAL] Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
