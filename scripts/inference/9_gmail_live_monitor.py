#!/usr/bin/env python3
"""
Script 9: Live Sentinel Shield - Always-On Phishing Detection
==============================================================

Real-time IMAP monitoring with persistent loop and IMAP IDLE support.
Integrates with SentinelCore for continuous 3-stream phishing analysis.

Features:
- Always-on monitoring with IMAP IDLE for server-push notifications
- State management tracking processed UIDs
- Real-time Shield Report with formatted threat alerts
- Graceful reconnection on network failures
- Full 3-stream ensemble fusion with SHAP explanations

CRITICAL IMAP IMPLEMENTATION NOTES:
1. Message Walking: Uses for part in msg.walk() to traverse multipart structures
2. Filename Sanitization: Safely handles None filenames with try/except blocks
3. Byte-Stream Integrity: Calls part.get_payload(decode=True) for Base64 decoding

Security:
- Credentials loaded from .env file (never hardcoded)
- Base64 payloads always decoded before analysis
- Corrupt filenames caught and logged without crashing
- State file persists processed UIDs between restarts
"""

import sys
import email
from email.message import Message
import imaplib
import time
import json
import re
import socket
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set
import tempfile
import logging
from logging.handlers import RotatingFileHandler

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

# Setup dual logging: console + persistent file with automatic rotation
console_formatter = logging.Formatter(
    '%(message)s'  # Clean console output
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler]
)
logger = logging.getLogger(__name__)

# Add file handler with rotation: max 5MB per file, keep 3 backups
log_file = Path(__file__).parent.parent.parent / ".sentinel_shield.log"
try:
    file_handler = RotatingFileHandler(
        log_file,
        mode='a',
        maxBytes=5 * 1024 * 1024,  # 5MB per file
        backupCount=3,  # Keep .log.1, .log.2, .log.3 as backups
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not create log file: {e}")

# Print startup banner
print("\n" + "="*70)
print("        🛡️  LIVE SENTINEL SHIELD  🛡️")
print("      Always-On Phishing Detection Monitor")
print("="*70 + "\n")


# ============================================================================
# PHASE 8: CONFIGURATION LOADING (Open-Source Readiness)
# ============================================================================

def load_config() -> Dict:
    """
    Load SENTINEL configuration from sentinel_config.yaml.
    Falls back to sensible defaults if config file doesn't exist.
    Ensures cross-OS compatibility with Path operations.
    """
    config_path = Path(__file__).parent.parent.parent / "sentinel_config.yaml"
    
    # Default config (backward compatible with existing behavior)
    default_config = {
        'imap': {
            'server': 'imap.gmail.com',
            'port': 993,
            'folder_to_monitor': 'INBOX',
            'polling_interval_seconds': 30,
            'connection_timeout_seconds': 30,
            'idle_timeout_seconds': 600
        },
        'security': {
            'whitelist_domains': [
                'accounts.google.com',
                'no-reply@accounts.google.com',
                'security@microsoft.com',
                'noreply@google.com',
                'support@google.com',
                'accounts@google.com',
                'security-noreply@google.com'
            ],
            'max_attachment_size_mb': 50,
            'analysis_timeout_seconds': 30,
            'dangerous_extensions': ['.exe', '.bat', '.com', '.pif', '.scr', '.vbs', '.js', '.jar', '.zip', '.rar', '.7z']
        },
        'models': {
            'enable_distilbert': True,
            'enable_xgboost': True,
            'enable_attachment_analysis': True,
            'fallback_to_heuristics': True,
            'use_gpu_if_available': True
        },
        'storage': {
            'state_file': '.sentinel_state.json',
            'metrics_file': '.sentinel_metrics.json',
            'threat_registry_file': '.sentinel_threat_registry.json',
            'log_file': '.sentinel_shield.log',
            'temp_attachment_dir': 'Data/temp_attachments'
        }
    }
    
    # Try to load YAML config if available and file exists
    if YAML_AVAILABLE and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f) or {}
                # Merge with defaults (loaded config takes precedence)
                for key in default_config:
                    if key in loaded_config and isinstance(loaded_config[key], dict):
                        default_config[key].update(loaded_config[key])
                    elif key in loaded_config:
                        default_config[key] = loaded_config[key]
                logger.info(f"[CONFIG] Loaded from {config_path}")
                return default_config
        except Exception as e:
            logger.warning(f"[CONFIG] Failed to load {config_path}: {e}. Using defaults.")
            return default_config
    else:
        if not YAML_AVAILABLE:
            logger.info("[CONFIG] PyYAML not installed. Using hardcoded defaults. Install with: pip install pyyaml")
        else:
            logger.info(f"[CONFIG] {config_path} not found. Using hardcoded defaults.")
        return default_config


# Load configuration at startup
CONFIG = load_config()
TRUSTED_SENDER_DOMAINS = CONFIG['security']['whitelist_domains']
MAX_ATTACHMENT_SIZE_MB = CONFIG['security']['max_attachment_size_mb']
ANALYSIS_TIMEOUT_SECONDS = CONFIG['security']['analysis_timeout_seconds']
IMAP_SERVER = CONFIG['imap']['server']
IMAP_PORT = CONFIG['imap']['port']
FOLDER_TO_MONITOR = CONFIG['imap']['folder_to_monitor']


class LiveSentinelShield:
    """
    Always-On Sentinel Shield - Real-time phishing detection with IMAP IDLE.
    
    Architecture:
    - IMAP IDLE for server-push notifications
    - State management to track processed emails
    - Full 3-stream ensemble fusion with SHAP
    - Shield Report formatting for threat alerts
    - Automatic reconnection on failures
    """
    
    def __init__(self, email_address: str, app_password: str):
        """
        Initialize Sentinel Shield.
        
        Args:
            email_address: Gmail address (e.g., user@gmail.com)
            app_password: 16-character Gmail App Password
        """
        self.email_address = email_address
        self.app_password = app_password
        self.imap_server = None
        self.temp_dir = Path(tempfile.gettempdir()) / "sentinel_shield_downloads"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # State management
        self.state_file = Path(__file__).parent.parent.parent / ".sentinel_state.json"
        self.processed_uids = self._load_processed_uids()
        self.idle_timeout = 280  # IMAP IDLE timeout (28 minutes, RFC 6143)
        
        # Threat registry - PERSISTENT HISTORY (never resets)
        self.threat_registry_file = Path(__file__).parent.parent.parent / ".sentinel_threat_registry.json"
        self.threat_registry = self._load_threat_registry()
        
        # Metrics tracking - PERSISTENT across sessions
        self.metrics = {
            'total_analyzed': 0,
            'threats_detected': 0,
            'safe_emails': 0,
            'suspicious_emails': 0,
            'emails_with_attachments': 0,
            'emails_with_urls': 0,
            'read_emails': 0,
            'unread_emails': 0,
            'reply_emails': 0,
            'domain_distribution': {},
            'attachment_types': {},
            'threat_by_sender': {},
            'current_inbox_count': 0  # NEW: Track actual inbox size for live deletions
        }
        
        # Metrics file path
        self.metrics_file = Path(__file__).parent.parent.parent / ".sentinel_metrics.json"
        
        # LOAD PERSISTENT METRICS if file exists, else initialize to 0
        self._init_metrics()
        
        # Sentinel Core for analysis
        self.sentinel = None
        
        # Log init (to file only, not console)
        logger.info(f"[STARTUP] LiveSentinelShield initialized for {email_address}")
        logger.info(f"[STARTUP] Tracking {len(self.processed_uids)} previously processed emails")
        logger.info(f"[STARTUP] Metrics auto-reset to fresh state")
    
    def _init_metrics(self):
        """Load persistent metrics from file, or initialize to 0 if file doesn't exist."""
        try:
            # If metrics file exists, LOAD it (preserve history across restarts)
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # Merge loaded metrics with default template (in case new keys were added)
                    for key in self.metrics:
                        if key in loaded:
                            self.metrics[key] = loaded[key]
                logger.info("[STARTUP] Loaded persistent metrics from previous session")
            else:
                # First run - save the empty initialized metrics
                with open(self.metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metrics, f, indent=2)
                logger.info("[STARTUP] Initialized new metrics file")
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            # Initialize empty metrics as fallback
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2)
    
    def init_sentinel(self):
        """
        Initialize SentinelCore with all three analytical streams.
        
        Called after IMAP connection is verified.
        Loads DistilBERT (Stream A), XGBoost (Stream B), and Attachment Analyzer (Stream C).
        
        This method MUST be called before any emails are analyzed.
        """
        try:
            if self.sentinel is not None:
                logger.info("[SENTINEL] SentinelCore already initialized, skipping")
                return
            
            # Get model directory (workspace root / models)
            model_dir = Path(__file__).parent.parent.parent / "models"
            
            if not model_dir.exists():
                raise RuntimeError(f"[ERROR] Models directory not found: {model_dir}")
            
            logger.info(f"[SENTINEL] Loading SentinelCore with models from {model_dir}")
            
            # Import and instantiate SentinelCore
            from sentinel_core import SentinelCore
            self.sentinel = SentinelCore(model_dir=str(model_dir))
            
            logger.info("[SENTINEL] ✓ SentinelCore initialized successfully")
            logger.info("[SENTINEL]   - Stream A (DistilBERT): Text Classification")
            logger.info("[SENTINEL]   - Stream B (XGBoost): URL Analysis") 
            logger.info("[SENTINEL]   - Stream C (Attachment Analyzer): File Analysis")
            
        except ImportError as e:
            logger.error(f"[SENTINEL] Failed to import SentinelCore: {e}")
            raise RuntimeError(f"SentinelCore import failed. Ensure sentinel_core.py exists in scripts/inference/: {e}")
        except FileNotFoundError as e:
            logger.error(f"[SENTINEL] Model files not found: {e}")
            raise RuntimeError(f"Required model files not found. Check models/ directory: {e}")
        except Exception as e:
            logger.error(f"[SENTINEL] Failed to initialize SentinelCore: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"SentinelCore initialization failed: {e}")
    
    def analyze_with_timeout(self, email_content: str, urls: Optional[List[str]] = None, 
                             attachment_paths: Optional[List[str]] = None, 
                             timeout_secs: int = 60) -> Dict:
        """
        Wrapper around sentinel.analyze() with timeout protection.
        
        Prevents the monitoring loop from freezing indefinitely if model inference hangs.
        Uses threading to implement timeout.
        
        Args:
            email_content: Email text to analyze
            urls: List of URLs in email
            attachment_paths: List of attachment file paths
            timeout_secs: Max seconds to wait (default 60s)
            
        Returns:
            Analysis result dict, or error dict if timeout occurs
        """
        import threading
        
        result = {}
        exception = [None]  # Use list to allow modification in nested function
        
        def run_analysis():
            try:
                result['analysis'] = self.sentinel.analyze(
                    email_content=email_content,
                    urls=urls,
                    attachment_paths=attachment_paths
                )
            except Exception as e:
                exception[0] = e
        
        # Start analysis in background thread
        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()
        thread.join(timeout=timeout_secs)
        
        # Check if thread is still alive (timeout occurred)
        if thread.is_alive():
            logger.error(f"[TIMEOUT] Sentinel analysis exceeded {timeout_secs}s timeout - returning UNKNOWN verdict")
            return {
                'final_verdict': 'UNKNOWN',
                'final_confidence': 0.5,
                'stream_breakdown': {'stream_a': None, 'stream_b': None, 'stream_c': None},
                'xai_explanation': {
                    'override_triggered': True,
                    'override_reason': f'Analysis timeout after {timeout_secs}s',
                    'human_readable': f'Email analysis exceeded {timeout_secs}s timeout limit'
                },
                'metadata': {
                    'timeout_occurred': True,
                    'timeout_seconds': timeout_secs,
                    'has_urls': bool(urls and len(urls) > 0),
                    'has_attachments': bool(attachment_paths and len(attachment_paths) > 0),
                }
            }
        
        # Check if exception occurred
        if exception[0]:
            logger.error(f"[ANALYZE_ERROR] Exception during analysis: {type(exception[0]).__name__}: {exception[0]}")
            return {
                'final_verdict': 'UNKNOWN',
                'final_confidence': 0.5,
                'stream_breakdown': {'stream_a': None, 'stream_b': None, 'stream_c': None},
                'xai_explanation': {
                    'override_triggered': True,
                    'override_reason': f'Analysis error: {type(exception[0]).__name__}',
                    'human_readable': f'Email analysis failed: {str(exception[0])[:100]}'
                },
                'metadata': {
                    'error_occurred': True,
                    'error_type': type(exception[0]).__name__,
                    'has_urls': bool(urls and len(urls) > 0),
                    'has_attachments': bool(attachment_paths and len(attachment_paths) > 0),
                }
            }
        
        # Return successful result - CRITICAL: Check if analysis key exists
        if 'analysis' not in result or not result['analysis']:
            logger.error(f"[ANALYZE_ERROR] No analysis result returned - thread may not have completed")
            return {
                'final_verdict': 'UNKNOWN',
                'final_confidence': 0.5,
                'stream_breakdown': {'stream_a': None, 'stream_b': None, 'stream_c': None},
                'xai_explanation': {
                    'override_triggered': True,
                    'override_reason': 'No analysis result returned',
                    'human_readable': 'Email analysis produced no result'
                },
                'metadata': {
                    'no_result': True,
                    'has_urls': bool(urls and len(urls) > 0),
                    'has_attachments': bool(attachment_paths and len(attachment_paths) > 0),
                }
            }
        return result.get('analysis', {})
    
    def _is_trusted_sender(self, from_addr: str) -> bool:
        """
        CRITICAL SECURITY FIX: Properly validate email domain against whitelist.
        
        Uses EXACT email matching instead of dangerous substring matching.
        Extracts actual domain from email address (part after @).
        
        This prevents spoofing attacks like:
        - "attacker@evil-google.com" (substring "google.com" would match)
        - "google.com@evil-attacker.com" (same issue)
        
        Args:
            from_addr: Full email address (e.g., "user@google.com")
            
        Returns:
            True if email matches EXACTLY with trusted senders
        """
        if not from_addr or '@' not in from_addr:
            return False
        
        # Extract actual email domain (part after @)
        try:
            actual_domain = from_addr.lower().split('@')[-1].strip('> ')
        except Exception:
            return False
        
        # Whitelist of EXACT email addresses to trust
        # ONLY legitimate Gmail/Google and Microsoft security accounts
        TRUSTED_EXACT_EMAILS = {
            'no-reply@accounts.google.com',
            'security@microsoft.com',
            'noreply@google.com',
            'support@google.com',
            'accounts@google.com',
            'security-noreply@google.com',
        }
        
        # Whitelist of EXACT domains (only these specific domains, checked with == not 'in')
        TRUSTED_EXACT_DOMAINS = {
            'accounts.google.com',
            'google.com',
            'security.microsoft.com',
            'microsoft.com',
        }
        
        # Check exact email match first (most restrictive)
        if from_addr.lower().strip() in TRUSTED_EXACT_EMAILS:
            logger.info(f"[WHITELIST] Exact email match: {from_addr}")
            return True
        
        # Check exact domain match (NOT substring!)
        if actual_domain in TRUSTED_EXACT_DOMAINS:
            logger.info(f"[WHITELIST] Exact domain match: {from_addr} (domain: {actual_domain})")
            return True
        
        return False
    
    # DUPLICATE METHODS REMOVED - Using proper implementation below at line ~339
    
    def cleanup_orphaned_files(self):
        """Remove orphaned files from temp directory."""
        try:
            if self.temp_dir.exists():
                for file_path in self.temp_dir.glob('*'):
                    try:
                        file_path.unlink()
                        logger.info(f"[CLEANUP] Deleted orphaned file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Could not delete {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error in cleanup_orphaned_files: {e}")
    
    def _load_processed_uids(self) -> Set[str]:
        """Load set of already-processed UIDs from state file with UTF-8-sig encoding (handles BOM)."""
        if self.state_file.exists():
            try:
                # Use utf-8-sig to handle BOM (Byte Order Mark) that may exist in file
                with open(self.state_file, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                    uids = set(data.get('processed_uids', []))
                    logger.info(f"[STATE] Loaded {len(uids)} previously processed email UIDs")
                    return uids
            except json.JSONDecodeError as e:
                logger.error(f"[STATE] Corrupted state file (JSON decode error): {e}")
                logger.warning(f"[STATE] Resetting state file due to corruption")
                return set()
            except Exception as e:
                logger.warning(f"[STATE] Failed to load state file: {e}")
                return set()
        return set()
    
    def _save_processed_uids(self):
        """Persist processed UIDs to state file with UTF-8 encoding (no BOM)."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_uids': sorted(list(self.processed_uids)),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            logger.debug(f"[STATE] Saved {len(self.processed_uids)} processed UIDs to state file")
        except Exception as e:
            logger.error(f"[STATE] Failed to save state file: {e}")
    
    def _load_threat_registry(self) -> Dict:
        """Load persistent threat registry from file (never resets across sessions)."""
        if self.threat_registry_file.exists():
            try:
                with open(self.threat_registry_file, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                    threats = data.get('threats', [])
                    logger.info(f"[REGISTRY] Loaded {len(threats)} historical threat detections from registry")
                    return data
            except Exception as e:
                logger.warning(f"[REGISTRY] Could not load threat registry: {e}")
                return {'threats': [], 'last_updated': datetime.now().isoformat()}
        return {'threats': [], 'last_updated': datetime.now().isoformat()}
    
    def _save_threat_registry(self):
        """Persist threat registry to file (never deletes old threats)."""
        try:
            self.threat_registry['last_updated'] = datetime.now().isoformat()
            with open(self.threat_registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.threat_registry, f, indent=2, ensure_ascii=False)
            logger.debug(f"[REGISTRY] Saved threat registry with {len(self.threat_registry.get('threats', []))} total threats")
        except Exception as e:
            logger.error(f"[REGISTRY] Failed to save threat registry: {e}")
    
    def _register_threat(self, email_info: Dict, analysis: Dict):
        """Record a phishing threat to persistent registry with complete details."""
        if analysis.get('final_verdict') != 'PHISHING':
            return  # Only record actual threats
        
        # CRITICAL FIX 7: EXPLICIT SCORE EXTRACTION WITH FALLBACK DEFAULTS
        # Extract scores from multiple possible locations in the analysis dict
        # Priority: explicit score_a/b/c > stream_breakdown > fallback to 0.0
        
        stream_breakdown = analysis.get('stream_breakdown') or {}
        stream_a_info = stream_breakdown.get('stream_a') or {}
        stream_b_info = stream_breakdown.get('stream_b') or {}
        stream_c_info = stream_breakdown.get('stream_c') or {}
        xai_explanation = analysis.get('xai_explanation') or {}
        attachments = email_info.get('attachments') or []
        
        # CRITICAL: Use explicit score_a/b/c if available, otherwise extract from stream_breakdown
        score_a = analysis.get('score_a')
        score_b = analysis.get('score_b')
        score_c = analysis.get('score_c')
        
        # Fallback to stream_breakdown confidence if explicit scores not available
        if score_a is None:
            score_a = stream_a_info.get('confidence')
        if score_b is None:
            score_b = stream_b_info.get('confidence')
        if score_c is None:
            score_c = stream_c_info.get('confidence')
        
        # Final fallback: 0.0 (ensures JSON always has numeric values, never null)
        score_a = score_a if score_a is not None else 0.0
        score_b = score_b if score_b is not None else 0.0
        score_c = score_c if score_c is not None else 0.0
        
        threat_entry = {
            'uid': email_info.get('uid', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'sender': email_info.get('from', 'unknown'),
            'subject': email_info.get('subject', '(No Subject)'),
            'final_confidence': analysis.get('final_confidence', 0),
            # CRITICAL FIX 8: BOOLEAN FLAGS FOR UI DISPLAY
            # Dashboard needs explicit has_urls and has_attachments to show green checkmarks
            'has_urls': len(email_info.get('urls', [])) > 0 if email_info and 'urls' in email_info else False,
            'has_attachments': len(email_info.get('attachments', [])) > 0 if email_info and 'attachments' in email_info else False,
            # CRITICAL: Always write numeric scores (never null) for dashboard display
            'stream_a_score': float(score_a),
            'stream_b_score': float(score_b),
            'stream_c_score': float(score_c),
            'shap_explanation': xai_explanation.get('human_readable', 'No explanation available'),
            'urls_detected': email_info.get('urls') or [],
            'attachments_detected': [att.get('original_filename', 'unknown') if isinstance(att, dict) else 'unknown' for att in attachments],
            'stream_a_reasoning': stream_a_info.get('reasoning', ''),
            'stream_b_reasoning': stream_b_info.get('reasoning', ''),
            'stream_c_reasoning': stream_c_info.get('reasoning', ''),
            # Also include stream_breakdown for reference
            'stream_breakdown': {
                'stream_a': score_a,
                'stream_b': score_b,
                'stream_c': score_c
            }
        }
        
        # Add to registry
        self.threat_registry['threats'].append(threat_entry)
        self._save_threat_registry()
        logger.info(f"[REGISTRY] Recorded threat #{len(self.threat_registry['threats'])} | Sender: {email_info.get('from', 'unknown')} | Confidence: {analysis.get('final_confidence', 0):.1%} | Scores: A={score_a:.2f} B={score_b:.2f} C={score_c:.2f}")
    
    def get_threat_summary(self) -> Dict:
        """Get summary statistics of all threats detected (for reporting)."""
        threats = self.threat_registry.get('threats', [])
        if not threats:
            return {
                'total_threats': 0,
                'unique_senders': 0,
                'average_confidence': 0,
                'threats': []
            }
        
        unique_senders = set(t['from'] for t in threats)
        avg_confidence = sum(t['final_confidence'] for t in threats) / len(threats) if threats else 0
        
        return {
            'total_threats': len(threats),
            'unique_senders': len(unique_senders),
            'average_confidence': round(avg_confidence, 2),
            'earliest_threat': threats[0]['detected_at'] if threats else None,
            'latest_threat': threats[-1]['detected_at'] if threats else None,
            'threats': threats[-10:]  # Return last 10 for display
        }
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL, handling various URL formats."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www. prefix for consistency
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Return domain if valid
            return domain if domain else None
        except Exception as e:
            logger.debug(f"Failed to parse URL {url}: {e}")
            return None
    
    def update_metrics(self, analysis: Dict, email_info: Dict, read_status: str):
        """Update metrics based on email analysis results."""
        try:
            verdict = analysis.get('final_verdict', 'UNKNOWN')
            stream_breakdown = analysis.get('stream_breakdown') or {}
            attachments = email_info.get('attachments') or []
            urls = email_info.get('urls') or []
            
            # Count verdict types
            self.metrics['total_analyzed'] += 1
            if verdict == 'PHISHING':
                self.metrics['threats_detected'] += 1
                # Record this threat in persistent registry
                self._register_threat(email_info, analysis)
            elif verdict == 'SAFE':
                self.metrics['safe_emails'] += 1
            else:
                self.metrics['suspicious_emails'] += 1
            
            # Read/Unread tracking
            if read_status == 'READ':
                self.metrics['read_emails'] += 1
            else:
                self.metrics['unread_emails'] += 1
            
            # Attachment tracking - Extract file extensions properly
            if attachments:
                self.metrics['emails_with_attachments'] += 1
                for att in attachments:
                    if not isinstance(att, dict):
                        continue
                    # Use 'original_filename' key from extract_attachments output
                    filename = att.get('original_filename', '') or att.get('filename', '')
                    if filename and '.' in filename:
                        ext = filename.split('.')[-1].upper()
                    else:
                        ext = 'UNKNOWN'
                    self.metrics['attachment_types'][ext] = self.metrics['attachment_types'].get(ext, 0) + 1
            
            # URL tracking
            if urls:
                self.metrics['emails_with_urls'] += 1
            
            # Reply detection
            if 'Re:' in email_info.get('subject', '') or 'RE:' in email_info.get('subject', ''):
                self.metrics['reply_emails'] += 1
            
            # Domain tracking - Sender email domain
            from_addr = email_info.get('from', '')
            try:
                domain = from_addr.split('@')[-1].rstrip('>') if '@' in from_addr else 'unknown'
                self.metrics['domain_distribution'][domain] = self.metrics['domain_distribution'].get(domain, 0) + 1
            except:
                pass
            
            # URL domain tracking - Extract and track domains from URLs in email
            if urls:
                for url in urls:
                    try:
                        url_domain = self._extract_domain_from_url(url)
                        if url_domain:
                            self.metrics['domain_distribution'][url_domain] = \
                                self.metrics['domain_distribution'].get(url_domain, 0) + 1
                    except Exception as e:
                        logger.debug(f"Failed to extract domain from URL {url}: {e}")
            
            # Threat by sender
            if verdict == 'PHISHING':
                sender = from_addr.split('<')[0].strip() if '<' in from_addr else from_addr.split('@')[0]
                self.metrics['threat_by_sender'][sender] = self.metrics['threat_by_sender'].get(sender, 0) + 1

            # Persist live state after every processed email so the dashboard stays current.
            self.save_metrics()
        
        except Exception as e:
            logger.warning(f"Error updating metrics: {type(e).__name__}: {e}")
    
    def save_metrics(self):
        """Persist metrics to file with heartbeat timestamp."""
        try:
            # Add heartbeat timestamp to show monitor is alive
            metrics_with_heartbeat = self.metrics.copy()
            metrics_with_heartbeat['last_heartbeat'] = datetime.now().isoformat()
            
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_with_heartbeat, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
    
    def print_metrics_dashboard(self):
        """Print a formatted metrics dashboard."""
        try:
            total = self.metrics.get('total_analyzed', 0)
            if total == 0:
                return
            
            threats = self.metrics.get('threats_detected', 0)
            safe = self.metrics.get('safe_emails', 0)
            suspicious = self.metrics.get('suspicious_emails', 0)
            
            threat_rate = (threats / total * 100) if total > 0 else 0
            safe_rate = (safe / total * 100) if total > 0 else 0
            
            print()
            print("=" * 70)
            print("📊 SENTINEL SHIELD METRICS DASHBOARD")
            print("=" * 70)
            print(f"  Total Analyzed:     {total:6d}  |  Safe: {safe:4d}  Threats: {threats:4d}  Suspicious: {suspicious:4d}")
            print(f"  Threat Rate:        {threat_rate:6.2f}%   |  Safe Rate: {safe_rate:6.2f}%")
            print(f"  Read/Unread:        {self.metrics['read_emails']:4d}/{self.metrics['unread_emails']:4d}")
            print(f"  With Attachments:   {self.metrics['emails_with_attachments']:6d}   |  With URLs: {self.metrics['emails_with_urls']:6d}")
            print(f"  Reply Emails:       {self.metrics['reply_emails']:6d}")
            
            # Top domains
            if self.metrics['domain_distribution']:
                top_domains = sorted(self.metrics['domain_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
                print("\n  Top Domains:")
                for domain, count in top_domains:
                    print(f"    • {domain}: {count:4d} emails")
            
            # Attachment types
            if self.metrics['attachment_types']:
                print("\n  Attachment Types:")
                top_types = sorted(self.metrics['attachment_types'].items(), key=lambda x: x[1], reverse=True)[:3]
                for ftype, count in top_types:
                    print(f"    • {ftype}: {count:4d} files")
            
            print("\n" + "=" * 70)
            print()
        
        except Exception as e:
            logger.warning(f"Error printing dashboard: {e}")
    
    def is_trusted_sender(self, from_addr: str) -> bool:
        """
        Check if sender is in trusted whitelist.
        Bypasses ML analysis entirely for known safe senders.
        """
        if not from_addr:
            return False
        
        from_lower = from_addr.lower()
        for trusted_domain in TRUSTED_SENDER_DOMAINS:
            if trusted_domain.lower() in from_lower or from_addr.lower() == trusted_domain.lower():
                return True
        return False
    
    def get_whitelist_bypass_verdict(self) -> Dict:
        """
        Return a hardcoded SAFE verdict for whitelisted senders.
        Bypasses all ML streams.
        """
        return {
            'final_verdict': 'SAFE',
            'final_confidence': 0.0,
            'risk_score': 0.0,
            'metadata': {
                'has_urls': False,
                'has_attachments': False,
                'whitelisted': True
            },
            'stream_breakdown': {
                'stream_a': {'verdict': 'SAFE', 'confidence': 0.0},
                'stream_b': {'verdict': 'SAFE', 'confidence': 0.0},
                'stream_c': {'verdict': 'SAFE', 'confidence': 0.0}
            },
            'explanation': '[WHITELIST] Recognized trusted sender. All streams bypassed.'
        }
    
    def cleanup_orphaned_files(self):
        """Periodically clean up any orphaned attachment files in temp directory."""
        try:
            if not self.temp_dir.exists():
                return
            
            orphaned_count = 0
            for file_path in self.temp_dir.glob('*'):
                if file_path.is_file():
                    try:
                        # Delete any file in temp directory
                        file_path.unlink(missing_ok=True)
                        orphaned_count += 1
                        logger.debug(f"[CLEANUP] Orphaned: Deleted {file_path.name}")
                    except Exception as e:
                        logger.debug(f"[CLEANUP] Orphaned: Could not delete {file_path.name}: {e}")
            
            if orphaned_count > 0:
                logger.info(f"[CLEANUP] Orphaned file removal: Deleted {orphaned_count} file(s)")
        
        except Exception as e:
            logger.warning(f"Error during orphaned file cleanup: {e}")
    
    def connect(self) -> bool:
        """
        Connect to Gmail IMAP server and initialize SentinelCore.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to {IMAP_SERVER}:{IMAP_PORT}...")
            self.imap_server = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
            self.imap_server.login(self.email_address, self.app_password)
            logger.info(f"[✓] Connected and authenticated as {self.email_address}")
            
            # Initialize SentinelCore for analysis
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from sentinel_core import SentinelCore
                self.sentinel = SentinelCore()
                logger.info("[✓] Initialized SentinelCore for 3-stream analysis")
            except Exception as e:
                logger.error(f"Failed to initialize SentinelCore: {e}")
                return False
            
            return True
        except imaplib.IMAP4.error as e:
            logger.error(f"[✗] IMAP authentication failed: {e}")
            logger.error("Verify: email address and 16-character App Password are correct")
            return False
        except Exception as e:
            logger.error(f"[✗] Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Gracefully disconnect from IMAP server and save state."""
        try:
            if self.imap_server:
                try:
                    self.imap_server.close()
                except:
                    pass
                try:
                    self.imap_server.logout()
                except:
                    pass
                self.imap_server = None
                logger.info("[✓] Disconnected from IMAP server")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
        finally:
            self._save_processed_uids()
    
    def print_shield_report(self, email_info: Dict, analysis: Dict):
        """Print formatted Shield Report for threat analysis."""
        
        # Extract verdicts
        stream_a = analysis.get('stream_breakdown', {}).get('stream_a', {})
        stream_b = analysis.get('stream_breakdown', {}).get('stream_b', {})
        stream_c = analysis.get('stream_breakdown', {}).get('stream_c', {})
        
        final_verdict = analysis.get('final_verdict', 'UNKNOWN')
        final_confidence = analysis.get('final_confidence', 0.0)
        
        # Color codes for terminal
        BOLD = '\033[1m'
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        RESET = '\033[0m'
        
        # Determine shield status color
        if final_verdict == 'PHISHING':
            shield_color = RED
            shield_status = "🛑 THREAT DETECTED"
        elif final_verdict == 'SAFE':
            shield_color = GREEN
            shield_status = "✓ SAFE"
        else:
            shield_color = YELLOW
            shield_status = "⚠ SUSPICIOUS"
        
        # Print report
        print()
        print(f"{BOLD}{BLUE}{'='*70}{RESET}")
        print(f"{BOLD}{BLUE}[!] LIVE SENTINEL SHIELD - THREAT SCAN REPORT{RESET}")
        print(f"{BOLD}{BLUE}{'='*70}{RESET}")
        print()
        
        # Email details
        print(f"{BOLD}Email Details:{RESET}")
        print(f"  From:    {email_info.get('from', 'Unknown')}")
        print(f"  Subject: {email_info.get('subject', '(No Subject)')}")
        print(f"  UID:     {email_info.get('uid', 'Unknown')}")
        print(f"  Status:  {email_info.get('read_status', 'UNKNOWN')}")
        print()
        
        # Stream A: Text Analysis
        stream_a_verdict = stream_a.get('verdict', 'UNKNOWN')
        stream_a_confidence = stream_a.get('confidence', 0.0)
        stream_a_color = GREEN if stream_a_verdict == 'SAFE' else (RED if stream_a_verdict == 'PHISHING' else YELLOW)
        
        print(f"{BOLD}Stream A (Text Classification):{RESET}")
        print(f"  {stream_a_color}[{stream_a_verdict}] Confidence: {stream_a_confidence:.2%}{RESET}")
        print()
        
        # Stream B: URL Analysis (ALWAYS SHOW)
        stream_b_verdict = stream_b.get('verdict', 'UNKNOWN') if stream_b else 'NO_URLS'
        stream_b_confidence = stream_b.get('confidence', 0.0) if stream_b else 0.0
        stream_b_color = GREEN if stream_b_verdict == 'SAFE' else (RED if stream_b_verdict == 'PHISHING' else YELLOW)
        
        print(f"{BOLD}Stream B (URL Reputation):{RESET}")
        if stream_b:
            print(f"  {stream_b_color}[{stream_b_verdict}] Confidence: {stream_b_confidence:.2%}{RESET}")
        else:
            print(f"  ⚪ [NO_URLS] No URLs extracted from email")
        print()
        
        # Stream C: Attachment Analysis (ALWAYS SHOW)
        stream_c_verdict = stream_c.get('verdict', 'UNKNOWN') if stream_c else 'NO_ATTACHMENTS'
        stream_c_confidence = stream_c.get('confidence', 0.0) if stream_c else 0.0
        stream_c_color = GREEN if stream_c_verdict == 'SAFE' else (RED if stream_c_verdict == 'PHISHING' else YELLOW)
        
        print(f"{BOLD}Stream C (Attachment Analysis):{RESET}")
        if stream_c:
            print(f"  {stream_c_color}[{stream_c_verdict}] Confidence: {stream_c_confidence:.2%}{RESET}")
        else:
            print(f"  ⚪ [NO_ATTACHMENTS] No attachments to analyze")
        print()
        
        # Final Shield Status
        print(f"{BOLD}{BLUE}{'─'*70}{RESET}")
        print(f"{BOLD}{shield_color}FINAL VERDICT: {shield_status}{RESET}")
        print(f"{BOLD}Risk Score: {final_confidence:.2%}{RESET}")
        print(f"{BOLD}{BLUE}{'='*70}{RESET}")
        print()
        
        # XAI Explanation
        xai = analysis.get('xai_explanation', {})
        if xai.get('override_triggered'):
            print(f"{BOLD}{YELLOW}[!] OVERRIDE TRIGGERED: {xai.get('override_reason', 'Unknown')}{RESET}")
            print()
        
        if xai.get('human_readable'):
            print(f"{BOLD}Explanation:{RESET}")
            print(f"  {xai.get('human_readable')}")
            print()
    
    def monitor_new_emails(self, check_unread_only: bool = False):
        """
        Monitor for new emails and analyze with SentinelCore.
        
        ⚠️ IMPORTANT FIX: check_unread_only now defaults to FALSE
        This ensures the monitor processes ALL emails in INBOX, not just unread ones.
        Previously read emails are still tracked in processed_uids to avoid duplicates.
        
        Args:
            check_unread_only: If True, only check UNSEEN emails. If False, check ALL emails.
        """
        try:
            status, mailbox_data = self.imap_server.select("INBOX")
            if status != "OK":
                logger.error("Failed to select INBOX")
                return
            
            # Search for emails: ALL = includes read + unread
            # UNSEEN = only unread (old behavior)
            search_criteria = "UNSEEN" if check_unread_only else "ALL"
            status, msg_ids = self.imap_server.search(None, search_criteria)
            if status != "OK":
                logger.error("Failed to search emails")
                return
            
            msg_id_list = msg_ids[0].split()
            if not msg_id_list:
                if check_unread_only:
                    logger.debug("No new unread emails")
                return
            
            logger.info(f"Found {len(msg_id_list)} emails to check")
            
            # Track emails found
            emails_found = []
            
            # Process each email
            for msg_id in msg_id_list:
                attachment_paths = []
                try:
                    status, msg_data = self.imap_server.fetch(msg_id, "(RFC822 UID)")
                    if status != "OK":
                        continue
                    
                    # Extract UID
                    uid_match = re.search(rb'UID (\d+)', msg_data[0][0])
                    uid = uid_match.group(1).decode() if uid_match else msg_id.decode()
                    
                    # Skip if already processed
                    if uid in self.processed_uids:
                        logger.debug(f"[PROCESS] Skipping already-processed UID {uid}")
                        continue
                    
                    # Parse message
                    msg = email.message_from_bytes(msg_data[0][1])
                    
                    # Extract email information
                    from_addr = msg.get('From', 'Unknown')
                    subject = msg.get('Subject', '(No Subject)')
                    body = self.extract_email_body(msg)
                    
                    # Extract flags to determine read/unread status
                    flags_match = re.search(rb'FLAGS \(([^)]*)\)', msg_data[0][0])
                    flags_str = flags_match.group(1).decode() if flags_match else ""
                    is_read = '\\Seen' in flags_str
                    read_status = "READ" if is_read else "UNREAD"
                    logger.info(f"[EMAIL] UID {uid}: [{read_status}]")
                    
                    # Extract URLs from raw message (BEFORE body is stripped)
                    # This captures both HTML href attributes and plaintext URLs
                    urls = self.extract_urls_from_message(msg)
                    
                    # Extract attachments
                    attachments = self.extract_attachments(msg)
                    attachment_paths = [a['file_path'] for a in attachments]
                    
                    # Log attachment extraction
                    if attachment_paths:
                        logger.info(f"[EXTRACT] Found {len(attachment_paths)} attachment(s) to analyze")
                    
                    # Prepare email content
                    email_content = f"Subject: {subject}\n\n{body}"
                    
                    # CRITICAL SECURITY FIX: Use proper domain validation (not substring matching!)
                    if self._is_trusted_sender(from_addr):
                        logger.info(f"[WHITELIST] Trusted sender recognized: {from_addr}. Bypassing ML analysis.")
                        # Skip to next email - no processing for whitelisted senders
                        self.processed_uids.add(uid)
                        self._save_processed_uids()
                        continue
                    
                    # Run 3-stream analysis only for non-whitelisted senders
                    logger.info(f"\n[ANALYZE] Email from {from_addr}: {subject}")
                    logger.info(f"  Analyzing with: Text={'✓' if email_content else '✗'} | URLs={'✓' if urls else '✗'} ({len(urls) if urls else 0}) | Attachments={'✓' if attachment_paths else '✗'} ({len(attachment_paths) if attachment_paths else 0})")
                    
                    analysis = self.analyze_with_timeout(
                        email_content=email_content,
                        urls=urls if urls else None,
                        attachment_paths=attachment_paths if attachment_paths else None,
                        timeout_secs=60  # 60-second timeout for analysis
                    )
                    
                    # Log stream results
                    stream_a = analysis.get('stream_breakdown', {}).get('stream_a', {})
                    stream_b = analysis.get('stream_breakdown', {}).get('stream_b', {})
                    stream_c = analysis.get('stream_breakdown', {}).get('stream_c', {})
                    
                    if stream_a:
                        logger.info(f"  [STREAM A] {stream_a.get('verdict', '?')}: {stream_a.get('confidence', 0):.0%}")
                    if stream_b:
                        logger.info(f"  [STREAM B] {stream_b.get('verdict', '?')}: {stream_b.get('confidence', 0):.0%}")
                    if stream_c:
                        logger.info(f"  [STREAM C] {stream_c.get('verdict', '?')}: {stream_c.get('confidence', 0):.0%}")
                    
                    final_verdict = analysis.get('final_verdict', 'UNKNOWN')
                    logger.info(f"  [FINAL] {final_verdict}: {analysis.get('final_confidence', 0):.0%}")
                    
                    # Prepare email info
                    email_info = {
                        'uid': uid,
                        'from': from_addr,
                        'subject': subject,
                        'urls': urls,
                        'attachments': attachments,
                        'read_status': read_status
                    }
                    
                    # Print Shield Report
                    self.print_shield_report(email_info, analysis)
                    
                    # ========================================================
                    # CRITICAL: Update metrics and persist IMMEDIATELY
                    # This ensures dashboard sees real-time KPI updates
                    # ========================================================
                    self.update_metrics(analysis, email_info, read_status)
                    # Note: save_metrics() is called inside update_metrics()
                    # This line is here for clarity and redundancy is safe
                    self.save_metrics()
                    
                    # Print dashboard every 10 emails
                    if self.metrics['total_analyzed'] % 10 == 0:
                        self.print_metrics_dashboard()
                    
                    # Mark as processed
                    self.processed_uids.add(uid)
                    self._save_processed_uids()
                    
                    # Mark as read
                    try:
                        self.imap_server.store(msg_id, '+FLAGS', '\\Seen')
                    except:
                        pass
                    
                except Exception as e:
                    logger.error(f"[PROCESS] Error processing email {msg_id}: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                finally:
                    # CRITICAL: Aggressive file cleanup - MUST complete before next email
                    for file_path in attachment_paths:
                        try:
                            file_path_obj = Path(file_path)
                            if file_path_obj.exists():
                                # Retry mechanism for locked files
                                max_retries = 3
                                for attempt in range(max_retries):
                                    try:
                                        # Force close any handles
                                        os.chmod(file_path, 0o777)
                                        # Attempt deletion
                                        file_path_obj.unlink(missing_ok=True)
                                        logger.info(f"[CLEANUP] Deleted attachment: {file_path}")
                                        
                                        # Verify deletion
                                        if file_path_obj.exists():
                                            logger.error(f"[CLEANUP] VERIFICATION FAILED - File still exists: {file_path}")
                                        else:
                                            logger.debug(f"[CLEANUP] Verified: {file_path} successfully deleted")
                                        break  # Success - exit retry loop
                                    except PermissionError:
                                        if attempt < max_retries - 1:
                                            import time
                                            time.sleep(0.5)  # Wait before retry
                                            continue
                                        else:
                                            raise
                            else:
                                logger.warning(f"[CLEANUP] File not found for deletion: {file_path}")
                        except Exception as e:
                            logger.error(f"[CLEANUP] CRITICAL - Failed to delete {file_path}: {type(e).__name__}: {e}")
        
        except Exception as e:
            logger.error(f"Error monitoring emails: {e}")
    
    def perform_initial_audit(self):
        """
        Historical audit mode: Scan last 50 emails on startup.
        
        - Fetch last 50 emails from INBOX
        - For each email, check if UID exists in .sentinel_state.json
        - IF NEW: Analyze with 3-stream fusion and print report
        - IF KNOWN: Skip silently
        - Seamlessly transition to live monitoring
        """
        try:
            logger.info("\n[INFO] Auditing last 50 emails...")
            
            status, mailbox_data = self.imap_server.select("INBOX")
            if status != "OK":
                logger.error("Failed to select INBOX for audit")
                return
            
            # Get all message IDs
            status, msg_ids = self.imap_server.search(None, 'ALL')
            if status != "OK":
                logger.error("Failed to search emails for audit")
                return
            
            msg_id_list = msg_ids[0].split()
            if not msg_id_list:
                logger.info("No emails in INBOX")
                return
            
            # Get all emails (increased from 50 to catch all unprocessed emails including Obite)
            # This ensures emails that are already read but not yet analyzed are captured
            audit_list = msg_id_list[-200:] if len(msg_id_list) > 200 else msg_id_list
            logger.info(f"Found {len(msg_id_list)} total emails, auditing last {len(audit_list)}...")
            
            threat_count = 0
            processed_count = 0
            skipped_count = 0
            threat_details = []  # Store threat details for summary
            
            # Process each email in audit list
            for msg_id in audit_list:
                attachment_paths = []
                try:
                    status, msg_data = self.imap_server.fetch(msg_id, "(RFC822 UID)")
                    if status != "OK":
                        continue
                    
                    # Extract UID
                    uid_match = re.search(rb'UID (\d+)', msg_data[0][0])
                    uid = uid_match.group(1).decode() if uid_match else msg_id.decode()
                    
                    # Check if already processed
                    if uid in self.processed_uids:
                        skipped_count += 1
                        continue
                    
                    # Parse message
                    msg = email.message_from_bytes(msg_data[0][1])
                    
                    # Extract email information
                    from_addr = msg.get('From', 'Unknown')
                    subject = msg.get('Subject', '(No Subject)')
                    body = self.extract_email_body(msg)
                    
                    # Extract flags to determine read/unread status
                    flags_match = re.search(rb'FLAGS \(([^)]*)\)', msg_data[0][0])
                    flags_str = flags_match.group(1).decode() if flags_match else ""
                    is_read = '\\Seen' in flags_str
                    read_status = "READ" if is_read else "UNREAD"
                    
                    # Extract URLs from raw message (BEFORE body is stripped)
                    # This captures both HTML href attributes and plaintext URLs
                    urls = self.extract_urls_from_message(msg)
                    attachments = self.extract_attachments(msg)
                    attachment_paths = [a['file_path'] for a in attachments]
                    
                    # Prepare email content
                    email_content = f"Subject: {subject}\n\n{body}"
                    
                    # CRITICAL SECURITY FIX: Use proper domain validation (not substring matching!)
                    if self._is_trusted_sender(from_addr):
                        logger.info(f"[WHITELIST] Audit: Trusted sender recognized: {from_addr}. Bypassing ML analysis.")
                        # Skip to next email - no processing for whitelisted senders
                        continue
                    
                    # Run 3-stream analysis only for non-whitelisted senders
                    analysis = self.analyze_with_timeout(
                        email_content=email_content,
                        urls=urls if urls else None,
                        attachment_paths=attachment_paths if attachment_paths else None,
                        timeout_secs=60  # 60-second timeout for analysis
                    )
                    
                    # Prepare email info
                    email_info = {
                        'uid': uid,
                        'from': from_addr,
                        'subject': subject,
                        'urls': urls,
                        'attachments': attachments,
                        'read_status': read_status
                    }
                    
                    # Count threats and store details for summary
                    if analysis.get('final_verdict') == 'PHISHING':
                        threat_count += 1
                        logger.info(f"[THREAT] {from_addr}: {subject}")
                        self.print_shield_report(email_info, analysis)
                        # Store threat for summary
                        threat_details.append({
                            'from': from_addr,
                            'subject': subject,
                            'confidence': analysis.get('final_confidence', 0),
                            'explanation': analysis.get('explanation', 'No explanation')
                        })
                    
                    # Update metrics for all analyzed emails
                    self.update_metrics(analysis, email_info, read_status)
                    self.save_metrics()
                    
                    # Print dashboard every 10 emails
                    if self.metrics['total_analyzed'] % 10 == 0:
                        self.print_metrics_dashboard()
                    
                    # Mark as processed
                    self.processed_uids.add(uid)
                    self._save_processed_uids()
                    processed_count += 1
                    
                    # Mark as read in IMAP
                    try:
                        self.imap_server.store(msg_id, '+FLAGS', '\\Seen')
                    except:
                        pass
                    
                except Exception as e:
                    logger.warning(f"Error processing audit email {msg_id}: {e}")
                    continue
                
                finally:
                    # CRITICAL: Clean up attachment files from audit
                    for file_path in attachment_paths:
                        try:
                            file_path_obj = Path(file_path)
                            if file_path_obj.exists():
                                # Retry mechanism for locked files
                                max_retries = 3
                                for attempt in range(max_retries):
                                    try:
                                        # Force close any handles
                                        os.chmod(file_path, 0o777)
                                        # Attempt deletion
                                        file_path_obj.unlink(missing_ok=True)
                                        logger.debug(f"[CLEANUP] Audit: Deleted attachment: {file_path}")
                                        break  # Success - exit retry loop
                                    except PermissionError:
                                        if attempt < max_retries - 1:
                                            import time
                                            time.sleep(0.5)  # Wait before retry
                                            continue
                                        else:
                                            raise
                        except Exception as e:
                            logger.warning(f"[CLEANUP] Audit: Failed to delete {file_path}: {type(e).__name__}: {e}")
            
            # Audit summary with threat details
            logger.info(f"\n[INFO] Audit complete: {processed_count} new emails analyzed, {threat_count} threats detected, {skipped_count} already known")
            if threat_details:
                logger.info("\n[AUDIT THREATS SUMMARY]")
                for i, threat in enumerate(threat_details, 1):
                    logger.info(f"  {i}. From: {threat['from']}")
                    logger.info(f"     Subject: {threat['subject']}")
                    logger.info(f"     Confidence: {threat['confidence']:.2%}")
                    logger.info(f"     Explanation: {threat['explanation']}")
            else:
                logger.info("[INFO] History clean. Sentinel Shield is now LIVE.")
            
        except Exception as e:
            logger.error(f"Error during historical audit: {e}")
    
    def extract_email_body(self, msg: Message) -> str:
        """
        Extract plaintext body from email message.
        Handles multipart emails, returns first text part found.
        """
        try:
            body = ""
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    charset = part.get_content_charset() or 'utf-8'
                    try:
                        body = part.get_payload(decode=True).decode(charset, errors='ignore')
                        return body
                    except:
                        pass
            return body if body else msg.get_payload()
        except Exception as e:
            logger.warning(f"Error extracting email body: {e}")
            return ""
    
    def extract_urls_from_message(self, msg: Message) -> List[str]:
        """
        Extract URLs from raw email message (both HTML href and plaintext).
        CRITICAL: Extracts from RAW payload BEFORE HTML is stripped for Stream A.
        
        Captures:
        1. HTML href attributes: href="https://..." or href='https://...'
        2. Plain text URLs: https://... or http://...
        3. Defanged URLs: hxxp:// or hxxps://
        
        Returns:
            List of unique URLs found in message
        """
        urls = set()  # Use set to avoid duplicates
        
        try:
            # Extract from all message parts
            for part in msg.walk():
                content_type = part.get_content_type()
                
                # Process both HTML and plaintext
                if content_type in ('text/html', 'text/plain'):
                    try:
                        charset = part.get_content_charset() or 'utf-8'
                        payload = part.get_payload(decode=True).decode(charset, errors='ignore')
                        
                        # Extract from HTML href attributes
                        href_pattern = r'href\s*=\s*["\']([^"\']+)["\']'
                        href_urls = re.findall(href_pattern, payload, re.IGNORECASE)
                        urls.update(href_urls)
                        
                        # Extract plain text URLs (http/https)
                        url_pattern = r'https?://[^\s<>"{}|\\^\[\]`]+'
                        plain_urls = re.findall(url_pattern, payload)
                        urls.update(plain_urls)
                        
                        # Extract defanged URLs (hxxp:// → http://)
                        defanged_pattern = r'hxxps?://[^\s<>"{}|\\^\[\]`]+'
                        defanged_urls = re.findall(defanged_pattern, payload, re.IGNORECASE)
                        for defanged in defanged_urls:
                            # Refang: hxxps?:// → https?://
                            refanged = defanged.replace('hxxp', 'http')
                            urls.add(refanged)
                        
                    except Exception as e:
                        logger.debug(f"Error extracting URLs from {content_type} part: {e}")
                        continue
            
            # Filter out common false positives
            filtered_urls = []
            for url in urls:
                # Skip non-URL patterns that look like URLs
                if url.lower().startswith(('http://', 'https://')):
                    # Remove trailing punctuation that's likely not part of URL
                    url = re.sub(r'[,;.!?\)]+$', '', url)
                    if url and len(url) > 10:  # Minimum reasonable URL length
                        filtered_urls.append(url)
            
            logger.debug(f"[URL EXTRACT] Found {len(filtered_urls)} unique URLs from message")
            return list(set(filtered_urls))  # Final dedup
            
        except Exception as e:
            logger.warning(f"Error extracting URLs from message: {e}")
            return []
    
    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from email text using regex.
        
        IMPORTANT: This is a fallback for text that's already been processed.
        For complete URL extraction, use extract_urls_from_message() instead.
        
        Extracts:
        - Plain text URLs: https://... or http://...
        - Defanged URLs: hxxp:// or hxxps://
        """
        try:
            urls = set()
            
            # Extract plain text URLs (http/https)
            url_pattern = r'https?://[^\s<>"{}|\\^\[\]`]+'
            plain_urls = re.findall(url_pattern, text)
            urls.update(plain_urls)
            
            # Extract defanged URLs (hxxp:// → http://)
            defanged_pattern = r'hxxps?://[^\s<>"{}|\\^\[\]`]+'
            defanged_urls = re.findall(defanged_pattern, text, re.IGNORECASE)
            for defanged in defanged_urls:
                # Refang: hxxps?:// → https?://
                refanged = defanged.replace('hxxp', 'http')
                urls.add(refanged)
            
            return list(set(urls))  # Remove duplicates
        except Exception as e:
            logger.warning(f"Error extracting URLs: {e}")
            return []
    
    def extract_attachments(self, msg: Message, msg_id: Optional[bytes] = None) -> List[Dict]:
        """
        Extract attachments from email message.
        
        CRITICAL IMPLEMENTATION:
        1. Uses msg.walk() to traverse multipart structure
        2. Skips multipart containers
        3. Handles None filenames safely
        4. Decodes Base64 payload with decode=True
        
        Args:
            msg: Parsed email message
            msg_id: Optional IMAP message ID (for backwards compatibility)
            
        Returns:
            List of dicts with file info and path
        """
        attachments = []
        
        try:
            # ================================================================
            # CRITICAL #1: Walking Logic - Traverse multipart message structure
            # ================================================================
            for part in msg.walk():
                # Skip multipart containers (they have no actual content)
                if part.get_content_maintype() == 'multipart':
                    continue
                
                # ============================================================
                # CRITICAL #2: Filename Sanitization - Handle None safely
                # ============================================================
                filename = None
                try:
                    filename = part.get_filename()
                except Exception as e:
                    logger.warning(f"Error getting filename from part: {e}")
                    continue
                
                # Skip parts without filenames (e.g., email body)
                if not filename:
                    continue
                
                # ============================================================
                # CRITICAL #3: Byte-Stream Integrity - Decode Base64 payload
                # ============================================================
                try:
                    # Get payload with Base64 decoding enabled
                    payload = part.get_payload(decode=True)
                    
                    if payload is None:
                        logger.warning(f"Empty payload for {filename}")
                        continue
                    
                    # ========================================================
                    # PHASE 8: CHECK ATTACHMENT SIZE AGAINST CONFIG LIMIT
                    # ========================================================
                    file_size_mb = len(payload) / (1024 * 1024)
                    if file_size_mb > MAX_ATTACHMENT_SIZE_MB:
                        logger.info(f"[SKIPPED] Attachment '{filename}' is {file_size_mb:.2f}MB (exceeds {MAX_ATTACHMENT_SIZE_MB}MB limit). Stream C analysis will be bypassed.")
                        continue
                    
                    # Sanitize filename for safe file operations
                    safe_filename = self._sanitize_filename(filename)
                    file_path = self.temp_dir / safe_filename
                    
                    # Write to temporary file
                    with open(file_path, 'wb', encoding=None) as f:
                        f.write(payload)
                    
                    attachments.append({
                        'original_filename': filename,
                        'safe_filename': safe_filename,
                        'file_path': str(file_path),
                        'size': len(payload),
                        'content_type': part.get_content_type()
                    })
                    
                    logger.debug(f"Extracted: {filename} ({len(payload)} bytes)")
                    
                except Exception as e:
                    logger.error(f"Failed to extract {filename}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
        
        return attachments
    
    @staticmethod
    def _sanitize_filename(filename: str, max_len: int = 255) -> str:
        """
        Sanitize filename for safe filesystem operations.
        
        Args:
            filename: Original filename from email
            max_len: Maximum filename length
            
        Returns:
            Safe filename
        """
        import re
        
        # CRITICAL FIX: Strip testing/defanged/malware suffixes FIRST
        # These are added by email scanners/sandbox analysis tools
        malware_suffixes = ['.malware', '.vir', '.test', '.defanged', '.sandbox', '.quarantine']
        for suffix in malware_suffixes:
            if filename.lower().endswith(suffix):
                filename = filename[:-len(suffix)]
                logger.debug(f"[SANITIZE] Stripped '{suffix}' from filename -> '{filename}'")
                break  # Only strip the first matching suffix
        
        # Remove path separators and null bytes
        safe = filename.replace('/', '_').replace('\\', '_').replace('\x00', '')
        
        # Remove control characters
        safe = ''.join(c for c in safe if ord(c) >= 32)
        
        # Replace problematic characters
        safe = re.sub(r'[<>:"|?*]', '_', safe)
        
        # Limit length
        if len(safe) > max_len:
            base, ext = safe.rsplit('.', 1) if '.' in safe else (safe, '')
            safe = base[:max_len-len(ext)-1] + '.' + ext if ext else safe[:max_len]
        
        return safe or "unnamed_attachment"
    
    def run_idle_loop(self):
        """
        Run persistent monitoring loop with explicit 30-second polling.
        
        SIMPLIFIED APPROACH:
        1. Every 30 seconds, refresh INBOX state
        2. Search for UNSEEN emails
        3. Analyze any new emails
        4. Comprehensive error handling with logging
        5. Auto-reconnect on failure
        """
        reconnect_delay = 5
        poll_interval = 30  # Explicit 30-second polling
        poll_counter = 0
        
        logger.info("[MONITOR] Entering explicit 30-second polling loop")
        
        while True:
            try:
                # Ensure connected
                if not self.imap_server:
                    logger.info("[RECONNECT] IMAP connection lost, attempting reconnection...")
                    if not self.connect():
                        logger.warning(f"[RECONNECT] Failed. Retrying in {reconnect_delay}s...")
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, 300)  # Exponential backoff, max 5min
                        continue
                    reconnect_delay = 5  # Reset delay on successful connection
                    logger.info("[RECONNECT] Connection restored")
                
                # EXPLICIT 30-SECOND POLLING LOOP
                while True:
                    try:
                        # Refresh INBOX state EVERY 30 seconds
                        logger.debug("[POLL] Refreshing INBOX state...")
                        status = self.imap_server.select("INBOX")[0]
                        if status != "OK":
                            logger.error("[POLL] Failed to select INBOX")
                            self.imap_server = None
                            break  # Break inner loop, reconnect
                        
                        # Search for UNSEEN emails
                        logger.debug("[POLL] Searching for UNSEEN emails...")
                        status, msg_ids = self.imap_server.search(None, "UNSEEN")
                        if status != "OK":
                            logger.error("[POLL] Failed to search emails")
                            self.imap_server = None
                            break  # Break inner loop, reconnect
                        
                        # NEW: Get current total inbox count (for live deletion tracking)
                        logger.debug("[POLL] Counting total emails in INBOX...")
                        status, all_msg_ids = self.imap_server.search(None, "ALL")
                        if status == "OK" and all_msg_ids[0]:
                            current_inbox_count = len(all_msg_ids[0].split())
                            self.metrics['current_inbox_count'] = current_inbox_count
                            logger.debug(f"[POLL] Current inbox total: {current_inbox_count} emails")
                            # Persist the updated count immediately
                            self.save_metrics()
                        
                        # Check and analyze any new emails
                        if msg_ids[0]:
                            email_count = len(msg_ids[0].split())
                            logger.info(f"[POLL] Found {email_count} new unseen emails")
                            logger.info("[MONITOR] ═══════════════════════════════════════════")
                            logger.info("[MONITOR] EMAILS FOUND - BEGINNING ANALYSIS")
                            logger.info("[MONITOR] ═══════════════════════════════════════════")
                            self.monitor_new_emails(check_unread_only=True)
                            logger.info("[MONITOR] ═══════════════════════════════════════════")
                        else:
                            logger.debug("[POLL] No new emails at this time")
                        
                        # HEARTBEAT: Always save metrics to keep dashboard alive
                        # This ensures the timestamp is updated every 30 seconds
                        self.save_metrics()
                        logger.debug(f"[HEARTBEAT] Metrics updated (Active: {datetime.now().isoformat()})")
                        
                        # Periodic orphaned file cleanup (every 10 polls = every 5 minutes)
                        poll_counter += 1
                        if poll_counter % 10 == 0:
                            self.cleanup_orphaned_files()
                        
                        # Wait 30 seconds before next poll
                        logger.debug(f"[POLL] Waiting {poll_interval}s before next check...")
                        time.sleep(poll_interval)
                    
                    except (imaplib.IMAP4.abort, socket.timeout, socket.error) as e:
                        # Connection errors - break to reconnect
                        logger.warning(f"[POLL] Connection error ({type(e).__name__}): {e}")
                        self.imap_server = None
                        break  # Break inner loop, reconnect
                    
                    except Exception as e:
                        # ANY other error - log and continue polling
                        logger.error(f"[POLL] Unexpected error: {type(e).__name__}: {e}")
                        logger.info(f"[POLL] Continuing in {poll_interval}s...")
                        time.sleep(poll_interval)
                        continue
            
            except KeyboardInterrupt:
                logger.info("\n[!] Shutting down Sentinel Shield...")
                self.disconnect()
                logger.info("[SHUTDOWN] State saved. Goodbye!")
                break
            
            except Exception as e:
                logger.error(f"[ERROR] Unexpected error in main loop: {type(e).__name__}: {e}")
                self.imap_server = None
                logger.info(f"[RECONNECT] Will retry in {reconnect_delay}s...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 300)


def main():
    """
    Main entry point - Start LiveSentinelShield Always-On monitoring.
    """
    print("\n")
    print("=" * 60)
    print("  🛡️  LIVE SENTINEL SHIELD  🛡️")
    print("  Always-On Phishing Detection Monitor")
    print("=" * 60)
    print()
    
    try:
        # Load credentials from .env
        sys.path.insert(0, str(Path(__file__).parent))
        from sentinel_core import SentinelCore
        
        print("[...] Loading Gmail credentials from .env")
        email_addr, app_pass = SentinelCore.load_gmail_credentials()
        
        if not email_addr or not app_pass:
            print("[✗] Failed to load credentials. Ensure .env file is configured.")
            return 1
        
        print(f"[✓] Credentials loaded for {email_addr}")
        
        # Create and start shield
        print("[...] Initializing Sentinel Shield")
        shield = LiveSentinelShield(email_addr, app_pass)
        
        # Test connection first
        print("[...] Testing IMAP connection to imap.gmail.com:993")
        if not shield.connect():
            print("[✗] Failed to connect to Gmail. Check credentials and try again.")
            return 1
        
        print("[✓] IMAP connection successful")
        
        # Initialize SentinelCore
        print("[...] Loading ML models (DistilBERT, XGBoost, Stream C)")
        if shield.sentinel is None:
            shield.init_sentinel()
        print("[✓] ML models loaded successfully")
        
        # Print startup banner
        print("\n" + "="*70)
        print("          ✓ SENTINEL SHIELD READY FOR DEPLOYMENT")
        print("="*70)
        print(f"\n  📧 Email Account:        {email_addr}")
        print(f"  🔍 Detection Streams:    Stream A (Text) + Stream B (URL) + Stream C (Files)")
        print(f"  📊 Metrics File:         .sentinel_metrics.json")
        print(f"  📝 Log File:             .sentinel_shield.log (auto-rotating 5MB)")
        print(f"  💾 Temp Directory:       {shield.temp_dir}")
        print(f"  ⏱️  Polling Interval:     30 seconds (explicit IMAP)")
        print("\n" + "="*70)
        print(f"State File:    {shield.state_file}")
        print(f"Tracked UIDs:  {len(shield.processed_uids)}")
        print("="*70)
        print()
        print("MONITORING MODE: IMAP IDLE (Real-time server-push notifications)")
        print("Press Ctrl+C to stop and save state")
        print()
        print("="*70)
        print()
        
        # Perform historical audit first
        try:
            shield.perform_initial_audit()
        except Exception as e:
            logger.error(f"Error during initial audit: {e}")
            shield.disconnect()
            return 1
        
        print()
        print("="*70)
        print("[!] ENTERING LIVE MONITORING MODE")
        print("="*70)
        print()
        
        # Run persistent monitoring loop
        shield.run_idle_loop()
        
        logger.info("\n[✓] Sentinel Shield shutdown complete")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n[!] Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
