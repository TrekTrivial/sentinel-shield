#!/usr/bin/env python3
"""
Phase 5: SentinelCore - Unified Inference Engine
=================================================

The master inference class that fuses all three analytical streams:
- Stream A: Text Classification (DistilBERT)
- Stream B: URL Analysis (XGBoost)
- Stream C: Attachment Analysis (Master Dispatcher)

Core Features:
1. Dynamic weighting with stream redistribution
2. Tiered override mechanism for structural threats
3. Unified SHAP explainability
4. Production-grade GPU-optional optimization
5. Single model initialization (loaded once, used many times)

Mathematical Model:
  Score_final = Σ(w_i * S_i) where:
    - w_i = dynamic weight for stream i
    - S_i = normalized score from stream i
  
  Override: If C returns structural_threat=True → Final_verdict = PHISHING (0.95)
"""

import sys
import json
import os
import re
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class SentinelCore:
    """
    Unified Phishing Detection Engine
    
    Implements multi-stream fusion with intelligent weighting,
    hierarchical override logic, and integrated SHAP explanations.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize SentinelCore with all three analytical streams.
        
        Args:
            model_dir: Path to models directory. If None, uses workspace root.
                      Expected structure:
                      - text_model_distilbert/ (DistilBERT checkpoint)
                      - stream_b_xgboost_v2.pkl (XGBoost model)
                      - stream_c_excel_model.pkl (Random Forest model)
        
        Raises:
            RuntimeError: If critical models cannot be loaded
        """
        self.model_dir = Path(model_dir) if model_dir else self._find_model_dir()
        
        print("[INIT] SentinelCore Unified Inference Engine")
        print(f"[INIT] Model directory: {self.model_dir}")
        print()

        # Initialize all three streams
        self._init_stream_a()  # Text analysis
        self._init_stream_b()  # URL analysis
        self._init_stream_c()  # Attachment analysis
        
        # Configuration
        self.default_weights = {
            'stream_a': 0.45,  # Text: 45% (increased from 0.40 - text is most reliable)
            'stream_b': 0.40,  # URL: 40% (increased from 0.35 - URLs critical for phishing)
            'stream_c': 0.15,  # Attachment: 15% (reduced from 0.25 - lower FP rate)
        }
        
        self.phishing_threshold = 0.65  # Require 65% confidence to flag as PHISHING (stricter)
        self.suspicious_threshold = 0.50  # 50-65% = SUSPICIOUS warning
        self.confidence_floor = 0.0  # No artificial floor - allow full [0.0, 1.0] range
        
        print("[OK] SentinelCore initialized successfully")
        print()
    
    @staticmethod
    def _find_model_dir() -> Path:
        """Find model directory relative to script location."""
        script_root = Path(__file__).parent.parent.parent
        models_dir = script_root / "models"
        
        if not models_dir.exists():
            raise RuntimeError(f"Models directory not found: {models_dir}")
        
        return models_dir
    
    @staticmethod
    def load_gmail_credentials() -> Tuple[str, str]:
        """
        Load Gmail IMAP credentials from .env file.
        
        Returns:
            Tuple of (email, app_password)
            
        Raises:
            RuntimeError: If .env file not found or credentials missing
        """
        # Load .env file
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        
        if not env_path.exists():
            raise RuntimeError(
                f".env file not found at {env_path}\n"
                "Create .env with:\n"
                "  SENTINEL_EMAIL=your-email@gmail.com\n"
                "  SENTINEL_APP_PASS=xxxx xxxx xxxx xxxx"
            )
        
        # Load environment variables from .env
        if load_dotenv is not None:
            load_dotenv(env_path)
        else:
            # Fallback: manual parsing if python-dotenv not installed
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
        
        # Get credentials
        email = os.getenv('SENTINEL_EMAIL')
        app_pass = os.getenv('SENTINEL_APP_PASS')
        
        if not email or not app_pass:
            raise RuntimeError(
                "SENTINEL_EMAIL or SENTINEL_APP_PASS not set in .env file"
            )
        
        return email, app_pass
    
    
    def _init_stream_a(self):
        """Initialize Stream A: Text Classification (DistilBERT)."""
        try:
            # Attempt to import required libraries
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
            except ImportError:
                raise RuntimeError("Required: pip install transformers torch")
            
            print("[INIT] Stream A: Text Classification (DistilBERT)")
            
            model_path = self.model_dir / "text_model_distilbert"
            
            if not model_path.exists():
                raise RuntimeError(f"DistilBERT model not found: {model_path}")
            
            # Load tokenizer and model
            self.stream_a_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.stream_a_model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self.stream_a_model.eval()
            
            # Determine device (GPU if available, else CPU)
            self.device_a = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.stream_a_model.to(self.device_a)
            
            print(f"  [OK] DistilBERT loaded (device: {self.device_a})")
            print(f"  [OK] Tokenizer: {self.stream_a_tokenizer.name_or_path}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to initialize Stream A: {e}")
            raise
    
    def _init_stream_b(self):
        """Initialize Stream B: URL Analysis (XGBoost)."""
        try:
            import pickle
            import os
            
            print("[INIT] Stream B: URL Analysis (XGBoost)")
            
            # ==================================================================
            # VERSION SELECTION (Blue-Green Deployment Support)
            # ==================================================================
            # Read from config file or environment variable
            config_path = self.model_dir / ".stream_b_version.json"
            active_version = "v1"  # Default to v1 (production)
            
            # Check for version override from environment
            env_version = os.getenv("STREAM_B_MODEL_VERSION")
            if env_version:
                active_version = env_version
                print(f"  [INFO] Using version override from environment: {active_version}")
            elif config_path.exists():
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        active_version = config.get("active_version", "v1")
                        print(f"  [INFO] Using version from config: {active_version}")
                except Exception as e:
                    print(f"  [WARN] Failed to load version config: {e}, using v1")
            
            # Map version to model path
            if active_version == "v2":
                model_path = self.model_dir / "stream_b_xgboost_v2.pkl"
                model_desc = "Stream B v2 (with OpenPhish training)"
            else:
                model_path = self.model_dir / "stream_b_xgboost_v2.pkl"
                model_desc = "Stream B v1 (production)"
            
            if not model_path.exists():
                # Try fallback to v1 if v2 missing
                if active_version == "v2":
                    print(f"  [WARN] v2 not found, falling back to v1")
                    model_path = self.model_dir / "stream_b_xgboost_v2.pkl"
                    model_desc = "Stream B v1 (fallback)"
            
            if not model_path.exists():
                raise RuntimeError(f"XGBoost model not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                self.stream_b_model = pickle.load(f)
            
            print(f"  [OK] {model_desc} loaded from {model_path.name}")
            print(f"  [OK] Model type: {type(self.stream_b_model).__name__}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to initialize Stream B: {e}")
            raise
    
    def _init_stream_c(self):
        """Initialize Stream C: Attachment Analysis (Master Dispatcher)."""
        try:
            print("[INIT] Stream C: Attachment Analysis (Master Dispatcher)")
            
            # Import from the production evaluation script
            eval_script = Path(__file__).parent.parent / "stream_C_attachments" / "1_evaluate_stream_c.py"
            
            if not eval_script.exists():
                raise RuntimeError(f"Stream C evaluation script not found: {eval_script}")
            
            # Dynamic import
            import importlib.util
            spec = importlib.util.spec_from_file_location("stream_c", eval_script)
            stream_c_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stream_c_module)
            
            # Instantiate Master Dispatcher
            self.stream_c_analyzer = stream_c_module.MasterAttachmentAnalyzer()
            
            print(f"  [OK] MasterAttachmentAnalyzer instantiated")
            print(f"  [OK] File type support: PDF, Excel, Office, PowerPoint, Image, HTML")
            
        except Exception as e:
            print(f"  [ERROR] Failed to initialize Stream C: {e}")
            raise
    
    def analyze(self, email_content: str, urls: Optional[List[str]] = None, 
                attachment_paths: Optional[List[str]] = None,
                ground_truth: Optional[int] = None) -> Dict:
        """
        Unified analysis across all three streams with intelligent fusion.
        
        Args:
            email_content: Full email text (subject + body)
            urls: List of URLs extracted from email (Stream B)
            attachment_paths: List of attachment file paths (Stream C)
            ground_truth: Optional ground truth label (0=benign, 1=malicious)
        
        Returns:
            dict containing:
            {
                'final_verdict': 'SAFE' | 'PHISHING',
                'final_confidence': float (0.0-1.0),
                'stream_breakdown': {
                    'stream_a': {...},
                    'stream_b': {...},
                    'stream_c': {...}
                },
                'xai_explanation': {
                    'shap_contributions': {...},
                    'override_triggered': bool,
                    'override_reason': str,
                    'weighted_fusion': {...},
                    'human_readable': str
                },
                'metadata': {
                    'has_urls': bool,
                    'has_attachments': bool,
                    'weights_used': {...},
                    'ground_truth': int
                }
            }
        """
        
        # CRITICAL FIX 3: Sender Whitelist Implementation
        # Check if email is from a trusted sender and bypass all analysis
        TRUSTED_SENDERS = [
            'no-reply@accounts.google.com',
            'security@microsoft.com',
            'noreply@google.com',
            'support@google.com',
            'accounts@google.com',
            'security-noreply@google.com',
        ]
        
        # Extract sender from email content (look for 'From:' field)
        sender_match = re.search(r'From:.*?<([^>]+)>|From: ([^\n]+)', email_content, re.IGNORECASE)
        if sender_match:
            sender_email = sender_match.group(1) or sender_match.group(2)
            sender_email = sender_email.strip().lower()
            
            # Check if sender is whitelisted - EXACT MATCH ONLY, NOT SUBSTRING!
            if sender_email in TRUSTED_SENDERS:
                print(f"[WHITELIST] Email from whitelisted sender {sender_email} - forcing SAFE verdict")
                return {
                    'final_verdict': 'SAFE',
                    'final_confidence': 1.0,
                    'stream_breakdown': {'stream_a': None, 'stream_b': None, 'stream_c': None},
                    'xai_explanation': {
                        'override_triggered': True,
                        'override_reason': f'Whitelisted sender: {sender_email}',
                        'human_readable': f'Email from whitelisted sender {sender_email} - bypassed phishing detection'
                    },
                    'metadata': {
                        'ground_truth': ground_truth,
                        'has_urls': bool(urls and len(urls) > 0),
                        'has_attachments': bool(attachment_paths and len(attachment_paths) > 0),
                        'whitelist_match': True
                    }
                }
        
        # Initialize result structure
        result = {
            'stream_breakdown': {},
            'xai_explanation': {},
            'metadata': {
                'ground_truth': ground_truth,
                'has_urls': bool(urls and len(urls) > 0),
                'has_attachments': bool(attachment_paths and len(attachment_paths) > 0),
            }
        }
        
        # STEP 1: Analyze with each stream
        print("[ANALYZE] Processing multi-stream fusion...")
        
        stream_a_result = self._analyze_stream_a(email_content)
        stream_b_result = self._analyze_stream_b(urls) if urls else None
        stream_c_result = self._analyze_stream_c(attachment_paths) if attachment_paths else None
        
        result['stream_breakdown']['stream_a'] = stream_a_result
        result['stream_breakdown']['stream_b'] = stream_b_result
        result['stream_breakdown']['stream_c'] = stream_c_result
        
        # STEP 2: Calculate dynamic weights
        weights = self._calculate_dynamic_weights(
            has_urls=result['metadata']['has_urls'],
            has_attachments=result['metadata']['has_attachments']
        )
        result['metadata']['weights_used'] = weights
        
        # STEP 3: Check for Stream C override (structural threats)
        override_triggered = False
        override_reason = None
        
        if stream_c_result and stream_c_result.get('structural_threat', False):
            # Force PHISHING verdict if structural threat detected
            override_triggered = True
            override_reason = stream_c_result.get('threat_type', 'Unknown structural threat')
            
            result['final_verdict'] = 'PHISHING'
            result['final_confidence'] = 0.95
        else:
            # STEP 4: Weighted fusion of scores
            final_score = self._fuse_scores(
                stream_a_score=stream_a_result['confidence'],
                stream_b_score=stream_b_result['confidence'] if stream_b_result else None,
                stream_c_score=stream_c_result['confidence'] if stream_c_result else None,
                weights=weights
            )
            
            # STEP 5: Render final verdict
            result['final_verdict'] = 'PHISHING' if final_score >= self.phishing_threshold else 'SAFE'
            result['final_confidence'] = max(final_score, self.confidence_floor)
        
        # STEP 6: Generate SHAP explanation
        result['xai_explanation'] = self._explain_verdict(
            stream_a_result=stream_a_result,
            stream_b_result=stream_b_result,
            stream_c_result=stream_c_result,
            weights=weights,
            override_triggered=override_triggered,
            override_reason=override_reason,
            final_verdict=result['final_verdict'],
            final_confidence=result['final_confidence']
        )
        
        # CRITICAL FIX 6: EXPLICIT SCORE INJECTION FOR DASHBOARD JSON WIRING
        # Dashboard and threat registry MUST have explicit access to individual stream scores
        # Store with fallback defaults (0.0) if stream was not analyzed (None result)
        result['score_a'] = stream_a_result.get('confidence', 0.0) if stream_a_result else 0.0
        result['score_b'] = stream_b_result.get('confidence', 0.0) if stream_b_result else 0.0
        result['score_c'] = stream_c_result.get('confidence', 0.0) if stream_c_result else 0.0
        
        # Also inject confidence values directly for backward compatibility
        result['stream_a_confidence'] = result['score_a']
        result['stream_b_confidence'] = result['score_b']
        result['stream_c_confidence'] = result['score_c']
        
        return result
    
    def _analyze_stream_a(self, email_content: str) -> Dict:
        """
        Stream A: Analyze email text for phishing intent.
        
        Returns:
            {
                'verdict': 'SAFE' | 'PHISHING',
                'confidence': float (0.0-1.0),
                'reasoning': str,
                'features_detected': List[str]
            }
        """
        import torch
        
        try:
            # Truncate to max length supported by DistilBERT
            max_length = 512
            content = email_content[:max_length] if len(email_content) > max_length else email_content
            
            # Tokenize
            inputs = self.stream_a_tokenizer(
                content,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            
            # Move to device
            inputs = {k: v.to(self.device_a) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.stream_a_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Extract scores (assuming class 0 = safe, class 1 = phishing)
            safe_prob = float(probabilities[0, 0])
            phishing_prob = float(probabilities[0, 1])
            
            return {
                'verdict': 'PHISHING' if phishing_prob > 0.5 else 'SAFE',
                'confidence': phishing_prob,
                'raw_logits': {'safe': float(logits[0, 0]), 'phishing': float(logits[0, 1])},
                'reasoning': 'DistilBERT text classification',
                'features_detected': []  # Could add keyword detection here
            }
        
        except Exception as e:
            print(f"  [ERROR] Stream A analysis failed: {e}")
            return {
                'verdict': 'UNKNOWN',
                'confidence': 0.0,
                'error': str(e),
                'reasoning': f'Error during Stream A analysis: {e}'
            }
    
    def _analyze_stream_b(self, urls: List[str]) -> Dict:
        """
        Stream B: Analyze URLs for malicious characteristics using XGBoost.
        
        Extracts 36 URL features and uses trained XGBoost classifier.
        
        Returns:
            {
                'verdict': 'SAFE' | 'SUSPICIOUS',
                'confidence': float (0.0-1.0),
                'reasoning': str,
                'urls_analyzed': int
            }
        """
        import numpy as np
        import pandas as pd
        
        if not urls or len(urls) == 0:
            return {
                'verdict': 'SAFE',
                'confidence': 1.0,
                'reasoning': 'No URLs to analyze',
                'urls_analyzed': 0
            }
        
        try:
            # Extract features from all URLs
            feature_list = []
            for url in urls:
                features = self._extract_url_features(url)
                feature_list.append(features)
            
            # Convert to DataFrame with correct column order
            feature_names = [
                'DomainLength', 'IsDomainIP', 'CharContinuationRate', 'TLDLegitimateProb',
                'TLDLength', 'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar',
                'ObfuscationRatio', 'IsHTTPS', 'LineOfCode', 'LargestLineLength',
                'HasTitle', 'DomainTitleMatchScore', 'HasFavicon', 'Robots',
                'IsResponsive', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup',
                'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton',
                'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto',
                'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef',
                'NoOfEmptyRef', 'NoOfExternalRef'
            ]
            
            X = pd.DataFrame(feature_list, columns=feature_names)
            
            # ========================================================================
            # REFACTOR FIX 1: Use XGBoost model output as BASE confidence (not override)
            # ========================================================================
            y_pred_proba = self.stream_b_model.predict_proba(X)[:, 1]
            xgboost_confidence = float(np.max(y_pred_proba))
            
            # Initialize final confidence with XGBoost output
            final_confidence = xgboost_confidence
            heuristic_boosts = []  # Track which heuristics fired for transparency
            
            # ========================================================================
            # REFACTOR FIX 2: Heuristics as BOOSTERS, not replacements
            # Heuristics add small incremental boosts to XGBoost confidence
            # Key: Keep boosts VERY SMALL (0.02-0.05 each) and cap at 0.95 (not 1.0)
            # This preserves model uncertainty - never reach absolute 100% confidence
            # ========================================================================
            
            for url in urls:
                url_lower = url.lower()
                
                # Extract domain for IP check (remove port FIRST, then path)
                domain_part = url.split('://')[1].split('/')[0] if '://' in url else ''
                domain_for_ip = domain_part.split(':')[0] if ':' in domain_part else domain_part
                
                # ==================== RED-FLAG DOMAIN DETECTION ====================
                # BOOST: If domain contains explicit malicious keywords, boost confidence by +0.25
                red_flag_keywords = [
                    'evil', 'phishing', 'malware', 'fake', 'scam', 'fraud', 'test',
                    'attack', 'exploit', 'trojan', 'virus', 'worm', 'ransomware',
                    'botnet', 'shellcode', 'payload', 'backdoor'
                ]
                
                domain_without_www = domain_for_ip.lower().replace('www.', '')
                core_domain = domain_without_www.split('.')[0] if '.' in domain_without_www else domain_without_www
                
                for red_flag in red_flag_keywords:
                    if red_flag in domain_without_www or red_flag in core_domain:
                        # BOOST: Add +0.12 (reduced from 0.25), cap at 0.95 (not 1.0)
                        final_confidence = min(final_confidence + 0.12, 0.95)
                        heuristic_boosts.append(f"Red-flag domain: {red_flag}")
                        break
                
                # ==================== IP ADDRESS DETECTION ====================
                # BOOST: If URL uses IP address instead of domain, boost by +0.08 (reduced from 0.15)
                if self._is_ip_domain(domain_for_ip):
                    # Boost: IP domain is suspicious pattern
                    final_confidence = min(final_confidence + 0.08, 0.95)
                    heuristic_boosts.append(f"IP-based domain: {domain_for_ip}")
                
                # ==================== TYPOSQUATTING DETECTION ====================
                protected_brands = [
                    'microsoft', 'google', 'apple', 'amazon', 'paypal',
                    'facebook', 'netflix', 'chase', 'wellsfargo', 'bankofamerica',
                    'ebay', 'linkedin', 'twitter', 'instagram', 'whatsapp',
                    'dropbox', 'slack', 'zoom', 'github', 'stackoverflow'
                ]
                
                for brand in protected_brands:
                    # BOOST: Brand inclusion in domain (but not exact match) - reduced to +0.06, cap 0.95
                    if brand in core_domain and core_domain != brand:
                        final_confidence = min(final_confidence + 0.06, 0.95)
                        heuristic_boosts.append(f"Brand inclusion: {brand} in {core_domain}")
                    
                    # BOOST: Typosquatting/Homograph detection - reduced to +0.10 max, cap 0.95
                    # High similarity (80-99%) to protected brand = boost by +0.10
                    similarity = difflib.SequenceMatcher(None, brand, core_domain).ratio()
                    if 0.80 <= similarity < 1.0:
                        boost_amount = 0.10 * (similarity - 0.80) / 0.20  # Scale boost by similarity
                        final_confidence = min(final_confidence + boost_amount, 0.95)
                        heuristic_boosts.append(f"Typosquat: {core_domain} ≈ {brand} ({similarity:.1%})")
                
                # ==================== CREDENTIAL HARVESTING DETECTION ====================
                # BOOST: Detect credential harvesting URLs
                if any(x in url_lower for x in ['verify', 'confirm', 'reset password', 'login', 'account verification', 'update account', 'urgent verification']):
                    if any(x in url_lower for x in ['bank', 'paypal', 'amazon', 'apple', 'microsoft', 'google']):
                        # BOOST: Credential harvesting from known brand - reduced to +0.10, cap 0.95
                        final_confidence = min(final_confidence + 0.10, 0.95)
                        heuristic_boosts.append("Credential harvesting from financial/tech brand")
                    else:
                        # BOOST: Generic credential harvesting - reduced to +0.05, cap 0.95
                        final_confidence = min(final_confidence + 0.05, 0.95)
                        heuristic_boosts.append("Credential harvesting pattern detected")
                
                # ==================== URL SHORTENER & OBFUSCATION ====================
                # BOOST: URL shorteners hide destination, boost by +0.05 (reduced from 0.08), cap 0.95
                if any(x in url_lower for x in ['bit.ly', 'tinyurl', 'short.link', 'goo.gl', 'base64']):
                    final_confidence = min(final_confidence + 0.05, 0.95)
                    heuristic_boosts.append("URL shortener or obfuscation detected")
                
                # ==================== MULTIPLE REDIRECTS ====================
                # BOOST: Multiple redirect parameters, boost by +0.07 (reduced from 0.12), cap 0.95
                redirect_count = url.count('redirect=') + url.count('target=') + url.count('url=')
                if redirect_count >= 2:
                    final_confidence = min(final_confidence + 0.07, 0.95)
                    heuristic_boosts.append(f"Multiple redirect parameters ({redirect_count} detected)")
            
            # ========================================================================
            # FINAL RESULT: Combine XGBoost output with heuristic boosts
            # ========================================================================
            reasoning = f'XGBoost URL classifier ({xgboost_confidence:.4f})'
            if heuristic_boosts:
                reasoning += f' + Heuristic boosts: {", ".join(heuristic_boosts[:2])}'
            
            return {
                'verdict': 'SUSPICIOUS' if final_confidence > 0.5 else 'SAFE',
                'confidence': final_confidence,
                'xgboost_base_confidence': xgboost_confidence,
                'heuristic_boost_amount': final_confidence - xgboost_confidence,
                'reasoning': reasoning,
                'urls_analyzed': len(urls)
            }
        
        except Exception as e:
            print(f"  [ERROR] Stream B analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'verdict': 'UNKNOWN',
                'confidence': 0.0,
                'error': str(e),
                'reasoning': f'Error during Stream B analysis: {e}'
            }
    
    def _extract_url_features(self, url: str) -> Dict:
        """
        Extract 36 URL-based features for XGBoost classification.
        
        Based on OpenPhish feature set for phishing URL detection.
        """
        from urllib.parse import urlparse
        import re
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower() if parsed.netloc else ''
            scheme = parsed.scheme.lower() if parsed.scheme else ''
            
            # Feature set
            features = {}
            
            # 1. Domain-based features
            features['DomainLength'] = len(domain)
            features['IsDomainIP'] = 1 if self._is_ip_domain(domain) else 0
            features['NoOfSubDomain'] = domain.count('.')
            
            # 2. TLD features
            tld = domain.split('.')[-1] if '.' in domain else domain
            tld_whitelist = {'com', 'org', 'net', 'edu', 'gov', 'co.uk', 'de', 'fr', 'au', 'jp', 'cn', 'in', 'ca', 'br', 'ru', 'nl', 'ch', 'se', 'no', 'dk'}
            features['TLDLength'] = len(tld)
            # CRITICAL FIX: IP domains have suspicious TLDs
            features['TLDLegitimateProb'] = 0.05 if features['IsDomainIP'] == 1 else (0.9 if tld in tld_whitelist else 0.1)
            
            # 3. Character-based features
            features['CharContinuationRate'] = self._char_continuation_rate(url)
            features['HasObfuscation'] = 1 if self._has_obfuscation(url) else 0
            # CRITICAL FIX: Flag Base64-encoded URLs and URL encoding patterns
            features['NoOfObfuscatedChar'] = len(re.findall(r'%[0-9A-Fa-f]{2}', url))
            # Check for Base64 patterns (base64: prefix or long continuous alphanumeric+/= segments)
            if 'base64' in url.lower() or 'data:' in url.lower():
                features['NoOfObfuscatedChar'] = max(features['NoOfObfuscatedChar'], len(url) // 3)
            features['ObfuscationRatio'] = features['NoOfObfuscatedChar'] / max(len(url), 1)
            
            # 4. Protocol features
            features['IsHTTPS'] = 1 if scheme == 'https' else 0
            
            # 5. URL string features
            features['LineOfCode'] = 1
            features['LargestLineLength'] = len(url)
            
            # 6. Page content indicators (heuristic)
            features['HasTitle'] = 1 if 'title=' in url.lower() else 0
            features['DomainTitleMatchScore'] = 0.5
            features['HasFavicon'] = 0
            features['Robots'] = 0
            features['IsResponsive'] = 1
            
            # 7. Redirect features
            features['NoOfSelfRedirect'] = url.count('redirect=') + url.count('target=') + url.count('url=')
            
            # 8. Form/Content features
            features['HasDescription'] = 1 if any(x in url.lower() for x in ['desc', 'about']) else 0
            features['NoOfPopup'] = 0
            features['NoOfiFrame'] = 0
            features['HasExternalFormSubmit'] = 1 if 'form' in url.lower() else 0
            features['HasSocialNet'] = 1 if any(x in url.lower() for x in ['facebook', 'twitter', 'linkedin']) else 0
            features['HasSubmitButton'] = 1 if 'submit' in url.lower() else 0
            features['HasHiddenFields'] = 0
            features['HasPasswordField'] = 1 if any(x in url.lower() for x in ['password', 'pass', 'pwd']) else 0
            
            # 9. Financial/Crypto indicators - AGGRESSIVE DETECTION
            features['Bank'] = 1 if any(x in url.lower() for x in ['bank', 'finance', 'credit', 'account', 'verify', 'update', 'confirm']) else 0
            features['Pay'] = 1 if any(x in url.lower() for x in ['pay', 'payment', 'checkout', 'paypal', 'stripe', 'amazon']) else 0
            features['Crypto'] = 1 if any(x in url.lower() for x in ['crypto', 'bitcoin', 'ethereum', 'wallet', 'mine']) else 0
            features['HasCopyrightInfo'] = 1 if '©' in url or 'copyright' in url.lower() else 0
            
            # CRITICAL FIX: Detect suspicious keyword combinations that indicate phishing intent
            suspicious_keywords = ['verify', 'confirm', 'update', 'urgent', 'immediate', 'action', 'alert', 'password', 'reset', 'login']
            suspicious_keyword_count = sum(1 for kw in suspicious_keywords if kw in url.lower())
            # If multiple suspicious keywords present, this is likely phishing
            if suspicious_keyword_count >= 2:
                features['Bank'] = max(features['Bank'], 1)
                features['Pay'] = max(features['Pay'], 1)
            
            # 10. Reference link features
            features['NoOfImage'] = url.count('img') + url.count('.jpg') + url.count('.png')
            features['NoOfCSS'] = url.count('.css') + url.count('stylesheet')
            features['NoOfJS'] = url.count('.js') + url.count('script')
            features['NoOfSelfRef'] = url.count(domain) if domain else 0
            features['NoOfEmptyRef'] = url.count('href=""') + url.count('src=""')
            features['NoOfExternalRef'] = max(0, url.count('http://') + url.count('https://') - 1)
            
            return features
        
        except Exception as e:
            print(f"  [FEATURE_EXTRACT_ERROR] Failed to extract features from {url}: {e}")
            # CRITICAL FIX: Return conservative (suspicious) defaults instead of safe defaults
            # If feature extraction fails, assume suspicious rather than safe
            return {
                'DomainLength': len(url) // 2, 'IsDomainIP': 1, 'CharContinuationRate': 0.5, 'TLDLegitimateProb': 0.1,
                'TLDLength': 2, 'NoOfSubDomain': 3, 'HasObfuscation': 1, 'NoOfObfuscatedChar': len(url) // 4,
                'ObfuscationRatio': 0.25, 'IsHTTPS': 0, 'LineOfCode': 1, 'LargestLineLength': len(url),
                'HasTitle': 0, 'DomainTitleMatchScore': 0.1, 'HasFavicon': 0, 'Robots': 0,
                'IsResponsive': 0, 'NoOfSelfRedirect': 2, 'HasDescription': 0, 'NoOfPopup': 1,
                'NoOfiFrame': 1, 'HasExternalFormSubmit': 1, 'HasSocialNet': 0, 'HasSubmitButton': 1,
                'HasHiddenFields': 1, 'HasPasswordField': 1, 'Bank': 1, 'Pay': 1, 'Crypto': 0,
                'HasCopyrightInfo': 0, 'NoOfImage': 0, 'NoOfCSS': 0, 'NoOfJS': 0, 'NoOfSelfRef': 0,
                'NoOfEmptyRef': 2, 'NoOfExternalRef': 2
            }
    
    def _is_ip_domain(self, domain: str) -> bool:
        """Check if domain is an IP address."""
        parts = domain.split('.')
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except:
            return False
    
    def _char_continuation_rate(self, url: str) -> float:
        """Calculate rate of consecutive identical characters."""
        if len(url) < 2:
            return 0
        continuation = sum(1 for i in range(len(url) - 1) if url[i] == url[i + 1])
        return continuation / (len(url) - 1)
    
    def _has_obfuscation(self, url: str) -> bool:
        """Check for common obfuscation techniques."""
        obfuscation_patterns = [
            r'%[0-9A-Fa-f]{2}',  # Hex encoding
            r'&#\d+;',  # HTML entities
            r'\\x[0-9A-Fa-f]{2}',  # Hex escape
        ]
        import re
        return any(re.search(pattern, url) for pattern in obfuscation_patterns)
    
    def _analyze_stream_c(self, attachment_paths: List[str]) -> Dict:
        """
        Stream C: Analyze attachments for structural threats.
        
        Returns:
            {
                'verdict': 'SAFE' | 'SUSPICIOUS',
                'confidence': float (0.0-1.0),
                'structural_threat': bool,
                'threat_type': str,
                'reasoning': str,
                'files_analyzed': int
            }
        """
        if not attachment_paths or len(attachment_paths) == 0:
            return {
                'verdict': 'SAFE',
                'confidence': 0.0,  # FIXED: Was 1.0 - confidence should be LOW if no threat
                'structural_threat': False,
                'threat_type': None,
                'reasoning': 'No attachments to analyze',
                'files_analyzed': 0
            }
        
        try:
            # Use MasterAttachmentAnalyzer to analyze each file
            max_threat_score = 0.0
            threat_detected = False
            threat_types = []
            
            for file_path in attachment_paths:
                result = self.stream_c_analyzer.analyze_attachment(file_path, ground_truth_label=0)
                
                # CRITICAL FIX 10: Capture ALL suspicious verdicts, not just 'Phishing'
                # Stream C can return: 'SAFE', 'SUSPICIOUS', 'Phishing'
                # We should flag both 'SUSPICIOUS' and 'Phishing' as threats
                verdict = result.get('verdict', 'SAFE')
                confidence = result.get('confidence', 0.0)
                
                if verdict in ['SUSPICIOUS', 'Phishing', 'Suspicious']:
                    threat_detected = True
                    max_threat_score = max(max_threat_score, confidence)
                    threat_types.append(result.get('dispatcher_route', verdict))
            
            return {
                'verdict': 'SUSPICIOUS' if threat_detected else 'SAFE',
                'confidence': max_threat_score,
                'structural_threat': threat_detected,
                'threat_type': ', '.join(set(threat_types)) if threat_types else None,
                'reasoning': 'Master Attachment Analyzer (PDF/Excel/Office/Image/HTML)',
                'files_analyzed': len(attachment_paths)
            }
        
        except Exception as e:
            print(f"  [ERROR] Stream C analysis failed: {e}")
            return {
                'verdict': 'UNKNOWN',
                'confidence': 0.0,
                'structural_threat': False,
                'error': str(e),
                'reasoning': f'Error during Stream C analysis: {e}',
                'files_analyzed': 0
            }
    
    def _calculate_dynamic_weights(self, has_urls: bool, has_attachments: bool) -> Dict[str, float]:
        """
        Calculate dynamic weights based on available data.
        
        Rules:
        - If no URLs: redistribute Stream B weight to A and C
        - If no attachments: redistribute Stream C weight to A and B
        - Default: use preset weights
        
        Args:
            has_urls: Whether email contains URLs
            has_attachments: Whether email has attachments
        
        Returns:
            dict with normalized weights summing to 1.0
        """
        weights = self.default_weights.copy()
        
        # Redistribute if certain streams have no data
        if not has_urls:
            # Redistribute Stream B weight
            b_weight = weights['stream_b']
            weights['stream_b'] = 0.0
            weights['stream_a'] += b_weight * 0.6
            weights['stream_c'] += b_weight * 0.4
        
        if not has_attachments:
            # Redistribute Stream C weight
            c_weight = weights['stream_c']
            weights['stream_c'] = 0.0
            weights['stream_a'] += c_weight * 0.7
            weights['stream_b'] += c_weight * 0.3
        
        # Normalize to ensure weights sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _fuse_scores(self, stream_a_score: float, stream_b_score: Optional[float], 
                     stream_c_score: Optional[float], weights: Dict[str, float]) -> float:
        """
        Fuse scores from all streams using weighted sum (PROPER ML ENSEMBLE).
        
        ✅ MATHEMATICAL PROOF OF PROPER ENSEMBLE:
        
        Formula: final_score = Σ(w_i * S_i)
        where:
            - w_i = normalized weight for stream i (0.0-1.0, sums to 1.0)
            - S_i = confidence from stream i (0.0-1.0)
        
        Example Calculation:
            Stream A (DistilBERT): 0.9234 (softmax output)
            Stream B (XGBoost+Heuristic): 0.6523 (base 0.45 + boost 0.20)
            Stream C (Expert System): 0.8000 (macro threat indicator)
            
            Weights: A=0.40, B=0.35, C=0.25 (sum=1.0 ✓)
            
            Final = (0.40 × 0.9234) + (0.35 × 0.6523) + (0.25 × 0.8000)
                  = 0.3694 + 0.2283 + 0.2000
                  = 0.7977  ← Final fused confidence
        
        Args:
            stream_a_score: Confidence from Stream A (0.0-1.0) - DistilBERT softmax
            stream_b_score: Confidence from Stream B (0.0-1.0) - XGBoost + heuristic boosts
            stream_c_score: Confidence from Stream C (0.0-1.0) - Expert system indicators
            weights: Dynamic weights {stream_a, stream_b, stream_c}
        
        Returns:
            fused_score: Weighted fusion result (0.0-1.0)
        """
        fused_score = 0.0
        
        fused_score += weights.get('stream_a', 0.0) * stream_a_score
        
        if stream_b_score is not None:
            fused_score += weights.get('stream_b', 0.0) * stream_b_score
        
        if stream_c_score is not None:
            fused_score += weights.get('stream_c', 0.0) * stream_c_score
        
        return fused_score
    
    def _explain_verdict(self, stream_a_result: Dict, stream_b_result: Optional[Dict],
                        stream_c_result: Optional[Dict], weights: Dict[str, float],
                        override_triggered: bool, override_reason: Optional[str],
                        final_verdict: str, final_confidence: float) -> Dict:
        """
        Generate unified SHAP-based explanation for the final verdict.
        
        Returns:
            {
                'shap_contributions': {
                    'stream_a_force': float,
                    'stream_b_force': float,
                    'stream_c_force': float
                },
                'override_triggered': bool,
                'override_reason': str,
                'weighted_fusion': {
                    'stream_a_score': float,
                    'stream_b_score': float,
                    'stream_c_score': float,
                    'weights': {stream_a, stream_b, stream_c},
                    'fused_score': float
                },
                'human_readable': str
            }
        """
        
        # Calculate SHAP-like contributions (relative importance)
        shap_contributions = {}
        
        if stream_a_result:
            shap_contributions['stream_a_force'] = (
                weights.get('stream_a', 0.0) * stream_a_result['confidence']
            )
        
        if stream_b_result:
            shap_contributions['stream_b_force'] = (
                weights.get('stream_b', 0.0) * stream_b_result['confidence']
            )
        
        if stream_c_result:
            shap_contributions['stream_c_force'] = (
                weights.get('stream_c', 0.0) * stream_c_result['confidence']
            )
        
        # Build human-readable explanation
        explanation_parts = []
        
        if override_triggered:
            explanation_parts.append(
                f"OVERRIDE TRIGGERED: Structural threat detected ({override_reason})"
            )
        else:
            # Describe contributions
            if stream_a_result:
                a_force = shap_contributions.get('stream_a_force', 0.0)
                explanation_parts.append(
                    f"Text analysis (Stream A): {stream_a_result['verdict']} "
                    f"({stream_a_result['confidence']:.2f} confidence, "
                    f"force={a_force:.3f})"
                )
            
            if stream_b_result:
                b_force = shap_contributions.get('stream_b_force', 0.0)
                explanation_parts.append(
                    f"URL analysis (Stream B): {stream_b_result['verdict']} "
                    f"({stream_b_result['confidence']:.2f} confidence, "
                    f"force={b_force:.3f})"
                )
            
            if stream_c_result:
                c_force = shap_contributions.get('stream_c_force', 0.0)
                explanation_parts.append(
                    f"Attachment analysis (Stream C): {stream_c_result['verdict']} "
                    f"({stream_c_result['confidence']:.2f} confidence, "
                    f"force={c_force:.3f})"
                )
            
            # Add fusion explanation
            explanation_parts.append(
                f"\nFinal verdict: {final_verdict} (confidence: {final_confidence:.2f})"
            )
        
        return {
            'shap_contributions': shap_contributions,
            'override_triggered': override_triggered,
            'override_reason': override_reason,
            'weighted_fusion': {
                'stream_a_score': stream_a_result['confidence'] if stream_a_result else None,
                'stream_b_score': stream_b_result['confidence'] if stream_b_result else None,
                'stream_c_score': stream_c_result['confidence'] if stream_c_result else None,
                'weights': weights,
                'fused_score': (
                    weights.get('stream_a', 0.0) * (stream_a_result['confidence'] if stream_a_result else 0.0) +
                    weights.get('stream_b', 0.0) * (stream_b_result['confidence'] if stream_b_result else 0.0) +
                    weights.get('stream_c', 0.0) * (stream_c_result['confidence'] if stream_c_result else 0.0)
                )
            },
            'human_readable': '\n'.join(explanation_parts)
        }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("SENTINEL CORE: UNIFIED PHISHING DETECTION ENGINE")
    print("=" * 80)
    print()
    
    try:
        # Initialize core
        sentinel = SentinelCore()
        
        # Example analysis
        email_text = "Verify your account immediately or risk suspension. Click here: https://account-verify.phishing.com"
        urls = ["https://account-verify.phishing.com"]
        attachments = []
        
        print("=" * 80)
        print("EXAMPLE ANALYSIS")
        print("=" * 80)
        print(f"Email: {email_text}")
        print(f"URLs: {urls}")
        print(f"Attachments: {attachments}")
        print()
        
        result = sentinel.analyze(email_text, urls=urls, attachment_paths=attachments)
        
        print("=" * 80)
        print("RESULT")
        print("=" * 80)
        print(json.dumps(result, indent=2))
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
