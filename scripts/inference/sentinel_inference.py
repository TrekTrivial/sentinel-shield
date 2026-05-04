#!/usr/bin/env python3
"""
SENTINEL - Unified Inference Pipeline
Processes emails through Stream A (TEXT) → Stream B (URL) → Stream C (ATTACHMENTS)
Returns explainable predictions with confidence scores
Cross-platform compatible (Windows/Linux)
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib

# Stream C imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'stream_C_attachments'))
from attachment_extractor import AttachmentExtractor

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SentinelPipeline:
    """
    SENTINEL: Explainable AI Phishing Detection System
    
    Three-Stream Architecture:
    - Stream A: Text analysis (DistilBERT)
    - Stream B: URL analysis (XGBoost)
    - Stream C: Attachment extraction → Route to Stream A
    
    Provides explainability via:
    - Individual stream confidence scores
    - Risk assessment per stream
    - Detailed classification reasoning
    """
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize SENTINEL pipeline with all three streams.
        
        Args:
            model_dir: Path to models directory (cross-platform compatible)
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("=" * 90)
        logger.info("SENTINEL: Explainable Phishing Detection System")
        logger.info("=" * 90)
        logger.info(f"\n✓ Device: {self.device}")
        
        # Load Stream A (Text)
        self._load_stream_a()
        
        # Load Stream B (URL)
        self._load_stream_b()
        
        # Initialize Stream C (Attachments)
        self._init_stream_c()
        
        # Ensemble weights (can be tuned based on validation)
        self.stream_weights = {
            'stream_a': 0.50,  # Text: most reliable on diverse emails
            'stream_b': 0.25,  # URL: structural check
            'stream_c': 0.25,  # Attachment: extracted text re-analyzed by Stream A
        }
        
        logger.info("\n✓ SENTINEL pipeline initialized")
        logger.info(f"  Stream A (TEXT):        DistilBERT")
        logger.info(f"  Stream B (URL):         XGBoost")
        logger.info(f"  Stream C (ATTACHMENTS): Extractor + Stream A routing")
        logger.info("\n" + "=" * 90 + "\n")
    
    def _load_stream_a(self):
        """Load Stream A: DistilBERT text classifier"""
        try:
            model_path = self.model_dir / 'text_model_distilbert'
            
            logger.info("Loading Stream A (DistilBERT Text Classifier)...")
            self.stream_a_tokenizer = DistilBertTokenizer.from_pretrained(str(model_path))
            self.stream_a_model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
            self.stream_a_model.to(self.device)
            self.stream_a_model.eval()
            
            logger.info("  ✓ DistilBERT loaded successfully")
        except Exception as e:
            logger.error(f"  ✗ Failed to load Stream A: {e}")
            self.stream_a_model = None
    
    def _load_stream_b(self):
        """Load Stream B: XGBoost URL classifier"""
        try:
            model_path = self.model_dir / 'url_model_xgboost' / 'xgboost_url_model.pkl'
            
            logger.info("Loading Stream B (XGBoost URL Classifier)...")
            self.stream_b_model = joblib.load(str(model_path))
            self.stream_b_features = [
                'DomainLength', 'IsDomainIP', 'CharContinuationRate', 'TLDLegitimateProb',
                'TLDLength', 'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar',
                'ObfuscationRatio', 'IsHTTPS', 'LineOfCode', 'LargestLineLength', 'HasTitle',
                'DomainTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive',
                'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 'NoOfiFrame',
                'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields',
                'HasPasswordField', 'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo', 'NoOfImage',
                'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef'
            ]
            
            logger.info("  ✓ XGBoost URL model loaded successfully")
        except Exception as e:
            logger.error(f"  ✗ Failed to load Stream B: {e}")
            self.stream_b_model = None
    
    def _init_stream_c(self):
        """Initialize Stream C: Attachment extractor"""
        try:
            logger.info("Initializing Stream C (Attachment Extractor)...")
            self.stream_c_extractor = AttachmentExtractor(enable_ocr=True)
            logger.info("  ✓ Attachment extractor initialized")
        except Exception as e:
            logger.error(f"  ✗ Failed to initialize Stream C: {e}")
            self.stream_c_extractor = None
    
    def analyze_email(self, 
                     email_text: str,
                     email_urls: Optional[List[str]] = None,
                     attachment_paths: Optional[List[str]] = None) -> Dict:
        """
        Analyze email through all three streams.
        
        Args:
            email_text: Email body text
            email_urls: List of URLs found in email
            attachment_paths: List of attachment file paths
            
        Returns:
            {
                'prediction': 'phishing' | 'legitimate',
                'risk_score': 0.0-1.0 (1.0 = high risk),
                'confidence': 0.0-1.0 (model certainty),
                'streams': {
                    'stream_a': {...},  # Text classification
                    'stream_b': {...},  # URL classification
                    'stream_c': {...},  # Attachment classification
                },
                'reasoning': str,  # Explainable reasoning
                'recommendations': List[str],
            }
        """
        
        results = {
            'prediction': None,
            'risk_score': 0.0,
            'confidence': 0.0,
            'streams': {},
            'reasoning': [],
            'recommendations': [],
        }
        
        # Stream A: Analyze email text
        if self.stream_a_model:
            stream_a_result = self._analyze_stream_a(email_text)
            results['streams']['stream_a'] = stream_a_result
            results['reasoning'].append(stream_a_result['reasoning'])
        else:
            results['streams']['stream_a'] = {'error': 'Stream A not available'}
        
        # Stream B: Analyze URLs
        if email_urls and self.stream_b_model:
            stream_b_result = self._analyze_stream_b(email_urls)
            results['streams']['stream_b'] = stream_b_result
            results['reasoning'].append(stream_b_result['reasoning'])
        else:
            results['streams']['stream_b'] = {'skipped': 'No URLs provided'}
        
        # Stream C: Extract from attachments and analyze
        if attachment_paths and self.stream_c_extractor and self.stream_a_model:
            stream_c_result = self._analyze_stream_c(attachment_paths)
            results['streams']['stream_c'] = stream_c_result
            results['reasoning'].append(stream_c_result['reasoning'])
        else:
            results['streams']['stream_c'] = {'skipped': 'No attachments provided'}
        
        # Ensemble decision
        results = self._ensemble_decision(results)
        
        return results
    
    def _analyze_stream_a(self, text: str) -> Dict:
        """Stream A: Text classification via DistilBERT"""
        
        try:
            if not text or len(text.strip()) < 10:
                return {
                    'prediction': 'unknown',
                    'confidence': 0.0,
                    'risk_score': 0.5,
                    'reasoning': 'Text too short for analysis',
                    'char_count': len(text)
                }
            
            # Tokenize and truncate to 384 tokens
            tokens = self.stream_a_tokenizer(
                text[:2000],  # Limit to 2000 chars for speed
                truncation=True,
                padding=True,
                max_length=384,
                return_tensors='pt'
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.stream_a_model(**tokens)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]
            
            phishing_prob = probs[1].item()  # Class 1 = phishing
            confidence = max(probs[0].item(), probs[1].item())
            
            return {
                'prediction': 'phishing' if phishing_prob > 0.5 else 'legitimate',
                'confidence': confidence,
                'risk_score': phishing_prob,
                'reasoning': self._get_text_reasoning(phishing_prob),
                'probability_distribution': {
                    'legitimate': probs[0].item(),
                    'phishing': probs[1].item()
                }
            }
            
        except Exception as e:
            return {'error': f'Stream A analysis failed: {e}'}
    
    def _analyze_stream_b(self, urls: List[str]) -> Dict:
        """Stream B: URL classification via XGBoost"""
        
        try:
            if not urls:
                return {'skipped': 'No URLs provided'}
            
            # For simplicity, analyze first URL (can be extended)
            url = urls[0]
            
            # Extract basic features (simplified - would need full URLFeatureExtractor)
            features = self._extract_url_features(url)
            
            if not features:
                return {'error': f'Could not extract features from URL: {url}'}
            
            # Predict
            feature_df = pd.DataFrame([features])
            feature_cols = [col for col in self.stream_b_features if col in feature_df.columns]
            
            if len(feature_cols) < len(self.stream_b_features):
                logger.warning(f"  ⚠ Only {len(feature_cols)}/{len(self.stream_b_features)} features available")
            
            X = feature_df[feature_cols]
            pred_proba = self.stream_b_model.predict_proba(X)[0]
            
            phishing_prob = pred_proba[1]
            
            return {
                'prediction': 'phishing' if phishing_prob > 0.5 else 'legitimate',
                'confidence': max(pred_proba),
                'risk_score': phishing_prob,
                'reasoning': self._get_url_reasoning(url, phishing_prob),
                'url': url,
                'probability_distribution': {
                    'legitimate': pred_proba[0],
                    'phishing': pred_proba[1]
                }
            }
            
        except Exception as e:
            return {'error': f'Stream B analysis failed: {e}'}
    
    def _analyze_stream_c(self, attachment_paths: List[str]) -> Dict:
        """Stream C: Extract attachments → analyze text via Stream A"""
        
        try:
            extracted_texts = []
            extraction_confidence = []
            
            for att_path in attachment_paths[:3]:  # Limit to 3 attachments for speed
                result = self.stream_c_extractor.extract(att_path)
                
                if result['success']:
                    extracted_texts.append(result['text'])
                    extraction_confidence.append(result['confidence'])
            
            if not extracted_texts:
                return {'skipped': 'No text could be extracted from attachments'}
            
            # Combine extracted texts
            combined_text = "\n".join(extracted_texts)
            
            # Analyze combined text through Stream A
            text_analysis = self._analyze_stream_a(combined_text)
            
            return {
                'prediction': text_analysis.get('prediction'),
                'confidence': text_analysis.get('confidence', 0.0),
                'risk_score': text_analysis.get('risk_score', 0.5),
                'reasoning': f"Extracted text from {len(extracted_texts)} attachment(s): {text_analysis.get('reasoning', '')}",
                'extracted_attachments': len(extracted_texts),
                'extraction_confidence': np.mean(extraction_confidence),
                'probability_distribution': text_analysis.get('probability_distribution')
            }
            
        except Exception as e:
            return {'error': f'Stream C analysis failed: {e}'}
    
    def _ensemble_decision(self, results: Dict) -> Dict:
        """Combine three streams with weighted voting"""
        
        decisions = []
        scores = []
        confidences = []
        
        weights = self.stream_weights
        total_weight = 0
        
        for stream_name, stream_result in results['streams'].items():
            if 'risk_score' in stream_result:
                weight = weights[stream_name]
                decisions.append(stream_result.get('prediction'))
                scores.append(stream_result['risk_score'] * weight)
                confidences.append(stream_result.get('confidence', 0.0) * weight)
                total_weight += weight
        
        if scores:
            ensemble_risk = sum(scores) / total_weight if total_weight > 0 else 0.5
            ensemble_confidence = sum(confidences) / total_weight if total_weight > 0 else 0.0
        else:
            ensemble_risk = 0.5
            ensemble_confidence = 0.0
        
        ensemble_prediction = 'phishing' if ensemble_risk > 0.5 else 'legitimate'
        
        results['prediction'] = ensemble_prediction
        results['risk_score'] = ensemble_risk
        results['confidence'] = ensemble_confidence
        
        # Generate recommendations
        if ensemble_risk > 0.7:
            results['recommendations'].append("🔴 HIGH RISK: Block email immediately")
        elif ensemble_risk > 0.5:
            results['recommendations'].append("🟠 SUSPICIOUS: Review before opening")
        else:
            results['recommendations'].append("🟢 SAFE: Likely legitimate")
        
        return results
    
    def _extract_url_features(self, url: str) -> Dict:
        """Extract basic URL features (simplified version)"""
        
        try:
            from urllib.parse import urlparse
            
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            
            return {
                'DomainLength': len(domain),
                'IsHTTPS': 1 if parsed.scheme == 'https' else 0,
                'CharContinuationRate': len([c for c in domain if c.isalpha()]) / len(domain) if domain else 0,
                'TLDLength': len(domain.split('.')[-1]) if '.' in domain else 0,
            }
        except:
            return {}
    
    @staticmethod
    def _get_text_reasoning(phishing_prob: float) -> str:
        """Explainable reasoning for text classification"""
        
        if phishing_prob > 0.8:
            return "Email body contains strong phishing indicators (urgent language, suspicious requests)"
        elif phishing_prob > 0.6:
            return "Email body has some phishing-like characteristics"
        elif phishing_prob > 0.4:
            return "Email body is borderline - manual review recommended"
        else:
            return "Email body appears legitimate"
    
    @staticmethod
    def _get_url_reasoning(url: str, phishing_prob: float) -> str:
        """Explainable reasoning for URL classification"""
        
        if phishing_prob > 0.8:
            return f"URL structure highly suspicious for phishing: {url[:50]}..."
        elif phishing_prob > 0.6:
            return f"URL has some suspicious characteristics"
        else:
            return f"URL structure appears legitimate"


# CLI Example
if __name__ == '__main__':
    # Initialize SENTINEL
    sentinel = SentinelPipeline()
    
    # Example email analysis
    example_email = """
    Dear valued customer,
    
    Your account has been compromised. Click here immediately to verify your information:
    https://verify-account-securely.tk/login
    
    Best regards,
    Account Security Team
    """
    
    example_urls = ['https://verify-account-securely.tk/login']
    
    print("\n" + "=" * 90)
    print("EXAMPLE ANALYSIS: Suspicious Email")
    print("=" * 90)
    
    result = sentinel.analyze_email(
        email_text=example_email,
        email_urls=example_urls,
        attachment_paths=None
    )
    
    print(f"\n📊 SENTINEL ANALYSIS RESULT")
    print(f"{'-' * 90}")
    print(f"Prediction:    {result['prediction'].upper()}")
    print(f"Risk Score:    {result['risk_score']:.1%} (0% = safe, 100% = phishing)")
    print(f"Confidence:    {result['confidence']:.1%}")
    print(f"\nReasoning:")
    for reason in result['reasoning']:
        print(f"  • {reason}")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  {rec}")
    print(f"\nDetailed Results:")
    print(json.dumps(result['streams'], indent=2))
    print("=" * 90)
