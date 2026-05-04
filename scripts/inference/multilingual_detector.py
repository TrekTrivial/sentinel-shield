#!/usr/bin/env python3
"""
Enhanced Stream A with multilingual phishing detection.
Translates non-English text to English before DistilBERT analysis.
"""

def add_multilingual_detection(email_text: str) -> str:
    """
    Detect and translate non-English phishing attempts to English.
    
    Returns:
        Enhanced text with translation hints for better phishing detection
    """
    try:
        from textblob import TextBlob
    except ImportError:
        print("[!] textblob not installed. Install: pip install textblob")
        return email_text
    
    # Detect language
    blob = TextBlob(email_text)
    detected_lang = blob.detect_language()
    
    if detected_lang != 'en':
        print(f"[*] Detected language: {detected_lang}, translating to English...")
        
        # Translate to English
        translated = blob.translate(from_lang=detected_lang, to_lang='en')
        
        # Combine original + translation for DistilBERT
        # This gives model context for pattern matching
        enhanced_text = f"[TRANSLATED from {detected_lang}]\n{str(translated)}\n[ORIGINAL]\n{email_text}"
        
        print(f"    Sample: {str(translated)[:100]}...")
        return enhanced_text
    
    return email_text


# Usage in sentinel_core.py:
# Before feeding text to DistilBERT:
# email_text = add_multilingual_detection(email_text)
# confidence = stream_a_model.predict(email_text)
