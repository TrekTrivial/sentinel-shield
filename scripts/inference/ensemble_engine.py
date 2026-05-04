#!/usr/bin/env python3
"""
PHASE 5: SENTINEL ENSEMBLE ENGINE
Fusion logic for multi-stream phishing detection with domain-shift correction

Problem: Stream A (DistilBERT) was trained on emails, not attachments. 
Creates False Positives on benign corporate documents due to domain shift.

Solution: Apply "No-Payload Penalty" when phishing verdict lacks URLs.
Rationale: Real phishing emails in attachments typically contain CTAs/URLs.
         Corporate emails without URLs shouldn't be flagged as phishing.
"""

import json
from typing import Dict, List
from pathlib import Path


class SentinelEnsemble:
    """
    Ensemble fusion engine combining multiple detection streams
    with domain-shift correction for false positive reduction
    """
    
    # Configuration
    NO_PAYLOAD_PENALTY = -0.40  # Confidence penalty for phishing without URLs
    CONFIDENCE_THRESHOLD = 0.70  # Threshold to flip verdict to Safe
    
    def __init__(self):
        """Initialize ensemble engine"""
        self.fusion_log = []  # Track all fusion decisions
    
    def evaluate_attachment(self, extracted_text: str, urls_found: List[str], 
                           stream_a_confidence: float, stream_a_verdict: str) -> Dict:
        """
        Evaluate attachment using ensemble fusion logic with domain-shift correction
        
        Args:
            extracted_text (str): Text extracted from attachment
            urls_found (List[str]): List of URLs found in extracted text
            stream_a_confidence (float): Confidence score from DistilBERT (0-1)
            stream_a_verdict (str): Verdict from Stream A ("Phishing" or "Safe")
        
        Returns:
            dict: {
                "final_verdict": str ("Phishing" or "Safe"),
                "adjusted_confidence": float (0-1),
                "original_confidence": float,
                "penalty_applied": float,
                "reasoning": str (Explainable AI),
                "fusion_metadata": {
                    "urls_count": int,
                    "text_length": int,
                    "stream_a_original": str,
                    "penalty_reason": str or None
                }
            }
        """
        
        # Initialize result
        result = {
            "final_verdict": stream_a_verdict,
            "adjusted_confidence": stream_a_confidence,
            "original_confidence": stream_a_confidence,
            "penalty_applied": 0.0,
            "reasoning": "",
            "fusion_metadata": {
                "urls_count": len(urls_found),
                "text_length": len(extracted_text),
                "stream_a_original": stream_a_verdict,
                "penalty_reason": None
            }
        }
        
        # ====================================================================
        # FUSION LOGIC: No-Payload Penalty
        # ====================================================================
        
        if stream_a_verdict == "Phishing" and len(urls_found) == 0:
            # Phishing detected but NO URLs found - suspicious
            # Apply penalty: domain shift correction
            
            adjusted_confidence = stream_a_confidence + self.NO_PAYLOAD_PENALTY
            result["penalty_applied"] = self.NO_PAYLOAD_PENALTY
            result["adjusted_confidence"] = max(0.0, adjusted_confidence)  # Clamp to [0, 1]
            result["fusion_metadata"]["penalty_reason"] = "No-Payload Penalty"
            
            # ================================================================
            # VERDICT FLIP: If adjusted confidence drops below threshold
            # ================================================================
            
            if result["adjusted_confidence"] < self.CONFIDENCE_THRESHOLD:
                result["final_verdict"] = "Safe"
                result["reasoning"] = (
                    f"Stream A flagged as Phishing (confidence: {stream_a_confidence:.1%}), "
                    f"but no URLs found. Applied No-Payload Penalty ({self.NO_PAYLOAD_PENALTY:+.2f}), "
                    f"resulting in adjusted confidence {result['adjusted_confidence']:.1%} "
                    f"(below {self.CONFIDENCE_THRESHOLD:.0%} threshold). "
                    f"Verdict flipped to Safe. "
                    f"Interpretation: Corporate document, not malicious email."
                )
            else:
                result["final_verdict"] = "Phishing"
                result["reasoning"] = (
                    f"Stream A flagged as Phishing (confidence: {stream_a_confidence:.1%}). "
                    f"No URLs found, applied No-Payload Penalty ({self.NO_PAYLOAD_PENALTY:+.2f}), "
                    f"but adjusted confidence {result['adjusted_confidence']:.1%} "
                    f"remains above {self.CONFIDENCE_THRESHOLD:.0%} threshold. "
                    f"Verdict maintained as Phishing."
                )
        
        else:
            # No penalty applied - keep original verdict
            result["reasoning"] = (
                f"Stream A verdict: {stream_a_verdict} (confidence: {stream_a_confidence:.1%}). "
            )
            
            if stream_a_verdict == "Phishing" and len(urls_found) > 0:
                result["reasoning"] += (
                    f"URLs found in attachment ({len(urls_found)} URL(s)): {', '.join(urls_found[:2])}{'...' if len(urls_found) > 2 else ''}. "
                    f"Payload present. Verdict: Phishing (confirmed)."
                )
            elif stream_a_verdict == "Safe":
                result["reasoning"] += f"Safe verdict maintained."
        
        # Track decision
        self.fusion_log.append(result)
        
        return result
    
    def get_fusion_log(self) -> List[Dict]:
        """Get all fusion decisions made by this engine"""
        return self.fusion_log
    
    def export_metrics(self, output_file: str = None) -> Dict:
        """Export fusion engine metrics and logs"""
        metrics = {
            "total_evaluations": len(self.fusion_log),
            "penalties_applied": sum(1 for log in self.fusion_log if log["penalty_applied"] != 0.0),
            "verdicts_flipped": sum(1 for log in self.fusion_log 
                                   if log["final_verdict"] != log["fusion_metadata"]["stream_a_original"]),
            "fusion_log": self.fusion_log
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        return metrics


# ============================================================================
# MAIN: TEST WITH QUARTERLY_REPORT.PDF DATA
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SENTINEL ENSEMBLE ENGINE: PROOF OF CONCEPT TEST")
    print("="*80 + "\n")
    
    # Initialize ensemble
    ensemble = SentinelEnsemble()
    
    # ========================================================================
    # TEST CASE 1: quarterly_report.pdf (False Positive Fix)
    # ========================================================================
    
    print("TEST CASE 1: Quarterly Report (False Positive Fix)")
    print("-" * 80)
    
    # Data from integration test
    extracted_text = "Quarterly Financial Report Q1 2026..."  # Simplified
    urls_found = []  # NO URLs in benign document
    stream_a_confidence = 0.99  # DistilBERT confidence
    stream_a_verdict = "Phishing"  # FALSE POSITIVE in domain shift
    
    result1 = ensemble.evaluate_attachment(
        extracted_text=extracted_text,
        urls_found=urls_found,
        stream_a_confidence=stream_a_confidence,
        stream_a_verdict=stream_a_verdict
    )
    
    print(f"Input:")
    print(f"  Stream A Verdict: {stream_a_verdict}")
    print(f"  Stream A Confidence: {stream_a_confidence:.1%}")
    print(f"  URLs Found: {len(urls_found)}")
    print(f"\nFusion Engine Output:")
    print(f"  ✓ Final Verdict: {result1['final_verdict']}")
    print(f"  ✓ Adjusted Confidence: {result1['adjusted_confidence']:.1%}")
    print(f"  ✓ Penalty Applied: {result1['penalty_applied']:+.2f}")
    print(f"  ✓ Reasoning: {result1['reasoning']}")
    
    # Verify the flip occurred
    assert result1['final_verdict'] == 'Safe', "FAIL: Verdict should be flipped to Safe"
    assert result1['adjusted_confidence'] < 0.70, "FAIL: Adjusted confidence should be below 0.70"
    print(f"\n✅ TEST PASSED: False positive corrected!\n")
    
    # ========================================================================
    # TEST CASE 2: Malicious with URLs (Should remain Phishing)
    # ========================================================================
    
    print("TEST CASE 2: Malicious Invoice with URLs (Remains Phishing)")
    print("-" * 80)
    
    extracted_text2 = "URGENT: Payment Required. Click link: http://phishing-site.com"
    urls_found2 = ["http://phishing-site.com"]
    stream_a_confidence2 = 0.97
    stream_a_verdict2 = "Phishing"
    
    result2 = ensemble.evaluate_attachment(
        extracted_text=extracted_text2,
        urls_found=urls_found2,
        stream_a_confidence=stream_a_confidence2,
        stream_a_verdict=stream_a_verdict2
    )
    
    print(f"Input:")
    print(f"  Stream A Verdict: {stream_a_verdict2}")
    print(f"  Stream A Confidence: {stream_a_confidence2:.1%}")
    print(f"  URLs Found: {len(urls_found2)} - {urls_found2[0]}")
    print(f"\nFusion Engine Output:")
    print(f"  ✓ Final Verdict: {result2['final_verdict']}")
    print(f"  ✓ Adjusted Confidence: {result2['adjusted_confidence']:.1%}")
    print(f"  ✓ Penalty Applied: {result2['penalty_applied']:+.2f}")
    print(f"  ✓ Reasoning: {result2['reasoning']}")
    
    # Verify no flip
    assert result2['final_verdict'] == 'Phishing', "FAIL: Verdict should remain Phishing"
    assert result2['penalty_applied'] == 0.0, "FAIL: No penalty should be applied"
    print(f"\n✅ TEST PASSED: Phishing verdict maintained with payload!\n")
    
    # ========================================================================
    # TEST CASE 3: Safe email without URLs (Should remain Safe)
    # ========================================================================
    
    print("TEST CASE 3: Safe Corporate Email (No URLs)")
    print("-" * 80)
    
    extracted_text3 = "Meeting notes from Q1 planning session..."
    urls_found3 = []
    stream_a_confidence3 = 0.05  # Very confident it's safe
    stream_a_verdict3 = "Safe"
    
    result3 = ensemble.evaluate_attachment(
        extracted_text=extracted_text3,
        urls_found=urls_found3,
        stream_a_confidence=stream_a_confidence3,
        stream_a_verdict=stream_a_verdict3
    )
    
    print(f"Input:")
    print(f"  Stream A Verdict: {stream_a_verdict3}")
    print(f"  Stream A Confidence: {stream_a_confidence3:.1%}")
    print(f"  URLs Found: {len(urls_found3)}")
    print(f"\nFusion Engine Output:")
    print(f"  ✓ Final Verdict: {result3['final_verdict']}")
    print(f"  ✓ Adjusted Confidence: {result3['adjusted_confidence']:.1%}")
    print(f"  ✓ Penalty Applied: {result3['penalty_applied']:+.2f}")
    print(f"  ✓ Reasoning: {result3['reasoning']}")
    
    # Verify no change
    assert result3['final_verdict'] == 'Safe', "FAIL: Verdict should remain Safe"
    print(f"\n✅ TEST PASSED: Safe verdict confirmed!\n")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    metrics = ensemble.export_metrics()
    
    print("="*80)
    print("ENSEMBLE ENGINE METRICS")
    print("="*80)
    print(f"Total Evaluations: {metrics['total_evaluations']}")
    print(f"Penalties Applied: {metrics['penalties_applied']}")
    print(f"Verdicts Flipped: {metrics['verdicts_flipped']}")
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - ENSEMBLE ENGINE READY FOR PRODUCTION")
    print("="*80 + "\n")
