#!/usr/bin/env python3
"""
SENTINEL: Multi-Modal Ensemble & SHAP Explainability Engine (Phase 5)

Purpose:
    Fuses predictions from Streams A (Text), B (URL), and C (Attachment) into
    a final ensemble verdict with SHAP-based explainability.

Architecture:
    1. Weighted Ensemble Fusion (30% Text + 30% URL + 40% Attachment)
    2. Critical Threat Override (Attachment structural threats bypass other signals)
    3. SHAP Kernel Explainer for feature importance attribution
    4. Confidence-based decision thresholding (>= 0.70 = Phishing)

Author: Sentinel XAI Team
Date: April 8, 2026
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass, asdict

# SHAP library for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[!] SHAP not installed. Install via: pip install shap")


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass
class StreamPrediction:
    """Single stream prediction result"""
    stream_name: str  # "Text" or "URL" or "Attachment"
    confidence: float  # [0.0, 1.0]
    threat_indicators: List[str]  # Detected threats
    
    def to_dict(self):
        return asdict(self)


@dataclass
class EnsembleResult:
    """Final ensemble fusion result"""
    final_verdict: str  # "Phishing" or "Benign"
    final_confidence: float  # [0.0, 1.0]
    stream_a_conf: float
    stream_b_conf: float
    stream_c_conf: float
    critical_threat_triggered: bool
    ensemble_score: float
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# ENSEMBLE ENGINE
# ============================================================================

class SentinelEnsemble:
    """
    Multi-Modal Ensemble Fusion Engine for Sentinel Phishing Detection.
    
    Combines independent predictions from three streams:
    - Stream A: DistilBERT-based text analysis
    - Stream B: XGBoost-based URL lexical features  
    - Stream C: Zero-shot NLP + structural attachment analysis
    
    Features:
    - Critical threat override (Stream C structural threats bypass other signals)
    - Dynamic weight reallocation when streams are unavailable (None)
    - SHAP explainability for isolated and combined predictions
    """
    
    # Base weights for reference
    BASE_WEIGHTS = {
        "text": 0.30,      # Stream A: DistilBERT
        "url": 0.30,       # Stream B: XGBoost
        "attachment": 0.40  # Stream C: Zero-shot + Structural
    }
    
    def __init__(self, weights: Dict[str, float] = None, threshold: float = 0.70):
        """
        Initialize Sentinel Ensemble with SHAP KernelExplainer for TRUE game-theoretic explanations.
        
        Args:
            weights: Stream weighting dict (default: 30% Text, 30% URL, 40% Attachment)
            threshold: Classification threshold (default: 0.70)
        """
        self.base_weights = weights or self.BASE_WEIGHTS.copy()
        self.threshold = threshold
        self.logger = self._setup_logger()
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.base_weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        
        self.logger.info(f"[INIT] Sentinel Ensemble initialized with base weights: {self.base_weights}")
        
        # =====================================================================
        # INITIALIZE SHAP EXPLAINER
        # =====================================================================
        self.logger.info("[INIT] Initializing SHAP KernelExplainer...")
        
        # Synthetic background dataset: 3 samples representing different risk profiles
        # Format: [stream_a_conf, stream_b_conf, stream_c_conf]
        self.background_data = np.array([
            [0.5, 0.5, 0.5],  # Neutral: medium confidence across all streams
            [0.2, 0.2, 0.2],  # Conservative: low confidence (safe emails)
            [0.8, 0.8, 0.8],  # Aggressive: high confidence (risky emails)
        ])
        
        # Initialize SHAP explainer with the weighted score function
        try:
            self.shap_explainer = shap.KernelExplainer(
                model=self._compute_raw_weighted_score,
                data=self.background_data
            )
            self.logger.info("[INIT] SHAP KernelExplainer initialized successfully")
        except Exception as e:
            raise RuntimeError(f"[INIT] FAILED to initialize SHAP explainer: {e}. This is FATAL - check SHAP installation.")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("SentinelEnsemble")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _compute_raw_weighted_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the raw weighted ensemble score for a batch of samples.
        
        This is the function that SHAP will call repeatedly to understand
        how each stream's contribution affects the final score.
        
        Args:
            X: Array of shape (n_samples, 3) where each row is [stream_a, stream_b, stream_c]
               Values range [0.0, 1.0] representing confidence scores
        
        Returns:
            Array of shape (n_samples,) with ensemble scores (before critical override)
        """
        predictions = []
        for sample in X:
            stream_a_conf = sample[0]  # Always present
            stream_b_conf = sample[1]  # Always present
            stream_c_conf = sample[2]  # Always present
            
            # Compute weighted fusion (simplified, no critical override for SHAP)
            # This allows SHAP to see pure additive effects of each stream
            ensemble_score = (
                self.base_weights["text"] * stream_a_conf +
                self.base_weights["url"] * stream_b_conf +
                self.base_weights["attachment"] * stream_c_conf
            )
            predictions.append(ensemble_score)
        
        return np.array(predictions)
    
    def _calculate_dynamic_weights(
        self,
        stream_a_conf: float = None,
        stream_b_conf: float = None,
        stream_c_conf: float = None
    ) -> Dict[str, float]:
        """
        Calculate dynamically reallocated weights when streams are unavailable (None).
        
        Logic:
            - If all streams present: use base weights
            - If a stream is None: reallocate its weight proportionally to active streams
        
        Example:
            Base: A=0.30, B=0.30, C=0.40 (total=1.0)
            If only B and C present:
                Active base sum: 0.30 + 0.40 = 0.70
                B new weight: 0.30 / 0.70 = 0.429
                C new weight: 0.40 / 0.70 = 0.571
        
        Args:
            stream_a_conf: Stream A confidence or None
            stream_b_conf: Stream B confidence or None
            stream_c_conf: Stream C confidence or None
        
        Returns:
            Dict with dynamically calculated weights
        """
        # Determine which streams are active
        active_streams = {}
        if stream_a_conf is not None:
            active_streams["text"] = self.base_weights["text"]
        if stream_b_conf is not None:
            active_streams["url"] = self.base_weights["url"]
        if stream_c_conf is not None:
            active_streams["attachment"] = self.base_weights["attachment"]
        
        # If no streams active, return equal weights (shouldn't happen)
        if not active_streams:
            raise ValueError("At least one stream must be provided (not None)")
        
        # Calculate total weight of active streams
        total_active_weight = sum(active_streams.values())
        
        # Normalize weights to sum to 1.0
        dynamic_weights = {k: v / total_active_weight for k, v in active_streams.items()}
        
        # Log the reallocation
        active_count = len(active_streams)
        if active_count < 3:
            self.logger.info(f"[DYNAMIC_WEIGHTS] {active_count}/3 streams active. Reallocating weights:")
            for stream, weight in self.base_weights.items():
                if stream in dynamic_weights:
                    self.logger.info(f"  {stream.upper()}: {weight:.2%} → {dynamic_weights[stream]:.2%}")
                else:
                    self.logger.info(f"  {stream.upper()}: {weight:.2%} → INACTIVE (None)")
        
        return dynamic_weights
    
    def fuse_predictions(
        self,
        stream_a_conf: float = None,
        stream_b_conf: float = None,
        stream_c_conf: float = None,
        c_structural_threat_flag: bool = False
    ) -> EnsembleResult:
        """
        Fuse predictions from three streams with dynamic weight reallocation and critical threat override.
        
        Handles isolated data (single stream) by dynamically reallocating weights.
        
        Mathematical Fusion:
            ensemble_score = (w_A * stream_a_conf) + (w_B * stream_b_conf) + (w_C * stream_c_conf)
            where weights are dynamically calculated if any stream is None
        
        Critical Override:
            If c_structural_threat_flag is True and Stream C is active:
                ensemble_score = 0.90 (automatically classified as Phishing)
        
        Args:
            stream_a_conf: Text model confidence [0.0, 1.0] or None if unavailable
            stream_b_conf: URL model confidence [0.0, 1.0] or None if unavailable
            stream_c_conf: Attachment model confidence [0.0, 1.0] or None if unavailable
            c_structural_threat_flag: Critical structural threat detected in Stream C
        
        Returns:
            EnsembleResult with final verdict and confidence
        """
        # Validate that at least one stream is provided
        if stream_a_conf is None and stream_b_conf is None and stream_c_conf is None:
            raise ValueError("At least one stream must be provided (not None)")
        
        # Validate input ranges for non-None streams
        for conf, name in [(stream_a_conf, "A"), (stream_b_conf, "B"), (stream_c_conf, "C")]:
            if conf is not None and not 0.0 <= conf <= 1.0:
                raise ValueError(f"Stream {name} confidence must be in [0.0, 1.0] or None, got {conf}")
        
        # =====================================================================
        # PHASE 1: Dynamic Weight Calculation & Weighted Ensemble Fusion
        # =====================================================================
        dynamic_weights = self._calculate_dynamic_weights(stream_a_conf, stream_b_conf, stream_c_conf)
        
        # Calculate ensemble score using only active streams
        ensemble_score = 0.0
        log_msg_parts = []
        
        if stream_a_conf is not None:
            contribution_a = dynamic_weights["text"] * stream_a_conf
            ensemble_score += contribution_a
            log_msg_parts.append(f"A: {stream_a_conf:.4f}×{dynamic_weights['text']:.2%}={contribution_a:.4f}")
        
        if stream_b_conf is not None:
            contribution_b = dynamic_weights["url"] * stream_b_conf
            ensemble_score += contribution_b
            log_msg_parts.append(f"B: {stream_b_conf:.4f}×{dynamic_weights['url']:.2%}={contribution_b:.4f}")
        
        if stream_c_conf is not None:
            contribution_c = dynamic_weights["attachment"] * stream_c_conf
            ensemble_score += contribution_c
            log_msg_parts.append(f"C: {stream_c_conf:.4f}×{dynamic_weights['attachment']:.2%}={contribution_c:.4f}")
        
        self.logger.info(f"[FUSION] Ensemble score: {ensemble_score:.4f}")
        self.logger.info(f"[FUSION] Contributing terms: {' + '.join(log_msg_parts)}")
        
        # =====================================================================
        # PHASE 2: Critical Threat Override
        # =====================================================================
        final_confidence = ensemble_score
        critical_override_triggered = False
        
        if stream_c_conf is not None and c_structural_threat_flag:
            self.logger.warning("[CRITICAL_THREAT] Structural threat detected in Stream C!")
            self.logger.warning("[OVERRIDE] Forcing confidence to 0.90 (Auto-Phishing)")
            final_confidence = 0.90
            critical_override_triggered = True
        
        # =====================================================================
        # PHASE 3: Classification Decision
        # =====================================================================
        if final_confidence >= self.threshold:
            verdict = "Phishing"
        else:
            verdict = "Benign"
        
        self.logger.info(f"[DECISION] Final confidence: {final_confidence:.4f} → Verdict: {verdict}")
        
        return EnsembleResult(
            final_verdict=verdict,
            final_confidence=final_confidence,
            stream_a_conf=stream_a_conf if stream_a_conf is not None else 0.0,
            stream_b_conf=stream_b_conf if stream_b_conf is not None else 0.0,
            stream_c_conf=stream_c_conf if stream_c_conf is not None else 0.0,
            critical_threat_triggered=critical_override_triggered,
            ensemble_score=ensemble_score
        )
    
    def explain_decision(
        self,
        stream_a_conf: float = None,
        stream_b_conf: float = None,
        stream_c_conf: float = None,
        c_structural_threat_flag: bool = False,
        c_threat_description: str = None
    ) -> Dict:
        """
        Generate REAL SHAP game-theoretic explanations for the ensemble decision.
        
        Uses SHAP (SHapley Additive exPlanations) Kernel method to compute how much
        each stream's contribution pushed the final ensemble score UP or DOWN from
        the baseline expectation.
        
        CRITICAL: This uses TRUE SHAP, not weighted arithmetic.
        De-blackboxed: Includes threat descriptions for structural analysis.
        
        Args:
            stream_a_conf: Text analysis confidence [0.0, 1.0] or None → converted to 0.0
            stream_b_conf: URL analysis confidence [0.0, 1.0] or None → converted to 0.0
            stream_c_conf: Attachment analysis confidence [0.0, 1.0] or None → converted to 0.0
            c_structural_threat_flag: Structural threat flag (triggers override)
            c_threat_description: Description of detected structural threats (for de-blackboxing)
        
        Returns:
            Dict with:
            - shap_values: np array of SHAP values for each stream
            - base_value: Expected value (model output on background data)
            - model_output: Actual ensemble score
            - feature_names: ["Stream_A_Text", "Stream_B_URL", "Stream_C_Attachment"]
            - explanation: Human-readable breakdown of how each stream affected the score
            - threat_description: Reason for Stream C contribution (de-blackboxing)
        
        Raises:
            RuntimeError: If SHAP computation fails (HARD EXCEPTION, no fallback)
        """
        self.logger.info("[XAI] Computing TRUE SHAP explanations (Kernel method)...")
        
        # Convert None to 0.0 for consistent feature vector
        a_conf = stream_a_conf if stream_a_conf is not None else 0.0
        b_conf = stream_b_conf if stream_b_conf is not None else 0.0
        c_conf = stream_c_conf if stream_c_conf is not None else 0.0
        
        # Create feature vector for this prediction
        # Format: [stream_a_conf, stream_b_conf, stream_c_conf]
        feature_vector = np.array([[a_conf, b_conf, c_conf]])
        feature_names = ["Stream_A_Text", "Stream_B_URL", "Stream_C_Attachment"]
        
        try:
            # Compute TRUE SHAP values using KernelExplainer
            self.logger.info("[XAI] Calling SHAP KernelExplainer.shap_values()...")
            shap_values = self.shap_explainer.shap_values(feature_vector)
            base_value = self.shap_explainer.expected_value
            
            self.logger.info("[XAI] SHAP computation successful")
            
            # Extract values (handle list or array return types)
            if isinstance(shap_values, list):
                shap_array = np.array(shap_values[0])  # First sample
            else:
                shap_array = shap_values[0]  # First sample
            
            # Compute model output (ensemble score)
            model_output = self._compute_raw_weighted_score(feature_vector)[0]
            
            # Verify math: base_value + sum(shap_values) should ≈ model_output
            shap_sum = float(np.sum(shap_array))
            computed_output = float(base_value) + shap_sum
            
            self.logger.debug(f"[XAI] Base value: {base_value:.6f}")
            self.logger.debug(f"[XAI] Sum of SHAP values: {shap_sum:.6f}")
            self.logger.debug(f"[XAI] Expected model output: {computed_output:.6f}")
            self.logger.debug(f"[XAI] Actual model output: {model_output:.6f}")
            
            # Build human-readable explanation
            explanation_text = f"""
BASE VALUE (Expected ensemble score): {base_value:.6f}

STREAM CONTRIBUTIONS (How each stream shifted the score):
"""
            
            # Sort by absolute SHAP value for priority
            shap_indices = np.argsort(np.abs(shap_array))[::-1]
            
            for rank, idx in enumerate(shap_indices, 1):
                stream_name = feature_names[idx]
                stream_input = feature_vector[0, idx]
                shap_value = shap_array[idx]
                direction = "↑ INCREASED" if shap_value > 0 else "↓ DECREASED"
                
                contribution_line = f"  {rank}. {stream_name:25s} = {stream_input:.4f} → {direction} score by {abs(shap_value):+.6f}"
                
                # De-blackboxing: Add threat description for Stream C if available
                if stream_name == "Stream_C_Attachment" and c_threat_description:
                    contribution_line += f"\n       Reason: {c_threat_description}"
                
                explanation_text += contribution_line + "\n"
            
            explanation_text += f"\nFINAL ENSEMBLE SCORE: {model_output:.6f}"
            
            # Add critical threat override explanation
            override_explanation = ""
            if c_structural_threat_flag:
                override_explanation = """
CRITICAL THREAT OVERRIDE TRIGGERED:
  ⚠️  Structural threat detected in attachment (Malformed PDF, JavaScript, Auto-launch, etc.)
  ⚠️  This indicator OVERRIDES standard ML weights
  ⚠️  Final verdict forced to PHISHING (0.90 confidence) due to critical structural threat
  ⚠️  This is a de-blackboxed decision: structural threats are non-negotiable indicators"""
            
            if model_output >= self.threshold:
                explanation_text += f" → PHISHING (threshold: {self.threshold})"
                if c_structural_threat_flag:
                    explanation_text += override_explanation
            else:
                explanation_text += f" → BENIGN (threshold: {self.threshold})"
                if c_structural_threat_flag:
                    explanation_text += override_explanation
            
            self.logger.info(explanation_text)
            
            return {
                "method": "TRUE_SHAP_KernelExplainer",
                "base_value": float(base_value),
                "model_output": float(model_output),
                "shap_values": shap_array.tolist(),
                "feature_names": feature_names,
                "input_values": feature_vector[0].tolist(),
                "explanation": explanation_text,
                "structural_threat_override": c_structural_threat_flag,
                "threat_description": c_threat_description if c_threat_description else "None",
                "contributions": {
                    feature_names[i]: {
                        "input": float(feature_vector[0, i]),
                        "shap_value": float(shap_array[i]),
                        "direction": "increases" if shap_array[i] > 0 else "decreases"
                    }
                    for i in range(len(feature_names))
                }
            }
        
        except Exception as e:
            # HARD EXCEPTION - user wants to debug SHAP failures
            error_msg = f"""
[XAI] ===== SHAP COMPUTATION FAILURE (HARD ERROR) =====
[XAI] Exception: {type(e).__name__}: {str(e)}
[XAI] Feature vector: {feature_vector}
[XAI] Feature names: {feature_names}
[XAI] THIS IS NOT A FALLBACK - You must debug this SHAP issue
[XAI] Check: SHAP version, numpy compatibility, background data shape
[XAI] =====================================================
"""
            self.logger.error(error_msg)
            raise RuntimeError(f"SHAP computation FAILED and there is NO FALLBACK. Debug: {error_msg}") from e
    
    def _compute_raw_weighted_score(self, feature_vectors):
        """
        Compute the raw weighted ensemble score WITHOUT applying threshold.
        Used internally by SHAP explainer to understand how scores are computed.
        
        Args:
            feature_vectors: np array of shape (n_samples, 3) with [stream_a, stream_b, stream_c]
        
        Returns:
            np array of raw scores [0.0, 1.0] for each sample (BEFORE threshold application)
        """
        scores = []
        for feature_vector in feature_vectors:
            a_conf = float(feature_vector[0]) if len(feature_vector) > 0 else 0.0
            b_conf = float(feature_vector[1]) if len(feature_vector) > 1 else 0.0
            c_conf = float(feature_vector[2]) if len(feature_vector) > 2 else 0.0
            
            # Apply fixed weights: 30% text, 30% URL, 40% attachment
            raw_score = (0.30 * a_conf) + (0.30 * b_conf) + (0.40 * c_conf)
            scores.append(np.clip(raw_score, 0.0, 1.0))
        
        return np.array(scores)


# ============================================================================
# VALIDATION & DEMONSTRATION
# ============================================================================

def run_validation_scenarios():
    """
    Run three test scenarios to validate the ensemble:
    1. Safe Email (low risk across all streams)
    2. Text-Heavy Phishing (high text signal, moderate others)
    3. Attachment-Heavy Phishing (high attachment threat)
    """
    
    ensemble = SentinelEnsemble()
    
    print("\n" + "="*100)
    print("SENTINEL PHASE 5: ENSEMBLE & SHAP VALIDATION")
    print("="*100)
    
    # Test Scenario 1: Safe Email
    print("\n[TEST 1] Safe Email")
    print("-" * 100)
    result1 = ensemble.fuse_predictions(
        stream_a_conf=0.25,  # Low text risk
        stream_b_conf=0.30,  # Low URL risk
        stream_c_conf=0.20,  # Low attachment risk
        c_structural_threat_flag=False
    )
    print(f"\nResult: {result1.final_verdict}")
    print(f"Final Confidence: {result1.final_confidence:.4f}")
    print(f"Ensemble Score: {result1.ensemble_score:.4f}")
    
    explanation1 = ensemble.explain_decision(0.25, 0.30, 0.20, False)
    print(f"\nXAI Explanation (TRUE SHAP):")
    print(f"  Method: {explanation1['method']}")
    print(f"  Base Value (Expected Score): {explanation1['base_value']:.6f}")
    print(f"  Model Output (Actual Score): {explanation1['model_output']:.6f}")
    print(f"  SHAP Values per Stream:")
    for fname, shap_val in zip(explanation1['feature_names'], explanation1['shap_values']):
        direction = "↑ INCREASES" if shap_val >= 0 else "↓ DECREASES"
        print(f"    {fname:25s}: {shap_val:+.6f} {direction} final score")
    
    # Test Scenario 2: Text-Heavy Phishing
    print("\n\n[TEST 2] Text-Heavy Phishing")
    print("-" * 100)
    result2 = ensemble.fuse_predictions(
        stream_a_conf=0.85,  # High text risk (phishing keywords, urgency)
        stream_b_conf=0.40,  # Moderate URL risk
        stream_c_conf=0.35,  # Low attachment risk
        c_structural_threat_flag=False
    )
    print(f"\nResult: {result2.final_verdict}")
    print(f"Final Confidence: {result2.final_confidence:.4f}")
    print(f"Ensemble Score: {result2.ensemble_score:.4f}")
    
    explanation2 = ensemble.explain_decision(0.85, 0.40, 0.35, False)
    print(f"\nXAI Explanation (TRUE SHAP):")
    print(f"  Method: {explanation2['method']}")
    print(f"  Base Value (Expected Score): {explanation2['base_value']:.6f}")
    print(f"  Model Output (Actual Score): {explanation2['model_output']:.6f}")
    print(f"  SHAP Values per Stream:")
    for fname, shap_val in zip(explanation2['feature_names'], explanation2['shap_values']):
        direction = "↑ INCREASES" if shap_val >= 0 else "↓ DECREASES"
        print(f"    {fname:25s}: {shap_val:+.6f} {direction} final score")
    
    # Test Scenario 3: Attachment-Heavy Phishing with Critical Threat
    print("\n\n[TEST 3] Attachment-Heavy Phishing (Critical Threat)")
    print("-" * 100)
    result3 = ensemble.fuse_predictions(
        stream_a_conf=0.40,  # Moderate text risk
        stream_b_conf=0.45,  # Moderate URL risk
        stream_c_conf=0.80,  # High attachment risk with structural threats
        c_structural_threat_flag=True  # CRITICAL: Structural threat detected
    )
    print(f"\nResult: {result3.final_verdict}")
    print(f"Final Confidence: {result3.final_confidence:.4f}")
    print(f"Ensemble Score (before override): {result3.ensemble_score:.4f}")
    print(f"Critical Threat Override: {result3.critical_threat_triggered}")
    
    explanation3 = ensemble.explain_decision(0.40, 0.45, 0.80, True)
    print(f"\nXAI Explanation (TRUE SHAP):")
    print(f"  Method: {explanation3['method']}")
    print(f"  Base Value (Expected Score): {explanation3['base_value']:.6f}")
    print(f"  Model Output (Actual Score): {explanation3['model_output']:.6f}")
    print(f"  SHAP Values per Stream:")
    for fname, shap_val in zip(explanation3['feature_names'], explanation3['shap_values']):
        direction = "↑ INCREASES" if shap_val >= 0 else "↓ DECREASES"
        print(f"    {fname:25s}: {shap_val:+.6f} {direction} final score")
    
    # Summary
    print("\n\n" + "="*100)
    print("VALIDATION SUMMARY")
    print("="*100)
    print(f"\nTest 1 (Safe Email):")
    print(f"  Final Verdict: {result1.final_verdict}")
    print(f"  Confidence: {result1.final_confidence:.4f}")
    print(f"  Status: {'PASS' if result1.final_verdict == 'Benign' else 'FAIL'}")
    
    print(f"\nTest 2 (Text Phishing):")
    print(f"  Final Verdict: {result2.final_verdict}")
    print(f"  Confidence: {result2.final_confidence:.4f}")
    print(f"  Status: {'PASS' if result2.final_verdict == 'Phishing' else 'FAIL'}")
    
    print(f"\nTest 3 (Attachment Phishing + Critical Threat):")
    print(f"  Final Verdict: {result3.final_verdict}")
    print(f"  Confidence: {result3.final_confidence:.4f}")
    print(f"  Critical Override: {result3.critical_threat_triggered}")
    print(f"  Status: {'PASS' if result3.final_verdict == 'Phishing' and result3.final_confidence >= 0.90 else 'FAIL'}")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    run_validation_scenarios()
