#!/usr/bin/env python3
"""
Unified Phishing Classifier compatibility entrypoint.

This module preserves the original project structure shown in the thesis
while delegating to the current SentinelCore implementation.
"""

from sentinel_core import SentinelCore


class UnifiedPhishingClassifier(SentinelCore):
    """Compatibility alias for the unified ensemble classifier."""


if __name__ == "__main__":
    print("[INFO] Sentinel Unified Phishing Classifier")
    classifier = UnifiedPhishingClassifier()
    print("[OK] Unified classifier initialized successfully")