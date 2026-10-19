"""Haze Detection and Severity Classification module.

Responsible for:
- Detecting presence of atmospheric haze, fog, or smog
- Estimating haze density and dark channel statistics
- Severity classification: Clear, Light, Moderate, Dense
"""

from src.detection.density import estimate_haze_density

__all__ = ["estimate_haze_density"]
