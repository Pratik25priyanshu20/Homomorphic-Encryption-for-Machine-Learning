"""
HE-safe preprocessing helpers to keep feature magnitudes stable under CKKS.
"""

import numpy as np

# ================================
# HE-SAFE PREPROCESSING CONSTANTS
# ================================
HE_FEATURE_SCALE = 0.25    # reduces feature magnitude → more CKKS-stable
HE_CLIP_MIN = -3.0         # avoid outliers causing saturation
HE_CLIP_MAX = 3.0          # safe range for CKKS + polynomial activations
HE_NORMALIZE = True        # optional: normalize vector length (L2 norm)


def he_preprocess(x_scaled: np.ndarray) -> np.ndarray:
    """
    Full HE-safe preprocessing pipeline.
    Input: already StandardScaler-transformed features.
    Output: scaled, clipped, optionally L2-normalized features.
    """
    x_he = x_scaled * HE_FEATURE_SCALE
    x_he = np.clip(x_he, HE_CLIP_MIN, HE_CLIP_MAX)

    if HE_NORMALIZE:
        norm = np.linalg.norm(x_he) + 1e-8
        x_he = x_he / norm

    return x_he
