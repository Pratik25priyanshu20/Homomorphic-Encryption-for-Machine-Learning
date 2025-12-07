"""
Lightweight helpers for reasoning about CKKS depth and scale budgets.

These utilities give quick, conservative estimates to catch obvious
configuration mistakes (e.g., picking coeff_mod_bit_sizes that do not
leave enough levels for your planned multiplicative depth).
"""

from typing import Iterable


def estimate_levels(coeff_mod_bit_sizes: Iterable[int], scale_bits: int) -> int:
    """
    Roughly estimate how many multiplications a circuit can sustain before
    exhausting the modulus chain, given a fixed scale.

    This is an approximation: we divide the total available bits by the
    scale bits and subtract one level for safety. It is sufficient as a
    guardrail in tests and config checks.
    """
    total_bits = sum(int(b) for b in coeff_mod_bit_sizes)
    if scale_bits <= 0:
        raise ValueError("scale_bits must be positive")

    # Subtract one level to keep a safety margin.
    return max(0, total_bits // scale_bits - 1)


def assert_depth_budget(required_muls: int, coeff_mod_bit_sizes: Iterable[int], scale_bits: int) -> None:
    """
    Raise if the estimated available levels are below the required multiplicative depth.
    """
    available = estimate_levels(coeff_mod_bit_sizes, scale_bits)
    if required_muls > available:
        raise ValueError(
            f"Insufficient depth: need {required_muls} muls but only ~{available} levels "
            f"for scale_bits={scale_bits} and coeff_mod_bit_sizes={list(coeff_mod_bit_sizes)}"
        )
