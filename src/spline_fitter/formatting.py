"""
Formatting Utilities
====================

Display formatting functions for numbers, durations, and errors.

This module provides consistent, human-readable formatting for
the GUI status bar, log files, and console output.
"""

from __future__ import annotations

import math
from typing import Optional


def format_error(value: float) -> str:
    """
    Format an error value for display.
    
    Small and large values are shown in scientific notation as
    "mantissa × 10^exp" for readability. Values in the normal
    range use fixed-point notation.
    
    Args:
        value: The error value to format
        
    Returns:
        Human-readable string representation
        
    Examples:
        >>> format_error(0.0000123)
        '1.230×10^-5'
        >>> format_error(0.00456)
        '0.004560'
        >>> format_error(12345.6)
        '1.235×10^4'
    """
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    
    if not math.isfinite(val):
        return str(val)
    
    abs_val = abs(val)
    
    # Use scientific notation for very small or very large values
    if 0 < abs_val < 1e-3 or abs_val >= 1e4:
        exponent = int(math.floor(math.log10(abs_val)))
        mantissa = val / (10 ** exponent)
        return f"{mantissa:.3f}×10^{exponent}"
    
    return f"{val:.6f}"


def format_duration(seconds: float) -> str:
    """
    Format a duration for human-readable display.
    
    Automatically selects appropriate units (microseconds, milliseconds,
    seconds, minutes, hours) based on magnitude.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string with units
        
    Examples:
        >>> format_duration(0.000123)
        '123 µs'
        >>> format_duration(0.456)
        '456 ms'
        >>> format_duration(12.34)
        '12.34 s'
        >>> format_duration(123.4)
        '2m 3s'
        >>> format_duration(3723.0)
        '1h 2m 3s'
    """
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    
    if seconds < 1.0:
        return f"{seconds * 1e3:.0f} ms"
    
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    
    minutes, secs = divmod(seconds, 60.0)
    
    if minutes < 60:
        return f"{int(minutes)}m {int(secs)}s"
    
    hours, minutes = divmod(int(minutes), 60)
    return f"{int(hours)}h {int(minutes)}m {int(secs)}s"


def format_count(value: int, singular: str, plural: Optional[str] = None) -> str:
    """
    Format a count with appropriate singular/plural noun.
    
    Args:
        value: The count
        singular: Singular form of the noun
        plural: Plural form (defaults to singular + 's')
        
    Returns:
        Formatted string like "1 point" or "5 points"
    """
    if plural is None:
        plural = singular + "s"
    
    noun = singular if value == 1 else plural
    return f"{value:,} {noun}"


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: Value to format (0.5 = 50%)
        precision: Decimal places to show
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


def sanitize_filename(name: str) -> str:
    """
    Convert a string to a safe filename.
    
    Replaces characters that are invalid in filenames with underscores.
    
    Args:
        name: The string to sanitize
        
    Returns:
        Filesystem-safe version of the string
    """
    # Characters that are valid in filenames
    valid_chars = set("abcdefghijklmnopqrstuvwxyz"
                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                      "0123456789-_.")
    
    result = []
    for char in name:
        if char in valid_chars:
            result.append(char)
        else:
            result.append("_")
    
    return "".join(result)
