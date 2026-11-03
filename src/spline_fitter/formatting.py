"""
Formatting Utilities
====================

Display formatting functions for numbers, durations, and errors.

This module provides consistent, human-readable formatting for
the GUI status bar, log files, and console output.

Numbers shown to the user take the decimal point of the system locale,
which the GUI sets with ``set_decimal_point``; files keep a dot.
"""

from __future__ import annotations

import math

_decimal_point = "."


def set_decimal_point(char: str) -> None:
    """Use this decimal point for every number formatted for display."""
    global _decimal_point
    _decimal_point = char or "."


def decimal_point() -> str:
    """The decimal point of numbers formatted for display."""
    return _decimal_point


def number(value: float, spec: str = "g") -> str:
    """``format(value, spec)`` with the display decimal point."""
    return format(value, spec).replace(".", _decimal_point)


def format_error(value: float) -> str:
    """
    Format an error value for display.
    
    Small and large values are shown in e-notation, like the readouts
    and reports. Values in the normal range use fixed-point notation.
    
    Args:
        value: The error value to format
        
    Returns:
        Human-readable string representation
        
    Examples:
        >>> format_error(0.0000123)
        '1.230e-05'
        >>> format_error(0.00456)
        '0.004560'
        >>> format_error(12345.6)
        '1.235e+04'
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
        return number(val, ".3e")

    return number(val, ".6f")


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
        return f"{number(seconds, '.2f')} s"

    minutes, secs = divmod(seconds, 60.0)

    if minutes < 60:
        return f"{int(minutes)}m {int(secs)}s"

    hours, minutes = divmod(int(minutes), 60)
    return f"{int(hours)}h {int(minutes)}m {int(secs)}s"


def format_count(value: int, singular: str, plural: str | None = None) -> str:
    """
    Format a count with appropriate singular/plural noun.
    
    Args:
        value: The count
        singular: Singular form of the noun
        plural: Plural form (defaults to singular + 's')
        
    Returns:
        Formatted string like "1 point" or "1,500 points"
    """
    if plural is None:
        plural = singular + "s"

    noun = singular if value == 1 else plural
    # Group thousands with whichever of comma and dot is not the decimal point
    separator = "." if _decimal_point == "," else ","
    return f"{value:,} {noun}".replace(",", separator)


def format_percentage(value: float, precision: int = 1) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: Value to format (0.5 = 50%)
        precision: Decimal places to show
        
    Returns:
        Formatted percentage string
    """
    return f"{number(value * 100, f'.{precision}f')}%"


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


def file_stem(name: str, default: str = "airfoil") -> str:
    """A name, such as an airfoil's, as a file name on every system: the
    characters Windows forbids (path separators among them) become
    underscores, and trailing dots and spaces go. Unlike
    :func:`sanitize_filename` it keeps spaces and other letters, so
    "NACA 2412 t/c 12%" becomes "NACA 2412 t_c 12%"."""
    stem = "".join("_" if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in name)
    return stem.strip().rstrip(". ") or default
