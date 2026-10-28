"""
Airfoil Spline Fitter - Main Entry Point
=========================================

Launch the application with:
    python -m spline_fitter

Or with command-line options:
    python -m spline_fitter --bbox-scale 5.0

Command Line Arguments:
    --bbox-scale <float>    Set the bounding box scale factor (default: 3.0)
    --help                  Show this help message
"""

from __future__ import annotations

import argparse
import sys


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="spline_fitter",
        description="Interactive B-spline fitting tool for airfoil coordinates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m spline_fitter
    python -m spline_fitter --bbox-scale 5.0
        """,
    )

    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=None,
        metavar="SCALE",
        help="Bounding box scale factor for control point bounds (default: 3.0)",
    )

    return parser.parse_args()


def main(bbox_scale: float | None = None) -> int:
    """
    Run the spline fitter application.
    
    Args:
        bbox_scale: Optional bounding box scale override
        
    Returns:
        Exit code (0 for success)
    """
    from PyQt6.QtWidgets import QApplication

    from .gui.ui import SplineFitterWindow

    # On Windows the taskbar shows Python's icon unless the app has its own ID
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AirfoilSplineFitter")
        except Exception:
            pass

    # Create application
    app = QApplication(sys.argv)

    # Create and configure main window
    window = SplineFitterWindow()

    # Apply command-line overrides
    if bbox_scale is not None:
        try:
            window._spin_bbox.setValue(float(bbox_scale))
        except Exception:
            pass

    # Show window
    # window.resize(1000, 600)
    window.showMaximized()

    # Run event loop
    return app.exec()


def cli_entry() -> None:
    """Command-line interface entry point."""
    args = parse_arguments()
    sys.exit(main(bbox_scale=args.bbox_scale))


if __name__ == "__main__":
    cli_entry()
