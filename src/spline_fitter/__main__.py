"""
Airfoil Spline Fitter - Main Entry Point
=========================================

Launch the application with:
    python -m spline_fitter          (installed)
    python -m src.spline_fitter      (from a source checkout)

Command Line Arguments:
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
        """,
    )

    return parser.parse_args()


def main() -> int:
    """
    Run the spline fitter application.

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

    # Show window
    # window.resize(1000, 600)
    window.showMaximized()

    # Run event loop
    return app.exec()


def cli_entry() -> None:
    """Command-line interface entry point."""
    parse_arguments()
    sys.exit(main())


if __name__ == "__main__":
    cli_entry()
