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
from typing import NoReturn


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


def main() -> NoReturn:
    """
    Run the spline fitter application, then end the process with the event
    loop's exit code (0 for success).

    It does not return: PyQt and pyqtgraph can crash while the interpreter
    tears down Qt objects after a clean exit (pyqtgraph's own notes say so),
    so the process ends before that teardown, once the window is destroyed
    (which saves its settings) and logging and the output are flushed.
    """
    import gc
    import logging
    import os

    from PyQt6.QtCore import QEvent
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
    code = app.exec()

    # Destroy the window while the application still exists (its QSettings
    # writes reach the store when they are destroyed with it)
    window.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    del window
    gc.collect()
    # Exit handlers are skipped (a library's handler can crash once the window
    # is gone; pyqtgraph's cleanup already ran when the application quit); the
    # application registers none, and logging is flushed here instead
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def cli_entry() -> None:
    """Command-line interface entry point."""
    parse_arguments()
    sys.exit(main())


if __name__ == "__main__":
    cli_entry()
