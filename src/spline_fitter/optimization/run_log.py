"""
Optimizer Run Logs
==================

CSV logs of optimization progress for post-analysis.
"""

from __future__ import annotations

import os
from datetime import datetime

from ..formatting import sanitize_filename


def default_log_dir() -> str | None:
    """
    Return the directory for optimizer logs, or None: logs are opt-in.

    ``SPLINE_FITTER_LOG_DIR`` names it when set and non-empty; otherwise no
    log file is written.
    """
    return os.environ.get("SPLINE_FITTER_LOG_DIR") or None


class OptimizerErrorLogger:
    """
    Logs optimization progress to CSV files for post-analysis.
    
    Each run generates a separate timestamped log file containing
    iteration number, evaluation count, and error value.
    
    Example log file:
        # Optimizer Error Log
        # Run: SD7080
        # Optimizer: de
        # Start Time: 2025-12-03T15:30:27
        iteration,evaluation,error
        1,15,0.002345678901
        2,30,0.001234567890
        ...
    """

    def __init__(self, run_name: str = "unnamed", log_dir: str | None = None):
        """
        Initialize a new logger for an optimization run.

        Args:
            run_name: Descriptive name for this run (e.g., airfoil name)
            log_dir: Directory for log files (default: default_log_dir(),
                resolved when the logger is made; None writes no file)
        """
        self.run_name = run_name
        self.start_time = datetime.now()
        self.log_dir = log_dir or default_log_dir()

        self.iterations: list[int] = []
        self.evaluations: list[int] = []
        self.errors: list[float] = []
        self.elapsed_times: list[float] = []

        self._file_path: str | None = None

    def log(self, iteration: int, evaluation: int, error: float, elapsed: float = 0.0) -> None:
        """
        Record an iteration's data.
        
        Args:
            iteration: Iteration number
            evaluation: Cumulative evaluation count
            error: Current best error value
            elapsed: Elapsed time in seconds since optimization start
        """
        self.iterations.append(iteration)
        self.evaluations.append(evaluation)
        self.errors.append(error)
        self.elapsed_times.append(elapsed)

    def save(self, optimizer_key: str = "unknown", extra_info: str = "") -> str | None:
        """
        Save the log to a CSV file.
        
        Args:
            optimizer_key: Name of the optimizer used
            extra_info: Additional info for the file header
            
        Returns:
            Path to the saved file, or None if nothing was written (no
            records, no log directory, or the save failed)
        """
        if not self.iterations or self.log_dir is None:
            return None

        # Ensure log directory exists; a log that cannot be written never
        # fails the run it records
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except OSError as e:
            print(f"Failed to save optimizer log: {e}")
            return None

        # Generate filename
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        safe_name = sanitize_filename(self.run_name)
        # Runs started within the same second get numbered files
        stem = f"{timestamp}_{safe_name}_{optimizer_key}"
        self._file_path = os.path.join(self.log_dir, f"{stem}.csv")
        number = 1
        while os.path.exists(self._file_path):
            number += 1
            self._file_path = os.path.join(self.log_dir, f"{stem}_{number}.csv")

        try:
            with open(self._file_path, "x", encoding="utf-8", newline="\n") as f:
                # Write header comments
                f.write("# Optimizer Error Log\n")
                f.write(f"# Run: {self.run_name}\n")
                f.write(f"# Optimizer: {optimizer_key}\n")
                f.write(f"# Start Time: {self.start_time.isoformat()}\n")
                if extra_info:
                    f.write(f"# Info: {extra_info}\n")
                f.write(f"# Total Iterations: {len(self.iterations)}\n")
                if self.evaluations:
                    f.write(f"# Total Evaluations: {self.evaluations[-1]}\n")
                if self.errors:
                    f.write(f"# Final Error: {self.errors[-1]:.12e}\n")
                    f.write(f"# Min Error: {min(self.errors):.12e}\n")
                if self.elapsed_times:
                    f.write(f"# Total Time: {self.elapsed_times[-1]:.3f}s\n")
                f.write("#\n")
                f.write("iteration,evaluation,error,elapsed_s\n")

                # Write data
                for it, ev, err, t in zip(self.iterations, self.evaluations, self.errors, self.elapsed_times, strict=False):
                    f.write(f"{it},{ev},{err:.12e},{t:.3f}\n")

            return self._file_path

        except Exception as e:
            print(f"Failed to save optimizer log: {e}")
            self._file_path = None
            return None

    @property
    def file_path(self) -> str | None:
        """Return the path of the saved log file, or None if not saved."""
        return self._file_path
