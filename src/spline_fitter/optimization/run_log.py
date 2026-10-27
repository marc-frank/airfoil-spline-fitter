"""
Optimizer Run Logs
==================

CSV logs of optimization progress for post-analysis.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from ..formatting import sanitize_filename


def _default_log_dir() -> str:
    """
    Return the directory for optimizer logs.

    ``SPLINE_FITTER_LOG_DIR`` overrides it. In a source checkout the logs go
    to ``optimizer_logs/`` at the project root; in an installed package they
    go to ``~/.spline_fitter/optimizer_logs``.
    """
    override = os.environ.get("SPLINE_FITTER_LOG_DIR")
    if override:
        return override
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    if os.path.isfile(os.path.join(project_root, "pyproject.toml")):
        return os.path.join(project_root, "optimizer_logs")
    return os.path.join(os.path.expanduser("~"), ".spline_fitter", "optimizer_logs")


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
    
    DEFAULT_LOG_DIR = _default_log_dir()
    
    def __init__(self, run_name: str = "unnamed", log_dir: Optional[str] = None):
        """
        Initialize a new logger for an optimization run.
        
        Args:
            run_name: Descriptive name for this run (e.g., airfoil name)
            log_dir: Directory for log files (default: DEFAULT_LOG_DIR)
        """
        self.run_name = run_name
        self.start_time = datetime.now()
        self.log_dir = log_dir or self.DEFAULT_LOG_DIR
        
        self.iterations: List[int] = []
        self.evaluations: List[int] = []
        self.errors: List[float] = []
        self.elapsed_times: List[float] = []
        
        self._file_path: Optional[str] = None
    
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
    
    def save(self, optimizer_key: str = "unknown", extra_info: str = "") -> Optional[str]:
        """
        Save the log to a CSV file.
        
        Args:
            optimizer_key: Name of the optimizer used
            extra_info: Additional info for the file header
            
        Returns:
            Path to the saved file, or None if save failed
        """
        if not self.iterations:
            return None
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate filename
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        safe_name = sanitize_filename(self.run_name)
        filename = f"{timestamp}_{safe_name}_{optimizer_key}.csv"
        self._file_path = os.path.join(self.log_dir, filename)
        
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
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
                for it, ev, err, t in zip(self.iterations, self.evaluations, self.errors, self.elapsed_times):
                    f.write(f"{it},{ev},{err:.12e},{t:.3f}\n")
            
            return self._file_path
            
        except Exception as e:
            print(f"Failed to save optimizer log: {e}")
            return None
    
    @property
    def file_path(self) -> Optional[str]:
        """Return the path of the saved log file, or None if not saved."""
        return self._file_path
