"""Optimizer run logs: where they go and that none overwrites another."""
from pathlib import Path

from spline_fitter.optimization.run_log import OptimizerErrorLogger


def test_log_dir_follows_the_environment_set_after_import(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLINE_FITTER_LOG_DIR", str(tmp_path))
    assert OptimizerErrorLogger("NACA 0012").log_dir == str(tmp_path)


def test_runs_started_in_the_same_second_keep_their_own_logs(tmp_path):
    first = OptimizerErrorLogger("NACA 0012", log_dir=str(tmp_path))
    second = OptimizerErrorLogger("NACA 0012", log_dir=str(tmp_path))
    second.start_time = first.start_time
    first.log(1, 1, 0.5)
    second.log(1, 1, 0.25)
    paths = [first.save("gn-varpro"), second.save("gn-varpro")]
    assert len(set(paths)) == 2
    assert [Path(p).read_text(encoding="utf-8").splitlines()[-1] for p in paths] == [
        "1,1,5.000000000000e-01,0.000", "1,1,2.500000000000e-01,0.000"]
