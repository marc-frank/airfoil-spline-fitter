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


def test_a_log_folder_that_cannot_be_made_does_not_fail_the_run(tmp_path):
    """save() reports a log it cannot write by returning None, also when the
    folder cannot be created, so a finished fit is never reported as failed."""
    blocker = tmp_path / "a file"
    blocker.write_text("", encoding="utf-8")
    logger = OptimizerErrorLogger("NACA 0012", log_dir=str(blocker / "logs"))
    logger.log(1, 1, 0.5)
    assert logger.save("gn-varpro") is None
    assert logger.file_path is None
