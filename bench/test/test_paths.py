"""Benchmark resource ownership, including source-only use without usim metadata."""

from pathlib import Path

from bench.paths import workspace_root


def test_source_root_ignores_usim_root(monkeypatch):
    monkeypatch.delenv("RVLN_BENCH_ROOT", raising=False)
    monkeypatch.setenv("USIM_ROOT", "/wrong/simulator")
    assert workspace_root() == Path(__file__).resolve().parents[1]
    assert (workspace_root() / "episodes/pilot.json").is_file()


def test_explicit_benchmark_root(monkeypatch, tmp_path):
    monkeypatch.setenv("RVLN_BENCH_ROOT", str(tmp_path))
    assert workspace_root() == tmp_path.resolve()
