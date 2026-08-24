#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from tools.trefftz_benchmarks import run_trefftz_benchmarks


def test_trefftz_benchmark_schema_and_replay():
    first = run_trefftz_benchmarks(
        (2, 4),
        boundary_points=32,
        evaluation_points=16,
        seed=21,
    )
    second = run_trefftz_benchmarks(
        (2, 4),
        boundary_points=32,
        evaluation_points=16,
        seed=21,
    )
    assert first["schema_version"] == 1
    assert first["dimensions"] == [2, 4]
    assert first["passed"]
    assert second["passed"]
    assert len(first["records"]) == 2
    for left, right in zip(first["records"], second["records"], strict=True):
        assert left["problem_id"] == right["problem_id"]
        assert left["certificate_id"] == right["certificate_id"]
        assert left["parameter_count"] == right["parameter_count"]
        assert left["passed"] == right["passed"]
