#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distributed PIC benchmark sizing and executed strong-scaling science."""

from __future__ import annotations

import math
import sys

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

import phydrax as phx
from benchmarks import pic_distributed as benchmark


@pytest.mark.parametrize("route", benchmark._ROUTES)
@pytest.mark.parametrize("parts", [[1, 2, 4], [1, 2, 3, 4, 5]])
def test_campaign_sizes_preserve_weak_work_and_bound_strong_rounding(
    route: str, parts: list[int]
) -> None:
    weak, strong = benchmark._problem_sizes(route, parts, 16, 512)
    assert {
        problem.capacity // count for problem, count in zip(weak, parts, strict=True)
    } == {512}
    assert (
        len(
            {
                math.prod(problem.counts) // count
                for problem, count in zip(weak, parts, strict=True)
            }
        )
        == 1
    )
    assert 512 * max(parts) <= strong.capacity < 512 * max(parts) + math.lcm(*parts)
    for count, problem in zip(parts, weak, strict=True):
        shape = benchmark._mesh_shape(benchmark._block_shape(route, count))
        assert math.prod(shape) == count
        assert shape == (1,) or all(value > 1 for value in shape)
        for candidate in (problem, strong):
            assert candidate.capacity % count == 0
            assert all(n % p == 0 for n, p in zip(candidate.counts, shape, strict=False))
            if route != "reduced-slabs":
                if len(shape) == 1:
                    assert candidate.counts[1] % count == 0
                else:
                    assert candidate.counts[2] % count == 0
                if candidate.subdomains is not None:
                    assert all(
                        n % p == 0
                        for n, p in zip(candidate.subdomains, shape, strict=False)
                    )
                    assert all(
                        n % p == 0
                        for n, p in zip(
                            candidate.counts, candidate.subdomains, strict=True
                        )
                    )
    if parts == [1, 2, 4]:
        assert strong.capacity == 2048
        assert strong.counts == ((64,) if route == "reduced-slabs" else (32, 32, 8))
        expected = (
            [(16,), (32,), (64,)]
            if route == "reduced-slabs"
            else [(16, 16, 8), (32, 16, 8), (32, 32, 8)]
        )
        assert [problem.counts for problem in weak] == expected
    else:
        assert strong.capacity == 2580
        assert strong.counts == ((120,) if route == "reduced-slabs" else (90, 60, 8))


@pytest.mark.parametrize("route", benchmark._ROUTES[1:])
@pytest.mark.parametrize("parts", [3, 5])
def test_prime_spectral_campaign_sizes_are_admitted_by_pic(
    route: str, parts: int
) -> None:
    if len(jax.devices()) < parts:
        pytest.skip(f"Needs {parts} devices.")
    selected = [1, 2, 3, 4, 5]
    weak, strong = benchmark._problem_sizes(route, selected, 16, 32)
    shape = benchmark._mesh_shape(benchmark._block_shape(route, parts))
    mesh = Mesh(np.asarray(jax.devices()[:parts], dtype=object).reshape(shape), ("x",))
    for problem in (weak[selected.index(parts)], strong):
        base, species, _, _ = benchmark._base(route, problem, mesh)
        solver = phx.solver.distribute_pic_field_solver(base, mesh)
        pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
        run = phx.solver.DistributedElectromagneticPICPlan(pic, packet_capacity=32)
        assert run.evidence.part_count == parts
        assert run.evidence.field_sharded
        assert run.evidence.mesh_shape == (parts,)
        assert base.plan.subdomains == problem.subdomains


def test_mixed_prime_strong_campaign_executes_the_same_science() -> None:
    if len(jax.devices()) < 5:
        pytest.skip("Needs five devices.")
    parts = [1, 2, 3, 4, 5]
    _, problem = benchmark._problem_sizes("reduced-slabs", parts, 16, 32)
    reference = None
    for count in parts:
        record, state = benchmark._case("reduced-slabs", count, problem, 32, 0, 1)
        physics = record["physics"]
        assert isinstance(physics, dict)
        assert physics["successful"]
        assert physics["rejection_reason"] == 0
        assert record["grid_shape"] == list(problem.counts)
        assert record["slots_per_species"] == problem.capacity
        assert int(state.accepted_step) == 1
        if reference is None:
            reference = state
        else:
            assert benchmark._field_difference(state, reference) < 1e-11
            np.testing.assert_array_equal(state.time, reference.time)
            for actual, expected in zip(state.species, reference.species, strict=True):
                active = np.asarray(actual.population.active)
                expected_active = np.asarray(expected.population.active)
                order = np.argsort(np.asarray(actual.population.id_lo)[active])
                expected_order = np.argsort(
                    np.asarray(expected.population.id_lo)[expected_active]
                )
                np.testing.assert_array_equal(
                    np.asarray(actual.population.id_lo)[active][order],
                    np.asarray(expected.population.id_lo)[expected_active][
                        expected_order
                    ],
                )
                np.testing.assert_allclose(
                    np.asarray(actual.particles.position)[active][order],
                    np.asarray(expected.particles.position)[expected_active][
                        expected_order
                    ],
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    np.asarray(actual.particles.proper_velocity)[active][order],
                    np.asarray(expected.particles.proper_velocity)[expected_active][
                        expected_order
                    ],
                    rtol=0.0,
                    atol=1e-12,
                )


@pytest.mark.parametrize(
    "options",
    [
        ["--parts", "0"],
        ["--parts", "2"],
        ["--cells-per-part", "0"],
        ["--slots-per-part", "0"],
        ["--packet-capacity", "0"],
        ["--warmup", "-1"],
        ["--repeats", "0"],
    ],
)
def test_cli_refuses_invalid_campaign_sizes(
    monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["pic_distributed", "--parts", "1", *options])
    with pytest.raises(ValueError, match="must be positive"):
        benchmark.main()
