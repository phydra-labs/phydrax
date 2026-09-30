#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact PIC chain conservation/work and phase-separated capacity campaign.

Run from the repository root with ``python -m benchmarks.pic_chain_integration
--output benchmarks/pic_chain_integration.json``. Mesh admission, cold dynamic
chain preparation, compiled preparation, and prepared gather/deposit reuse are
reported separately. Particle and trajectory interval capacities control each
campaign. Compiler temporary/output/code estimates are the official JAX values.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.stages import Compiled

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.discretization._cell_complex import simplicial_cell_complex
from phydrax.discretization._cubical_whitney import (
    CubicalSplineWhitneyKernel,
    PICShapeOrder,
)
from phydrax.discretization._simplicial_locator import PreparedSimplicialCellLocator
from phydrax.discretization._topology import CellComplexTopology
from phydrax.discretization.fem._simplicial_whitney_chains import SimplicialWhitneyKernel
from phydrax.exterior._chains import AbstractChainIntegrationKernel


@dataclass(frozen=True)
class _Bundle:
    kernel: AbstractChainIntegrationKernel
    topology: CellComplexTopology
    coordinates: Array


def _cubical(order: PICShapeOrder) -> _Bundle:
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(8) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=jnp.float64))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    coordinates = bridge.cochain.coordinates[0]
    if coordinates is None:
        raise ValueError("The structured bridge must own vertex coordinates.")
    return _Bundle(
        CubicalSplineWhitneyKernel(bridge, order), bridge.cochain.topology, coordinates
    )


def _simplicial() -> _Bundle:
    coordinates = jnp.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), dtype=jnp.float64
    )
    cells = np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32)
    mesh = phx.discretization.CellMesh(
        coordinates,
        (phx.discretization.CellBlock("triangles", "triangle", cells),),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    locator = PreparedSimplicialCellLocator(
        phx.discretization.fem.prepare_finite_element_cell_map(discretization, 0),
        discretization.default_runtime.coordinates,
        phx.discretization.SimplicialLocationPolicy(2, 8, 1),
    )
    simplices = tuple(
        np.asarray(
            sorted(
                {
                    tuple(sorted(simplex))
                    for cell in cells.tolist()
                    for simplex in itertools.combinations(cell, degree + 1)
                }
            ),
            dtype=np.int32,
        )
        for degree in range(3)
    )
    topology = simplicial_cell_complex(simplices)
    return _Bundle(SimplicialWhitneyKernel(topology, locator), topology, coordinates)


def _simplicial_nd(dimension: int) -> _Bundle:
    coordinates = jnp.concatenate(
        (
            jnp.zeros((1, dimension), dtype=jnp.float64),
            jnp.eye(dimension, dtype=jnp.float64),
            jnp.ones((1, dimension), dtype=jnp.float64),
        ),
        axis=0,
    )
    cells = np.asarray(
        (tuple(range(dimension + 1)), (1, dimension + 1, *range(2, dimension + 1))),
        dtype=np.int32,
    )
    mesh = phx.discretization.CellMesh.from_simplices(
        coordinates, cells, dimension=dimension
    )
    prepared = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.fem.form_element(
                f"simplex:{dimension}",
                0,
                1,
                family="trimmed",
                twist="untwisted",
                proxy="scalar",
            ),
        ),
    ).prepare()
    locator = PreparedSimplicialCellLocator(
        phx.discretization.fem.prepare_finite_element_cell_map(prepared, 0),
        prepared.default_runtime.coordinates,
        phx.discretization.SimplicialLocationPolicy(2, 8, 1),
    )
    simplices = tuple(
        np.asarray(
            sorted(
                {
                    tuple(sorted(simplex))
                    for cell in cells.tolist()
                    for simplex in itertools.combinations(cell, degree + 1)
                }
            ),
            dtype=np.int32,
        )
        for degree in range(dimension + 1)
    )
    topology = simplicial_cell_complex(simplices)
    return _Bundle(SimplicialWhitneyKernel(topology, locator), topology, coordinates)


def _particle_data(dimension: int, particles: int) -> tuple[Array, Array, Array, Array]:
    fraction = jnp.linspace(0.0, 1.0, particles, dtype=jnp.float64)
    if dimension == 2:
        start = jnp.stack((0.31 + 0.01 * fraction, 0.34 + 0.01 * fraction), axis=1)
        end = jnp.stack((0.66 - 0.01 * fraction, 0.63 - 0.01 * fraction), axis=1)
        constant = jnp.asarray((0.7, -0.4), dtype=jnp.float64)
    else:
        start = jnp.broadcast_to(
            ((0.5 + 0.01 * fraction) / dimension)[:, None], (particles, dimension)
        )
        end = jnp.broadcast_to(
            ((1.5 - 0.01 * fraction) / dimension)[:, None], (particles, dimension)
        )
        constant = jnp.linspace(0.7, -0.4, dimension, dtype=jnp.float64)
    return start, end, 0.5 + fraction, constant


def _compiler(compiled: Compiled) -> dict[str, object]:
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The backend did not supply compiler analysis.",
    )
    return asdict(evidence)


def _phase_evidence(
    kernel: AbstractChainIntegrationKernel,
    start: Array,
    end: Array,
    electric: Array,
    charges: Array,
    expected: np.ndarray,
    capacity: int,
) -> tuple[float, float]:
    particles = start.shape[0]
    rates = jnp.linspace(-2.0, 2.0, particles, dtype=jnp.float64).at[0].set(0.0)
    query = kernel.integrate_segments(
        start,
        end,
        weight="phase",
        phase_rate=rates,
        maximum_segments=capacity,
    )
    np.testing.assert_array_equal(query.successful, np.ones((particles,), dtype=np.bool_))
    host_rates = np.asarray(rates)
    phase_factor = np.ones_like(host_rates, dtype=np.complex128)
    nonzero = host_rates != 0
    phase_factor[nonzero] = np.expm1(1j * host_rates[nonzero]) / (
        1j * host_rates[nonzero]
    )
    gathered = query.gather(electric)
    phase_error = float(np.max(np.abs(np.asarray(gathered) - expected * phase_factor)))
    if phase_error > 2e-11:
        raise ValueError("Exact phase-weighted uniform-field moments failed.")
    weights = charges * jnp.exp(1j * jnp.linspace(0.2, 0.6, particles, dtype=jnp.float64))
    duality_error = float(
        jnp.abs(jnp.vdot(gathered, weights) - jnp.vdot(electric, query.deposit(weights)))
    )
    if duality_error > 2e-11:
        raise ValueError("Hermitian phase gather/deposit work identity failed.")
    return phase_error, duality_error


def _campaign(
    factory: Callable[[], _Bundle],
    particles: int,
    capacity: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    bundle, admission = measure_synchronized(factory)
    kernel = bundle.kernel
    start, end, charges, constant = _particle_data(bundle.coordinates.shape[1], particles)
    potential = bundle.coordinates @ constant
    incidence = bundle.topology.incidences[0]
    lower = incidence.relation.source_indices
    upper = incidence.relation.target_indices
    signs = incidence.signs
    electric = (
        jnp.zeros((kernel.dof_counts[1],), dtype=jnp.float64)
        .at[upper]
        .add(signs * potential[lower])
    )

    def execute(
        first: Array, last: Array, field: Array, charge: Array
    ) -> tuple[Array, Array, Array, Array]:
        chain = kernel.integrate_segments(first, last, maximum_segments=capacity)
        before = kernel.integrate_points(first)
        after = kernel.integrate_points(last)
        flow = chain.deposit(charge)
        difference = after.deposit(charge) - before.deposit(charge)
        divergence = jnp.zeros_like(difference).at[lower].add(signs * flow[upper])
        successful = chain.successful & before.successful & after.successful
        return chain.gather(field), flow, difference - divergence, successful

    arguments = (start, end, electric, charges)
    _, cold = measure_synchronized(lambda: execute(*arguments))
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(execute).lower(*arguments), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(*arguments), warmup=warmup, repeats=repeats
    )
    gathered, flow, continuity, successful = result
    expected = np.asarray((end - start) @ constant)
    np.testing.assert_allclose(gathered, expected, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(continuity, 0.0, rtol=0.0, atol=2e-11)
    np.testing.assert_array_equal(successful, np.ones((particles,), dtype=np.bool_))
    work_error = float(jnp.abs(electric @ flow - charges @ gathered))
    if work_error > 2e-11:
        raise ValueError("The chain gather/deposit work identity failed.")
    reverse_flow = execute(start[::-1], end[::-1], electric, charges[::-1])[1]
    np.testing.assert_allclose(reverse_flow, flow, rtol=2e-11, atol=2e-12)

    prepared, prepare_seconds = measure_synchronized(
        lambda: kernel.integrate_segments(start, end, maximum_segments=capacity)
    )

    def reuse(field: Array, charge: Array) -> tuple[Array, Array]:
        return prepared.gather(field), prepared.deposit(charge)

    reuse_compiled, reuse_compilation = measure_lower_and_compile(
        lambda: jax.jit(reuse).lower(electric, charges), lambda lowered: lowered.compile()
    )
    _, reuse_execution = measure_repeated(
        lambda: reuse_compiled(electric, charges), warmup=warmup, repeats=repeats
    )
    phase_error, phase_duality_error = _phase_evidence(
        kernel, start, end, electric, charges, expected, capacity
    )
    return {
        "configuration": {"particles": particles, "maximum_segments": capacity},
        "preparation_seconds": admission,
        "cold_execution_seconds": cold,
        "compilation": asdict(compilation),
        "execution": execution.to_seconds_dict(),
        "compiler": _compiler(compiled),
        "prepared_reuse": {
            "query_preparation_seconds": prepare_seconds,
            "compilation": asdict(reuse_compilation),
            "execution": reuse_execution.to_seconds_dict(),
            "compiler": _compiler(reuse_compiled),
        },
        "memory": {
            "prepared_kernel_bytes": logical_array_bytes(bundle),
            "prepared_chain_bytes": logical_array_bytes(prepared),
            "particle_input_bytes": logical_array_bytes(arguments),
        },
        "physics": {
            "successful": bool(jnp.all(successful)),
            "maximum_continuity_defect": float(jnp.max(jnp.abs(continuity))),
            "maximum_constant_field_error": float(
                np.max(np.abs(np.asarray(gathered) - expected))
            ),
            "work_duality_error": work_error,
            "maximum_phase_error": phase_error,
            "phase_work_duality_error": phase_duality_error,
            "slot_order_invariant": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--particles", type=int, nargs="+", default=(4, 32))
    parser.add_argument("--capacities", type=int, nargs="+", default=(8, 16))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    if any(value < 1 for value in (*arguments.particles, *arguments.capacities)):
        parser.error("Particle and interval capacities must be positive.")
    factories: tuple[tuple[str, Callable[[], _Bundle]], ...] = (
        ("cubical_linear", lambda: _cubical(1)),
        ("cubical_cubic", lambda: _cubical(3)),
        ("simplicial", _simplicial),
        ("simplicial_4d", lambda: _simplicial_nd(4)),
        ("simplicial_5d", lambda: _simplicial_nd(5)),
    )
    cases = {
        name: [
            _campaign(factory, particles, capacity, arguments.warmup, arguments.repeats)
            for particles in arguments.particles
            for capacity in arguments.capacities
        ]
        for name, factory in factories
    }
    driver = Path(__file__).resolve()
    identity = capture_benchmark_identity(
        driver.parents[1], driver, cases["cubical_linear"][0].keys()
    )
    record = {
        "kind": "pic-chain-integration-capacity-campaign",
        "identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "cases": cases,
    }
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is not None:
        write_json_atomic(arguments.output, record)
    print(text, end="")


if __name__ == "__main__":
    main()
