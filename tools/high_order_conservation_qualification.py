#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _field_discretization(mesh, system, cell_kind, degree=2):
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "state",
            phx.discretization.discontinuous_element(cell_kind, degree),
            component_shape=(system.component_count,),
        ),
    ).prepare()


def _boundary_set(discretization, boundary):
    exterior = tuple(
        int(value)
        for value in np.asarray(discretization.exterior_facet_domain.entity_indices)
    )
    return phx.discretization.fem.FiniteElementBoundarySet(
        discretization, {"boundary": (exterior, boundary)}
    )


def _constant_state(system, discretization, velocity=0.0):
    primitive = jnp.asarray((1.0,) + (float(velocity),) * system.dimension + (1.0,))
    return jnp.broadcast_to(
        system.primitive_to_conserved(primitive),
        discretization.field_spaces[0].vector_space.shape,
    )


def _tensor_problem(*, viscous=False, certified=False):
    mesh = phx.discretization.CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            phx.discretization.CellBlock(
                "cells", "quadrilateral", np.asarray(((0, 1, 2, 3),))
            ),
        ),
    )
    system = (
        phx.equations.CompressibleNavierStokesSystem(
            phx.equations.ConstantTransport(0.1, 0.2), 2
        )
        if viscous
        else phx.equations.EulerSystem(2)
    )
    discretization = _field_discretization(mesh, system, "quadrilateral")
    boundary = (
        phx.discretization.NoSlipAdiabaticWallBoundary(jnp.zeros((2,)))
        if viscous
        else phx.discretization.SlipWallBoundary()
    )
    boundaries = _boundary_set(discretization, boundary)
    interface = phx.discretization.EntropyStableEulerFluxPlan()
    volume = phx.discretization.EntropyConservativeEulerFluxPlan()
    entropy_pair = None
    compatibility = None
    if certified:
        entropy_pair = phx.equations.ideal_gas_euler_entropy_pair(system)
        left = system.primitive_to_conserved(
            jnp.asarray(((1.0, 0.2, 0.0, 1.0), (0.9, -0.1, 0.1, 0.8)))
        )
        right = system.primitive_to_conserved(
            jnp.asarray(((0.8, 0.1, -0.1, 0.9), (1.1, 0.0, 0.05, 1.2)))
        )
        compatibility = phx.equations.fem.certify_dgsem_flux_compatibility(
            system,
            volume,
            interface,
            entropy_pair,
            left,
            right,
            tolerance=3.0e-5,
            viscous_evidence="uncertified" if viscous else "absent",
        )
    method = phx.equations.fem.DGSEMConservationMethodPlan(
        volume,
        interface,
        compatibility=compatibility,
        viscous=(phx.equations.fem.LDGViscousFluxPlan() if viscous else None),
    )
    compiled = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            "tensor-conservation", "state", system, boundaries
        ),
        discretization,
        method,
        entropy_pair=entropy_pair,
    )
    return compiled, system, discretization


def _triangle_problem():
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int32),
    )
    system = phx.equations.EulerSystem(2)
    discretization = _field_discretization(mesh, system, "triangle")
    boundaries = _boundary_set(discretization, phx.discretization.ExtrapolationBoundary())
    compiled = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            "triangle-conservation", "state", system, boundaries
        ),
        discretization,
        phx.equations.fem.NodalDGConservationMethodPlan(
            phx.discretization.RusanovFluxPlan()
        ),
    )
    return compiled, system, discretization


def _hybrid_case(kind):
    points = {
        "prism": np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 1.0),
            )
        ),
        "pyramid": np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.5, 0.5, 1.0),
            )
        ),
    }[kind]
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", kind, np.arange(points.shape[0], dtype=np.int32)[None]
            ),
        ),
    )
    system = phx.equations.EulerSystem(3)
    discretization = _field_discretization(mesh, system, kind, degree=1)
    boundaries = _boundary_set(discretization, phx.discretization.ExtrapolationBoundary())
    compiled = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(kind, "state", system, boundaries),
        discretization,
        phx.equations.fem.NodalDGConservationMethodPlan(
            phx.discretization.RusanovFluxPlan()
        ),
    )
    state = _constant_state(system, discretization, velocity=0.05)
    return float(jnp.max(jnp.abs(compiled(0.0, state))))


def run() -> dict[str, object]:
    tensor, tensor_system, tensor_discretization = _tensor_problem(certified=True)
    tensor_state = _constant_state(tensor_system, tensor_discretization)
    tensor_error = float(jnp.max(jnp.abs(tensor(0.0, tensor_state))))

    filter_ = phx.equations.fem.EntropyFilterPlan(
        density_floor=1.0e-6, pressure_floor=1.0e-6
    ).prepare(tensor.dynamics)
    troubled = tensor_state.at[0, 0].set(1.0e-4).at[0, -1].set(1.0e-5)
    _filtered, filter_evidence = filter_.filter(0.0, troubled)

    viscous, viscous_system, viscous_discretization = _tensor_problem(viscous=True)
    viscous_state = _constant_state(viscous_system, viscous_discretization)
    viscous_error = float(jnp.max(jnp.abs(viscous(0.0, viscous_state))))

    triangle, triangle_system, triangle_discretization = _triangle_problem()
    triangle_state = _constant_state(
        triangle_system, triangle_discretization, velocity=0.05
    )
    triangle_error = float(jnp.max(jnp.abs(triangle(0.0, triangle_state))))

    partition = phx.discretization.fem.partition_cells_cost_aware(
        triangle_discretization, 2
    )
    phases = phx.discretization.fem.lower_distributed_finite_element_phases(
        triangle_discretization, partition
    )
    exactly_once = bool(
        jnp.all(
            jnp.sum(
                jnp.stack(
                    tuple(
                        phases.interface_mask(part)
                        for part in range(phases.partition.part_count)
                    )
                ),
                axis=0,
            )
            == 1
        )
    )

    prism_error = _hybrid_case("prism")
    pyramid_error = _hybrid_case("pyramid")
    result = {
        "tensor_wall_free_stream_error": tensor_error,
        "entropy_filter_successful": bool(filter_evidence.successful),
        "entropy_filter_mean_defect": float(filter_evidence.mean_defect),
        "viscous_rest_error": viscous_error,
        "triangle_free_stream_error": triangle_error,
        "prism_free_stream_error": prism_error,
        "pyramid_free_stream_error": pyramid_error,
        "distributed_interfaces_exactly_once": exactly_once,
        "partition_imbalance": float(partition.evidence.imbalance_ratio),
    }
    result["passed"] = bool(
        tensor_error <= 5.0e-9
        and filter_evidence.successful
        and float(filter_evidence.mean_defect) <= 5.0e-9
        and viscous_error <= 5.0e-9
        and triangle_error <= 5.0e-9
        and prism_error <= 5.0e-9
        and pyramid_error <= 5.0e-9
        and exactly_once
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/high_order_conservation_qualification.json"),
    )
    args = parser.parse_args()
    result = run()
    if not result["passed"]:
        raise RuntimeError("High-order conservation qualification failed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
