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


def _structured_quad_mesh(nx: int, ny: int):
    nx_ = int(nx)
    ny_ = int(ny)
    if nx_ <= 0 or ny_ <= 0:
        raise ValueError("Structured qualification dimensions must be positive.")
    coordinates = np.asarray(
        tuple((x / nx_, y / ny_) for y in range(ny_ + 1) for x in range(nx_ + 1))
    )
    stride = nx_ + 1
    cells = []
    for y in range(ny_):
        for x in range(nx_):
            lower_left = y * stride + x
            cells.append(
                (
                    lower_left,
                    lower_left + 1,
                    lower_left + stride + 1,
                    lower_left + stride,
                )
            )
    return phx.discretization.CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "cells", "quadrilateral", np.asarray(cells, dtype=np.int32)
            ),
        ),
    )


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


def _tensor_problem(
    *,
    viscous=False,
    sampled=False,
    physical_boundaries=True,
    nx=1,
    ny=1,
    degree=2,
):
    mesh = _structured_quad_mesh(nx, ny)
    system = (
        phx.equations.CompressibleNavierStokesSystem(
            phx.equations.ConstantTransport(0.1, 0.2), 2
        )
        if viscous
        else phx.equations.EulerSystem(2)
    )
    discretization = _field_discretization(mesh, system, "quadrilateral", degree=degree)
    if physical_boundaries:
        boundary = (
            phx.discretization.NoSlipAdiabaticWallBoundary(jnp.zeros((2,)))
            if viscous
            else phx.discretization.SlipWallBoundary()
        )
        boundaries = _boundary_set(discretization, boundary)
    else:
        boundaries = None
    interface = phx.discretization.EntropyStableEulerFluxPlan()
    volume = phx.discretization.EntropyConservativeEulerFluxPlan()
    entropy_pair = None
    compatibility = None
    if sampled:
        entropy_pair = phx.equations.ideal_gas_euler_entropy_pair(system)
        left = system.primitive_to_conserved(
            jnp.asarray(((1.0, 0.2, 0.0, 1.0), (0.9, -0.1, 0.1, 0.8)))
        )
        right = system.primitive_to_conserved(
            jnp.asarray(((0.8, 0.1, -0.1, 0.9), (1.1, 0.0, 0.05, 1.2)))
        )
        compatibility = phx.equations.fem.sample_dgsem_flux_compatibility(
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
        viscous=(
            phx.equations.fem.ViscousDGPlan(
                boundary_closures=(
                    ()
                    if boundaries is None
                    else (
                        phx.equations.fem.ViscousBoundaryClosure(
                            boundaries.patches[0].boundary.boundary_id
                        ),
                    )
                )
            )
            if viscous
            else None
        ),
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


def _triangle_problem(nx=1, ny=1, degree=2):
    nx_ = int(nx)
    ny_ = int(ny)
    coordinates = np.asarray(
        tuple((x / nx_, y / ny_) for y in range(ny_ + 1) for x in range(nx_ + 1))
    )
    stride = nx_ + 1
    triangles = []
    for y in range(ny_):
        for x in range(nx_):
            lower_left = y * stride + x
            lower_right = lower_left + 1
            upper_left = lower_left + stride
            upper_right = upper_left + 1
            triangles.extend(
                (
                    (lower_left, lower_right, upper_left),
                    (lower_right, upper_right, upper_left),
                )
            )
    mesh = phx.discretization.CellMesh.from_triangles(
        coordinates, np.asarray(triangles, dtype=np.int32)
    )
    system = phx.equations.EulerSystem(2)
    discretization = _field_discretization(mesh, system, "triangle", degree=degree)
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


def _advance(method, state, step_size, steps):
    value = state
    time = jnp.asarray(0.0, dtype=state.dtype)
    for step in range(int(steps)):
        result = method.step(
            jnp.asarray(step, dtype=jnp.int32),
            time,
            value,
            step_size,
            None,
        )
        if not bool(result.successful):
            raise RuntimeError("Qualification time advancement rejected a step.")
        value = result.accepted_state
        time = time + step_size
    return value


def _smooth_periodic_state(system, discretization):
    coordinates = discretization.dof_maps[0].dof_coordinates
    phase = 2.0 * jnp.pi * coordinates[:, 0]
    density = 1.0 + 0.05 * jnp.sin(phase)
    pressure = 1.0 + 0.03 * jnp.cos(phase)
    primitive = jnp.stack(
        (density, jnp.full_like(density, 0.2), jnp.full_like(density, 0.05), pressure),
        axis=-1,
    )
    return system.primitive_to_conserved(primitive)


def _discontinuous_periodic_state(system, discretization):
    coordinates = discretization.dof_maps[0].dof_coordinates
    left = coordinates[:, 0] < 0.5
    density = jnp.where(left, 1.0, 0.7)
    pressure = jnp.where(left, 1.0, 0.6)
    primitive = jnp.stack(
        (density, jnp.full_like(density, 0.15), jnp.zeros_like(density), pressure),
        axis=-1,
    )
    return system.primitive_to_conserved(primitive)


def _relative_integral_drift(dynamics, initial, final):
    initial_integral = dynamics.residual_with_diagnostics(0.0, initial)[1].total_integral
    final_integral = dynamics.residual_with_diagnostics(0.0, final)[1].total_integral
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(initial_integral)))
    return float(jnp.max(jnp.abs(final_integral - initial_integral)) / scale)


def _production_extension_checks(lane):
    checks = {}
    if lane in ("nightly", "weekly", "release"):
        pyramid = phx.discretization.fem.HybridReferenceFamily("pyramid", 3)
        pyramid_identity = float(
            jnp.max(
                jnp.abs(pyramid.tabulate(pyramid.nodes) - jnp.eye(pyramid.nodes.shape[0]))
            )
        )
        checks["pyramid_order_three_identity_defect"] = pyramid_identity
        _compiled, _system, discretization = _tensor_problem(physical_boundaries=False)
        coordinates = discretization.default_runtime.coordinates
        velocity = jnp.broadcast_to(jnp.asarray((0.1, -0.05)), coordinates.shape)
        current = phx.equations.fem.FiniteElementGeometrySnapshot(
            coordinates,
            velocity,
            0.0,
            topology_id=discretization.mesh.topology_id,
            geometry_layout_id="qualification-motion",
        )
        ale = phx.equations.fem.finite_element_ale_metric_evidence(
            discretization, current, current.advance(0.02), tolerance=2.0e-9
        )
        checks["uniform_translation_gcl_defect"] = float(ale.maximum_gcl_defect)
        checks["uniform_translation_gcl_passed"] = bool(ale.passed)
    if lane in ("weekly", "release"):
        mhd = phx.equations.IdealMHDSystem(1)
        mhd_state = mhd.primitive_to_conserved(
            jnp.asarray((1.0, 0.1, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0))
        )
        mhd_entropy = phx.equations.ideal_mhd_entropy_pair(mhd)
        checks["mhd_entropy_finite"] = bool(jnp.isfinite(mhd_entropy.entropy(mhd_state)))
        schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
            ("fuel", "oxidizer", "product"),
            (phx.equations.ChemicalPhaseKind.GAS,) * 3,
            jnp.asarray((0.002, 0.032, 0.018)),
            ("H", "O"),
            jnp.asarray(((2, 0, 2), (0, 2, 1)), dtype=jnp.int32),
            jnp.zeros((3,), dtype=jnp.int32),
            gas_standard_pressure=1.0e5,
        )
        species_thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
            schema,
            jnp.asarray(((20.0,), (22.0,), (25.0,))),
            jnp.asarray((0.0, 0.0, -2.4e5)),
            reference_temperature=300.0,
            minimum_temperature=200.0,
            maximum_temperature=4000.0,
        )
        ideal = phx.equations.IdealGasReferenceHelmholtzTerm(
            schema, species_thermodynamics
        )
        reacting = phx.equations.HomogeneousMixtureEulerSystem(
            phx.equations.HomogeneousHelmholtzPlan(
                ideal, phx.equations.ZeroResidualHelmholtzTerm(schema)
            )
        )
        reacting_state = reacting.primitive_to_conserved(
            jnp.asarray((0.04, 0.32, 0.18, 0.0, 1400.0))
        )
        checks["reacting_flux_finite"] = bool(
            jnp.all(jnp.isfinite(reacting.physical_flux(reacting_state, 0)))
        )
        turbulence = phx.equations.WALEPlan().kinematic_viscosity(
            jnp.asarray((1.0,)),
            phx.equations.TurbulenceArguments(
                jnp.asarray(((0.1, 0.2), (-0.1, -0.1))),
                jnp.asarray(0.02),
                jnp.asarray(0.1),
                jnp.asarray((0.1, 1.0)),
            ),
        )
        checks["les_viscosity_nonnegative"] = bool(
            jnp.isfinite(turbulence) & (turbulence >= 0.0)
        )
    if lane == "release":
        forest = phx.discretization.fem.GeneralHPForest.roots(
            ("triangle", "quadrilateral"),
            jnp.asarray(((2, 2, 0), (3, 3, 0))),
        )
        checks["hp_forest_active_cells"] = int(jnp.sum(forest.active))
        overset = phx.equations.fem.ConservativeOversetPlan(
            phx.equations.fem.OversetConnectivity(
                jnp.asarray(((0, 1),)),
                jnp.asarray((2,)),
                jnp.asarray(((0.5, 0.5),)),
                jnp.asarray(((0.25, 0.25),)),
                jnp.asarray((0.5,)),
                jnp.asarray((True,)),
            )
        )
        transfer = overset.transfer(jnp.ones((3, 1)))
        checks["overset_conservation_defect"] = float(transfer.conservation_defect)
        tape = phx.equations.fem.ReverseTimeTopologyTape(
            (
                phx.equations.fem.AcceptedStepAdjointRecord(
                    lambda value: 2.0 * value, 0, "qualification-step"
                ),
            )
        )
        adjoint = tape.reverse(jnp.ones((1,)))
        checks["topology_adjoint_valid"] = bool(adjoint.valid)
    return checks


def run(lane="release") -> dict[str, object]:
    wall, wall_system, wall_discretization = _tensor_problem(
        sampled=True, physical_boundaries=True
    )
    wall_state = _constant_state(wall_system, wall_discretization)
    wall_error = float(jnp.max(jnp.abs(wall(0.0, wall_state))))

    periodic, periodic_system, periodic_discretization = _tensor_problem(
        sampled=True, physical_boundaries=False, nx=2, ny=2
    )
    smooth_state = _smooth_periodic_state(periodic_system, periodic_discretization)
    step_size = jnp.minimum(
        periodic.dynamics.stable_step_evidence(smooth_state).step * 0.1,
        jnp.asarray(2.0e-3, dtype=smooth_state.dtype),
    )
    smooth_final = _advance(
        phx.solver.SSPRK33FixedStepMethod(periodic), smooth_state, step_size, 10
    )
    smooth_drift = _relative_integral_drift(periodic, smooth_state, smooth_final)
    smooth_change = float(jnp.max(jnp.abs(smooth_final - smooth_state)))

    filter_ = phx.equations.fem.EntropyFilterPlan(
        density_floor=1.0e-6, pressure_floor=1.0e-6
    ).prepare(periodic.dynamics)
    discontinuous = _discontinuous_periodic_state(
        periodic_system, periodic_discretization
    )
    filtered_final = _advance(
        phx.solver.SSPRK33FixedStepMethod(periodic, stage_transform=filter_),
        discontinuous,
        jnp.minimum(
            periodic.dynamics.stable_step_evidence(discontinuous).step * 0.05,
            jnp.asarray(1.0e-3, dtype=discontinuous.dtype),
        ),
        20,
    )
    filtered_drift = _relative_integral_drift(periodic, discontinuous, filtered_final)
    filtered_admissible = bool(
        periodic.dynamics.entropy_pair.admissible(filtered_final).all()
    )
    viscous, viscous_system, viscous_discretization = _tensor_problem(
        viscous=True, physical_boundaries=False, nx=2, ny=2
    )
    viscous_state = _smooth_periodic_state(viscous_system, viscous_discretization)
    viscous_step = jnp.minimum(
        viscous.dynamics.stable_step_evidence(viscous_state).step * 0.05,
        jnp.asarray(5.0e-4, dtype=viscous_state.dtype),
    )
    viscous_final = _advance(
        phx.solver.SSPRK33FixedStepMethod(viscous),
        viscous_state,
        viscous_step,
        5,
    )
    viscous_drift = _relative_integral_drift(viscous, viscous_state, viscous_final)
    viscous_change = float(jnp.max(jnp.abs(viscous_final - viscous_state)))

    triangle, triangle_system, triangle_discretization = _triangle_problem()
    triangle_state = _constant_state(
        triangle_system, triangle_discretization, velocity=0.05
    )
    triangle_error = float(jnp.max(jnp.abs(triangle(0.0, triangle_state))))
    triangle_coordinates = triangle_discretization.dof_maps[0].dof_coordinates
    triangle_primitive = jnp.stack(
        (
            1.0 + 0.03 * triangle_coordinates[:, 0],
            jnp.full((triangle_coordinates.shape[0],), 0.1),
            jnp.zeros((triangle_coordinates.shape[0],)),
            jnp.ones((triangle_coordinates.shape[0],)),
        ),
        axis=-1,
    )
    triangle_nonconstant = triangle_system.primitive_to_conserved(triangle_primitive)
    triangle_rate_norm = float(jnp.linalg.norm(triangle(0.0, triangle_nonconstant)))

    prism_error = _hybrid_case("prism")
    pyramid_error = _hybrid_case("pyramid")
    result = {
        "tensor_wall_free_stream_error": wall_error,
        "smooth_periodic_integral_drift": smooth_drift,
        "smooth_periodic_state_change": smooth_change,
        "filtered_periodic_integral_drift": filtered_drift,
        "filtered_periodic_admissible": filtered_admissible,
        "viscous_periodic_integral_drift": viscous_drift,
        "viscous_periodic_state_change": viscous_change,
        "triangle_free_stream_error": triangle_error,
        "triangle_nonconstant_rate_norm": triangle_rate_norm,
        "prism_free_stream_error": prism_error,
        "pyramid_free_stream_error": pyramid_error,
        "lane": lane,
        **_production_extension_checks(lane),
    }
    extension_passed = all(
        value is True
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
            and (value <= 5.0e-8 if "defect" in key else value >= 0.0)
        )
        for key, value in result.items()
        if key
        not in (
            "lane",
            "tensor_wall_free_stream_error",
            "smooth_periodic_integral_drift",
            "smooth_periodic_state_change",
            "filtered_periodic_integral_drift",
            "filtered_periodic_admissible",
            "viscous_periodic_integral_drift",
            "viscous_periodic_state_change",
            "triangle_free_stream_error",
            "triangle_nonconstant_rate_norm",
            "prism_free_stream_error",
            "pyramid_free_stream_error",
        )
    )
    result["passed"] = bool(
        wall_error <= 5.0e-9
        and smooth_drift <= 2.0e-9
        and smooth_change > 1.0e-8
        and filtered_drift <= 2.0e-9
        and filtered_admissible
        and viscous_drift <= 2.0e-9
        and viscous_change > 1.0e-9
        and triangle_error <= 5.0e-9
        and triangle_rate_norm > 1.0e-8
        and prism_error <= 5.0e-9
        and pyramid_error <= 5.0e-9
        and extension_passed
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/high_order_conservation_qualification.json"),
    )
    parser.add_argument(
        "--lane",
        choices=("pr", "nightly", "weekly", "release"),
        default="release",
    )
    args = parser.parse_args()
    result = run(args.lane)
    if not result["passed"]:
        raise RuntimeError("High-order conservation qualification failed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
