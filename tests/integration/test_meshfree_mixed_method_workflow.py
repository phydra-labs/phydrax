#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Meshfree point clouds coupled to finite elements through chart-authorized traces.

The manufactured material-interface field of the example is continuous with a
continuous normal flux and a conductivity jump ``k_plus / k_minus = 3``, so its
normal derivative jumps across ``x = 1``. Every scenario solves the original
coupled equations and checks independent nodal errors against that field.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from examples.meshfree_mixed_method_coupling import (
    exact,
    point_region,
    prepare_level,
    Side,
)
from phydrax.solver.coupling import (
    ConservativeFluxLaw,
    CoupledProblemPlan,
    CoupledSolution,
    InterfaceConductance,
    MortarImposition,
    MortarMultiplier,
    NitscheImposition,
    prepare_coupled_problem,
    solve_coupled_problem,
    TransmissionImposition,
    TransmissionSide,
)
from tests.unit.solver.coupling import test_finite_volume_components as contact
from tests.unit.solver.coupling._cases import interface_binding, plate_cover


POLICY = phx.linalg.LinearSolvePolicy(
    phx.linalg.DenseLU(),
    materialization=phx.linalg.MaterializationPolicy(
        max_entries=64_000_000, max_bytes=1024 * 1024 * 1024
    ),
)
MORTAR = MortarImposition(MortarMultiplier("side-trace", side="triangles"))


def _solve(
    resolution: int, imposition: TransmissionImposition, /, *, cloud: Side = "right"
) -> tuple[CoupledSolution, np.ndarray, np.ndarray]:
    prepared, dofs, points = prepare_level(resolution, imposition, cloud=cloud)
    return solve_coupled_problem(prepared, policy=POLICY), dofs, points


def _errors(
    solution: CoupledSolution, dofs: np.ndarray, points: np.ndarray, /
) -> tuple[float, float]:
    fe = np.asarray(solution.field("triangles", "u")) - np.asarray(exact(dofs))
    cloud = np.asarray(solution.field("cloud", "u")) - np.asarray(exact(points))
    return float(np.max(np.abs(fe))), float(np.max(np.abs(cloud)))


def test_mortar_material_interface_converges_with_continuity_and_flux_balance() -> None:
    coarse, coarse_dofs, coarse_points = _solve(6, MORTAR)
    fine, fine_dofs, fine_points = _solve(12, MORTAR)
    for solution in (coarse, fine):
        assert bool(solution.native_successful) and bool(solution.accepted)
    coarse_errors = _errors(coarse, coarse_dofs, coarse_points)
    fine_errors = _errors(fine, fine_dofs, fine_points)
    assert fine_errors[0] < 0.5 * coarse_errors[0]
    assert fine_errors[1] < 0.5 * coarse_errors[1]
    # Matching FE and chart traces are both piecewise linear on the interface
    # edges, so the mortar makes them coincide up to roundoff at every level.
    report = fine.interface("transmission")
    index = report.names.index("trace-mismatch-l2")
    assert float(report.values[index]) <= 1e-12 * float(report.scales[index])


def test_cloud_on_the_minus_side_reverses_the_trace_orientation() -> None:
    forward, forward_dofs, forward_points = _solve(12, MORTAR)
    reversed_, dofs, points = _solve(12, MORTAR, cloud="left")
    assert bool(reversed_.accepted)
    # The methods exchange halves: the cloud's outward normal is +x and its
    # material the minus conductivity; both orientations resolve the field.
    assert (
        _errors(reversed_, dofs, points)[1]
        < 2.0 * _errors(forward, forward_dofs, forward_points)[1] + 1e-3
    )


def test_one_sided_nitsche_uses_the_finite_element_flux_and_penalty() -> None:
    one_sided = NitscheImposition(penalty_factor=4.0, weights=(1.0, 0.0))
    coarse, coarse_dofs, coarse_points = _solve(6, one_sided)
    fine, fine_dofs, fine_points = _solve(12, one_sided)
    assert bool(fine.accepted)
    assert (
        _errors(fine, fine_dofs, fine_points)[1]
        < 0.5 * _errors(coarse, coarse_dofs, coarse_points)[1]
    )
    with pytest.raises(ValueError, match="zero flux"):
        _solve(6, NitscheImposition(penalty_factor=4.0, weights=(0.5, 0.5)))


def test_cloud_interface_trace_matches_the_exact_field_on_its_charts() -> None:
    cloud, points = point_region(12)
    domain = cloud.boundary_domain("u", "interface")
    rule = phx.discretization.FacetTraceRule("gauss-lobatto-legendre", points=4)
    trace = cloud.prepare_side_trace("u", domain, rule=rule)
    values = trace.apply(jnp.asarray(exact(points)))
    sites = np.asarray(trace.sites)
    np.testing.assert_allclose(sites[..., 0], 1.0)
    assert float(jnp.max(jnp.abs(values - exact(sites)))) < 1e-2
    np.testing.assert_allclose(np.sum(np.asarray(trace.weights)), 1.0, rtol=1e-12)


# --- Meshfree--finite-volume imperfect contact -----------------------------------------


def _contact_level(resolution: int, /) -> tuple[float, float, CoupledSolution]:
    """Cloud on the left half, cell-centered FV grid on the right, conductance ``h``."""
    cloud, points = point_region(
        resolution,
        side="left",
        value=contact._minus,
        load=contact._negative_laplacian(contact._minus),
        kappa=1.0,
    )
    cells, discretization = contact._owner(resolution)
    cover = plate_cover()
    binding = interface_binding(
        cover,
        cloud.field_space_id("u"),
        cells.field_space_id("u"),
        roles=("cloud", "right"),
    )
    law = ConservativeFluxLaw(
        "contact",
        binding,
        (
            TransmissionSide(
                "cloud", "cloud", "u", cloud.boundary_domain("u", "interface")
            ),
            TransmissionSide("right", "right", "u", contact._interface(discretization)),
        ),
        InterfaceConductance(contact.CONDUCTANCE),
    )
    plan = CoupledProblemPlan(
        "meshfree-fv-contact", components=(cloud, cells), bindings=(binding,), laws=(law,)
    )
    prepared = prepare_coupled_problem(plan, interface_owners=(cover,))
    solution = solve_coupled_problem(prepared, policy=POLICY)
    cloud_exact = contact._host(contact._minus, points)
    cloud_error = np.max(np.abs(np.asarray(solution.field("cloud", "u")) - cloud_exact))
    cell_exact = contact._host(contact._plus, np.asarray(discretization.cell_centers))
    cell_error = np.max(np.abs(np.asarray(solution.field("right", "u")) - cell_exact))
    return float(cloud_error), float(cell_error), solution


def test_meshfree_fv_conductance_contact_converges_with_the_implied_jump() -> None:
    coarse_cloud, coarse_cells, coarse = _contact_level(6)
    fine_cloud, fine_cells, fine = _contact_level(12)
    for solution in (coarse, fine):
        assert bool(solution.native_successful) and bool(solution.accepted)
    # The cell-average face state sits half a cell from the contact, so the
    # contact flux, and with it both sides, is first-order consistent.
    assert coarse_cells / fine_cells > 1.7
    assert coarse_cloud / fine_cloud > 1.7
    report = fine.interface("contact")
    # One shared density leaves the cloud's conormal rows exactly as it enters the cells.
    assert float(report.value("flux-conservation")) <= 1.0e-12 * float(
        report.scales[report.names.index("flux-conservation")]
    )
    # The traces carry the implied jump u_minus - u_plus = Q / h.
    assert float(report.value("trace-jump-l2")) == pytest.approx(
        contact._jump_l2(), rel=1.0e-1
    )
