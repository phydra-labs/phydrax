#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Ownership of boundary laws, constraints, gauges, and native solve status."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from tests.unit.solver.coupling._cases import (
    build_region,
    couple,
    dense_policy,
    interface_binding,
    NEUMANN,
    nodal_error,
    plate_cover,
    QUADRATIC,
    Region,
    RegionSpec,
    SMOOTH,
    transmission_law,
)


if TYPE_CHECKING:
    from phydrax.solver.coupling import AbstractSpatialComponent
    from phydrax.solver.coupling._interfaces import InterfaceOwner


cpl = phx.solver.coupling


@pytest.fixture(scope="module")
def left() -> Region:
    return build_region(RegionSpec("left", "fe", 0.0, 1.0, 3, 1), SMOOTH)


@pytest.fixture(scope="module")
def right() -> Region:
    return build_region(RegionSpec("right", "fe", 1.0, 2.0, 4, 1), SMOOTH)


def _prepare(
    left: Region,
    right: Region,
    laws: tuple[cpl.AbstractCouplingLaw, ...],
    binding: cpl.InterfaceBinding,
    cover: phx.domain.SubdomainCover,
) -> cpl.PreparedCoupledProblem:
    plan = cpl.CoupledProblemPlan(
        "ownership",
        components=(left.component, right.component),
        bindings=(binding,),
        laws=laws,
    )
    return cpl.prepare_coupled_problem(plan, interface_owners=(cover,))


def test_owner_boundary_law_on_the_interface_is_refused(right: Region) -> None:
    loaded = build_region(
        RegionSpec("left", "fe", 0.0, 1.0, 3, 1, interface_load=True), SMOOTH
    )
    with pytest.raises(ValueError, match="boundary law is imposed once"):
        couple(loaded, right, "mortar-side-trace")


def test_dirichlet_data_on_the_whole_interface_is_refused(right: Region) -> None:
    clamped = build_region(
        RegionSpec("left", "fe", 0.0, 1.0, 3, 1, dirichlet="whole-boundary"), SMOOTH
    )
    with pytest.raises(ValueError, match="imposes its whole trace strongly"):
        couple(clamped, right, "mortar-side-trace")


def test_second_law_on_the_same_interface_is_refused(left: Region, right: Region) -> None:
    cover = plate_cover()
    binding = interface_binding(
        cover, left.component.field_space_id("u"), right.component.field_space_id("u")
    )
    laws = (
        transmission_law(left, right, binding, "mortar-side-trace", law_id="gamma"),
        transmission_law(left, right, binding, "mortar-side-trace", law_id="gamma-again"),
    )
    with pytest.raises(ValueError, match="both act on facets"):
        _prepare(left, right, laws, binding, cover)


def test_partial_interface_coverage_is_refused(left: Region) -> None:
    short = build_region(RegionSpec("right", "fe", 1.0, 2.0, 4, 1, height=0.5), SMOOTH)
    with pytest.raises(ValueError, match="do not cover each other"):
        couple(left, short, "mortar-side-trace")


def test_regions_on_the_same_side_of_the_interface_are_refused(left: Region) -> None:
    # Both regions lie in x <= 1, so both outward normals point along +x.
    overlapping = build_region(RegionSpec("right", "fe", 0.0, 1.0, 4, 1), SMOOTH)
    with pytest.raises(
        ValueError, match="the law side is not this interface or lies across it"
    ):
        couple(left, overlapping, "mortar-side-trace")


def test_rank_deficient_multiplier_space_is_refused() -> None:
    coarse = build_region(RegionSpec("left", "fe", 0.0, 1.0, 2, 1), SMOOTH)
    fine = build_region(RegionSpec("right", "fe", 1.0, 2.0, 8, 1), SMOOTH)
    # 8 facets x 2 linear moments = 16 multipliers against 1 + 7 free trace rows.
    with pytest.raises(ValueError, match=r"rank deficient.*numerical rank \d+ of 16"):
        couple(coarse, fine, "mortar-discontinuous", side="right", multiplier_degree=1)


def test_matching_elimination_of_nonmatching_facets_is_refused(
    left: Region, right: Region
) -> None:
    with pytest.raises(ValueError, match="coincident facet partitions"):
        couple(left, right, "matching")


@pytest.fixture(scope="module")
def floating() -> tuple[Region, Region]:
    """Pure-Neumann owners of the compatible cubic x^2 - x^3/3 + y^2 - 2 y^3/3."""
    return (
        build_region(RegionSpec("left", "fe", 0.0, 1.0, 6, 1, dirichlet="none"), NEUMANN),
        build_region(
            RegionSpec("right", "fe", 1.0, 2.0, 8, 1, dirichlet="none"), NEUMANN
        ),
    )


def test_coupled_kernel_without_gauge_is_refused(floating: tuple[Region, Region]) -> None:
    with pytest.raises(ValueError, match="1-dimensional kernel.*CoupledGauge"):
        couple(*floating, "mortar-side-trace")


def test_gauged_floating_problem_matches_the_field_up_to_a_constant(
    floating: tuple[Region, Region],
) -> None:
    left, right = floating
    coupled = couple(
        left, right, "mortar-side-trace", gauge=cpl.CoupledGauge(compatibility="project")
    )
    policy = coupled.prepared.nullspace_policy
    assert policy is not None
    assert policy.right is not None
    assert policy.right.basis.shape[1] == 1
    # Unrestarted GMRES (restart above the 137 unknowns) on the gauged system.
    solution = cpl.solve_coupled_problem(
        coupled.prepared,
        policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(restart=160),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1.0e-10, absolute=0.0, max_steps=400
            ),
        ),
    )
    errors = [
        np.asarray(solution.field(region.spec.name, "u"))
        - NEUMANN.host(region.dof_points)
        for region in floating
    ]
    shift = float(np.mean(np.concatenate(errors)))

    assert bool(solution.native_successful)
    assert bool(solution.accepted)
    # P1 nodal error of the cubic field on h = 1/6 and 1/8 meshes.
    for error in errors:
        assert np.max(np.abs(error - shift)) <= 2.0e-2


def test_binding_field_identities_must_match_the_components(
    left: Region, right: Region
) -> None:
    cover = plate_cover()
    swapped = interface_binding(
        cover, right.component.field_space_id("u"), left.component.field_space_id("u")
    )
    law = transmission_law(left, right, swapped, "mortar-side-trace")
    with pytest.raises(ValueError, match="binds value field"):
        _prepare(left, right, (law,), swapped, cover)


def test_transmission_sides_must_follow_the_binding_orientation(
    left: Region, right: Region
) -> None:
    cover = plate_cover()
    binding = interface_binding(
        cover, left.component.field_space_id("u"), right.component.field_space_id("u")
    )
    reversed_law = cpl.ScalarTransmissionLaw(
        "gamma",
        binding,
        (
            cpl.TransmissionSide("right", "right", "u", right.interface),
            cpl.TransmissionSide("left", "left", "u", left.interface),
        ),
        cpl.MortarImposition(cpl.MortarMultiplier("side-trace", side="right")),
    )
    with pytest.raises(ValueError, match=r"\(minus, plus\) roles"):
        _prepare(left, right, (reversed_law,), binding, cover)


class _NoDefects(cpl.AbstractLawCertificate):
    law_id: str = eqx.field(static=True)

    def defects(
        self,
        fields: Mapping[tuple[str, str], Array],
        law_state: tuple[Array, ...],
        args: Mapping[str, object],
        /,
    ) -> cpl.InterfaceDefectReport:
        del fields, law_state, args
        empty = jnp.zeros((0,), dtype=jnp.float64)
        return cpl.InterfaceDefectReport(self.law_id, (), (), empty, empty)


class _PublishedLaw(cpl.AbstractCouplingLaw):
    """Custom domain law publishing typed contributions without an interface."""

    law_id: str = eqx.field(static=True)
    unknowns: tuple[cpl.LawBlock, ...]
    rows: tuple[cpl.LawBlock, ...]
    contributions: tuple[cpl.Contribution, ...]

    @property
    def bindings(self) -> tuple[cpl.InterfaceBinding, ...]:
        return ()

    def prepare(
        self,
        components: Mapping[str, AbstractSpatialComponent],
        interface_owners: tuple[InterfaceOwner, ...],
        /,
    ) -> cpl.PreparedLaw:
        del components, interface_owners
        certificate = _NoDefects(self.law_id)
        return cpl.PreparedLaw(
            self.law_id,
            binding_id=None,
            state_blocks=self.unknowns,
            row_blocks=self.rows,
            contributions=self.contributions,
            impositions=(),
            certificate=certificate,
            evidence=certificate,
        )


def _alone(region: Region, law: _PublishedLaw) -> cpl.PreparedCoupledProblem:
    plan = cpl.CoupledProblemPlan(
        "custom", components=(region.component,), bindings=(), laws=(law,)
    )
    return cpl.prepare_coupled_problem(plan)


def _reduced_mass(region: Region, space: cpl.ContributionSpace) -> cpl.LinearContribution:
    """A diagonal reaction term on the solve coordinates of ``region``."""
    reduced = region.component.state_blocks[0].space
    operator = phx.linalg.DenseLinearOperator(
        jnp.eye(reduced.size, dtype=jnp.float64),
        source=reduced,
        target=phx.linalg.DualSpace(reduced),
    )
    endpoint = cpl.ContributionEndpoint("left", "u", space=space)
    return cpl.LinearContribution(
        endpoint, endpoint, operator, law_id="reaction", imposition_id="reaction"
    )


def test_reduced_operator_addressed_as_a_full_field_is_refused(left: Region) -> None:
    accepted = _alone(
        left, _PublishedLaw("reaction", (), (), (_reduced_mass(left, "reduced"),))
    )
    assert accepted.execution == "linear"
    with pytest.raises(ValueError, match="Row-space mismatch"):
        _alone(left, _PublishedLaw("reaction", (), (), (_reduced_mass(left, "full"),)))


def test_law_blocks_that_do_not_square_are_refused(left: Region) -> None:
    unknowns = (cpl.LawBlock("extra", phx.linalg.ArraySpace((3,), dtype=jnp.float64)),)
    rows = (cpl.LawBlock("extra", phx.linalg.ArraySpace((2,), dtype=jnp.float64)),)
    with pytest.raises(ValueError, match="Row-space mismatch"):
        _alone(left, _PublishedLaw("extra-law", unknowns, rows, ()))


def test_native_solve_failure_propagates_without_acceptance(
    left: Region, right: Region
) -> None:
    coupled = couple(left, right, "mortar-side-trace")
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(),
        tolerance=phx.linalg.TolerancePolicy(relative=1.0e-14, absolute=0.0, max_steps=1),
    )
    solution = cpl.solve_coupled_problem(coupled.prepared, policy=policy)

    assert not bool(solution.native_successful)
    assert not bool(solution.accepted)
    assert not bool(solution.derivative_valid)
    # The truncated iterate is finite; failure is the native status, not NaN.
    for name in ("left", "right"):
        assert np.all(np.isfinite(np.asarray(solution.field(name, "u"))))


@pytest.mark.parametrize("kind", ["matching", "mortar-side-trace"])
def test_dirichlet_lifts_of_both_owners_are_composed_once(kind: str) -> None:
    cells = (3, 3) if kind == "matching" else (2, 4)
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, cells[0], 2), QUADRATIC)
    right = build_region(RegionSpec("right", "fe", 1.0, 2.0, cells[1], 2), QUADRATIC)
    coupled = couple(
        left, right, "matching" if kind == "matching" else "mortar-side-trace"
    )
    solution = cpl.solve_coupled_problem(coupled.prepared, policy=dense_policy())
    for region in (left, right):
        values = np.asarray(solution.field(region.spec.name, "u"))
        fixed = np.setdiff1d(np.arange(values.size), region.free_rows)
        data = QUADRATIC.host(region.dof_points[fixed])
        # Nonzero data on both owners: a doubled lift would read 2 g here.
        assert np.min(np.abs(data)) >= 0.5
        np.testing.assert_allclose(values[fixed], data, rtol=0.0, atol=1.0e-12)
        assert nodal_error(region, values, QUADRATIC) <= 1.0e-10


def test_elimination_refuses_charts_that_are_not_row_selections(left: Region) -> None:
    tied = build_region(
        RegionSpec("right", "fe", 1.0, 2.0, 3, 1, dirichlet="tied-corners"), SMOOTH
    )
    with pytest.raises(ValueError, match="not a row selection"):
        couple(left, tied, "matching")


def test_elimination_refuses_strongly_imposed_rows(left: Region) -> None:
    right = build_region(RegionSpec("right", "fe", 1.0, 2.0, 3, 1), SMOOTH)
    strong = np.setdiff1d(np.arange(right.dof_points.shape[0]), right.free_rows)[:2]
    columns = np.setdiff1d(np.arange(left.dof_points.shape[0]), left.free_rows)[:2]
    elimination = cpl.EliminationContribution(
        cpl.ContributionEndpoint("right", "u"),
        cpl.ContributionEndpoint("left", "u"),
        strong,
        columns,
        np.eye(2),
        law_id="tie",
        imposition_id="tie",
    )
    plan = cpl.CoupledProblemPlan(
        "strong",
        components=(left.component, right.component),
        bindings=(),
        laws=(_PublishedLaw("tie", (), (), (elimination,)),),
    )
    with pytest.raises(ValueError, match="Eliminated rows must be free rows"):
        cpl.prepare_coupled_problem(plan)


def test_elimination_refuses_strong_crosspoints_left_free_on_the_retained_side(
    left: Region,
) -> None:
    """Left fixes the crosspoints ``(1, 0)`` and ``(1, 1)``; right leaves them free.

    Eliminating left would keep its crosspoint constraint and tie nothing to
    right's free crosspoint rows, so the trace jump there is never imposed.
    Eliminating right instead ties its free crosspoints to left's data.
    """
    free_right = build_region(
        RegionSpec("right", "fe", 1.0, 2.0, 3, 1, dirichlet="none"), SMOOTH
    )
    with pytest.raises(ValueError, match="crosspoint Dirichlet value"):
        couple(left, free_right, "matching", side="left")

    coupled = couple(left, free_right, "matching", side="right")
    solution = cpl.solve_coupled_problem(coupled.prepared, policy=dense_policy())
    values = (
        np.asarray(solution.field("left", "u")),
        np.asarray(solution.field("right", "u")),
    )
    for point in ((1.0, 0.0), (1.0, 1.0)):
        rows = [
            np.flatnonzero(np.all(np.isclose(region.dof_points, point), axis=1))[0]
            for region in (left, free_right)
        ]
        assert abs(values[0][rows[0]] - values[1][rows[1]]) <= 1.0e-12
    assert bool(solution.accepted)
