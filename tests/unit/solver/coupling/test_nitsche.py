#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nitsche imposition of scalar transmission across nonmatching interfaces.

Two regions ``[0, 1] x [0, 1]`` and ``[1, 2] x [0, 1]`` solve
``-div(kappa grad u) = f`` with Dirichlet data on their exterior boundaries
and couple across ``x = 1`` through the owners' exact pointwise fluxes. The
references are independent: manufactured fields evaluated on the host, and
the analytic P1 trace-inverse constant ``|F| kappa / |K| = 2 kappa / h`` of the
structured right triangles whose vertical leg lies on the interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.flatten_util import ravel_pytree

import phydrax as phx
from tests.unit.solver.coupling._cases import (
    build_region,
    dense_policy,
    interface_binding,
    ManufacturedField,
    nodal_error,
    observed_rate,
    plate_cover,
    QUADRATIC,
    Region,
    RegionSpec,
    SMOOTH,
)


cpl = phx.solver.coupling

# Roundoff bound of exactly reproduced fields: dense LU of O(10^2..10^3)
# unknowns with condition numbers below ~1e4 (penalty ~ kappa / h) leaves
# orders of margin, while any consistency defect (>= 1e-3 here) is caught.
_EXACT = 1.0e-10


@dataclass(frozen=True, slots=True)
class Coupled:
    left: Region
    right: Region
    prepared: cpl.PreparedCoupledProblem


def _couple(left: Region, right: Region, imposition: cpl.NitscheImposition, /) -> Coupled:
    cover = plate_cover()
    binding = interface_binding(
        cover, left.component.field_space_id("u"), right.component.field_space_id("u")
    )
    law = cpl.ScalarTransmissionLaw(
        "gamma",
        binding,
        (
            cpl.TransmissionSide("left", left.spec.name, "u", left.interface),
            cpl.TransmissionSide("right", right.spec.name, "u", right.interface),
        ),
        imposition,
    )
    plan = cpl.CoupledProblemPlan(
        "nitsche",
        components=(left.component, right.component),
        bindings=(binding,),
        laws=(law,),
    )
    return Coupled(
        left, right, cpl.prepare_coupled_problem(plan, interface_owners=(cover,))
    )


def _fe_pair(
    cells: tuple[int, int],
    degree: int,
    fields: tuple[ManufacturedField, ManufacturedField],
    imposition: cpl.NitscheImposition,
    /,
    *,
    diffusivity: tuple[float, float] = (1.0, 1.0),
) -> Coupled:
    left = build_region(
        RegionSpec("left", "fe", 0.0, 1.0, cells[0], degree, diffusivity=diffusivity[0]),
        fields[0],
    )
    right = build_region(
        RegionSpec("right", "fe", 1.0, 2.0, cells[1], degree, diffusivity=diffusivity[1]),
        fields[1],
    )
    return _couple(left, right, imposition)


def _linear_field(field_id: str, slope: float, /) -> ManufacturedField:
    """``u = slope (x - 1) + y``: harmonic, continuous across ``x = 1``."""

    def value(points: Array) -> Array:
        return slope * (points[..., 0] - 1.0) + points[..., 1]

    def source(points: Array) -> Array:
        return jnp.zeros(points.shape[:-1], dtype=points.dtype)

    return ManufacturedField(field_id, value, source, 0.0, 1)


def _dense(operator: phx.linalg.AbstractLinearOperator, /) -> np.ndarray:
    """Host matrix of a small coupled operator from its actions on unit vectors."""
    flat, unravel = ravel_pytree(operator.source.zeros())
    identity = jnp.eye(flat.size, dtype=flat.dtype)
    columns = jax.vmap(lambda column: ravel_pytree(operator.mv(unravel(column)))[0])(
        identity
    )
    return np.asarray(columns).T


@pytest.mark.parametrize("variant", ["symmetric", "nonsymmetric"])
def test_diffusivity_jump_is_reproduced_exactly_on_a_nonmatching_interface(
    variant: cpl.NitscheVariant,
) -> None:
    """Piecewise-linear field with a 1:4 diffusivity jump, 3 versus 5 cells.

    ``u_minus = 4 (x - 1) + y`` and ``u_plus = (x - 1) + y`` are continuous
    with balanced fluxes ``1 * 4 = 4 * 1``; both P1 spaces contain them, so a
    consistent imposition reproduces them to roundoff.
    """
    fields = (_linear_field("minus", 4.0), _linear_field("plus", 1.0))
    coupled = _fe_pair(
        (3, 5),
        1,
        fields,
        cpl.NitscheImposition(variant, penalty_factor=2.0),
        diffusivity=(1.0, 4.0),
    )
    solution = cpl.solve_coupled_problem(coupled.prepared, policy=dense_policy())
    report = solution.interface("gamma")

    assert bool(solution.native_successful)
    assert bool(solution.accepted)
    assert nodal_error(coupled.left, solution.field("left", "u"), fields[0]) < _EXACT
    assert nodal_error(coupled.right, solution.field("right", "u"), fields[1]) < _EXACT
    assert report.names == ("law-residual", "flux-conservation", "trace-jump-l2")
    assert report.gated == (True, True, False)
    assert float(report.value("trace-jump-l2")) < _EXACT


def test_quadratic_field_is_reproduced_exactly_by_p2() -> None:
    """P2 on 3 versus 4 cells contains the quadratic manufactured field."""
    coupled = _fe_pair(
        (3, 4), 2, (QUADRATIC, QUADRATIC), cpl.NitscheImposition(penalty_factor=2.0)
    )
    solution = cpl.solve_coupled_problem(coupled.prepared, policy=dense_policy())

    assert bool(solution.accepted)
    assert nodal_error(coupled.left, solution.field("left", "u"), QUADRATIC) < _EXACT
    assert nodal_error(coupled.right, solution.field("right", "u"), QUADRATIC) < _EXACT


def test_constant_equilibrium_whose_law_terms_cancel_is_accepted() -> None:
    """``u = 1`` with zero source: every flux term and the jump vanish exactly.

    The owner reactions and Nitsche rows cancel to roundoff; the law and
    conservation certificates measure that roundoff against the uncancelled
    terms (not against the cancelled sums, which are roundoff themselves).
    """
    unit = ManufacturedField(
        "unit",
        lambda points: jnp.ones(points.shape[:-1], dtype=points.dtype),
        lambda points: jnp.zeros(points.shape[:-1], dtype=points.dtype),
        0.0,
        1,
    )
    coupled = _fe_pair((2, 3), 1, (unit, unit), cpl.NitscheImposition(penalty_factor=2.0))
    solution = cpl.solve_coupled_problem(coupled.prepared, policy=dense_policy())
    report = solution.interface("gamma")

    assert bool(solution.native_successful)
    assert nodal_error(coupled.left, solution.field("left", "u"), unit) < _EXACT
    assert nodal_error(coupled.right, solution.field("right", "u"), unit) < _EXACT
    assert bool(report.accepted(solution.tolerance))
    assert bool(solution.accepted)


@pytest.mark.parametrize(
    ("degree", "ladder", "minimum_rate"),
    [(1, (4, 8, 16), 1.8), (2, (2, 4, 8), 2.8)],
    ids=["p1", "p2"],
)
def test_nonmatching_refinement_converges_at_optimal_order(
    degree: int, ladder: tuple[int, ...], minimum_rate: float
) -> None:
    """Nodal errors of ``sin(pi x / 3) e^y`` on 2:3 nonmatching refinements.

    P_k nodal errors converge like ``h^(k+1)``; the certificate stays accepted
    on every level and the trace jump decays with the mesh. The ladders keep
    the dense reference solves within the default materialization budget.
    """
    sizes, errors, jumps = [], [], []
    for cells in ladder:
        coupled = _fe_pair(
            (cells, 3 * cells // 2),
            degree,
            (SMOOTH, SMOOTH),
            cpl.NitscheImposition(penalty_factor=2.0),
        )
        solution = cpl.solve_coupled_problem(coupled.prepared, policy=dense_policy())
        assert bool(solution.accepted)
        sizes.append(1.0 / cells)
        errors.append(
            max(
                nodal_error(coupled.left, solution.field("left", "u"), SMOOTH),
                nodal_error(coupled.right, solution.field("right", "u"), SMOOTH),
            )
        )
        jumps.append(float(solution.interface("gamma").value("trace-jump-l2")))

    assert observed_rate(np.asarray(sizes), np.asarray(errors)) >= minimum_rate, errors
    assert jumps[-1] < jumps[0] / 4.0, jumps


def test_penalty_is_the_factor_times_the_certified_p1_constants() -> None:
    """Evidence equals the analytic ``2 kappa / h`` constants and the declared weights."""
    coupled = _fe_pair(
        (4, 6),
        1,
        (SMOOTH, SMOOTH),
        cpl.NitscheImposition(penalty_factor=3.0, weights=(0.25, 0.75)),
        diffusivity=(2.0, 5.0),
    )
    evidence = coupled.prepared.laws[0].evidence
    assert isinstance(evidence, cpl.NitscheEvidence)
    minus, plus = evidence.stability
    assert minus is not None and plus is not None

    np.testing.assert_allclose(np.asarray(minus.constants), 2.0 * 2.0 * 4, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(plus.constants), 2.0 * 5.0 * 6, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(minus.cell_multiplicity), 1)
    expected = 3.0 * (0.25**2 * 16.0 + 0.75**2 * 60.0)
    np.testing.assert_allclose(evidence.penalty_range, (expected, expected), rtol=1e-12)
    assert evidence.coercivity_constant == pytest.approx(1.0 - 3.0**-0.5)
    assert evidence.flux_degrees == (0, 0)
    assert evidence.trace_degrees == (1, 1)


@pytest.mark.parametrize(
    ("variant", "factor"),
    [("symmetric", 1.05), ("nonsymmetric", 0.05)],
    ids=["symmetric-above-threshold", "nonsymmetric-small-penalty"],
)
def test_certified_variants_assemble_coercive_operators(
    variant: cpl.NitscheVariant, factor: float
) -> None:
    """The symmetric part of the assembled coupled operator is positive definite.

    Just above the certified symmetric threshold and at a tiny nonsymmetric
    penalty, the Dirichlet-anchored coupled operator stays coercive; the
    symmetric variant is also symmetric, the nonsymmetric one is not.
    """
    coupled = _fe_pair(
        (3, 4),
        2,
        (QUADRATIC, QUADRATIC),
        cpl.NitscheImposition(variant, penalty_factor=factor),
        diffusivity=(1.0, 10.0),
    )
    system, _ = coupled.prepared.linear_system()
    matrix = _dense(system.operator)
    symmetric = 0.5 * (matrix + matrix.T)
    skew = np.linalg.norm(matrix - matrix.T) / np.linalg.norm(matrix)

    assert np.linalg.eigvalsh(symmetric)[0] > 0.0
    if variant == "symmetric":
        assert skew < 1.0e-12
    else:
        assert skew > 1.0e-3


def test_symmetric_penalty_at_or_below_the_certified_bound_is_refused() -> None:
    """A factor <= 1 does not dominate the flux terms; the refusal reports the penalty."""
    with pytest.raises(ValueError, match=r"penalty_factor > 1.*penalty densities"):
        _fe_pair((3, 4), 1, (SMOOTH, SMOOTH), cpl.NitscheImposition(penalty_factor=1.0))


def test_virtual_element_side_with_flux_weight_is_refused() -> None:
    """The virtual interior gradient is not computable: no pointwise flux exists."""
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, 3, 1), SMOOTH)
    right = build_region(RegionSpec("right", "vem", 1.0, 2.0, 3, 1), SMOOTH)

    with pytest.raises(ValueError, match="virtual-element owner"):
        _couple(left, right, cpl.NitscheImposition(penalty_factor=2.0))


def test_one_sided_nitsche_couples_a_virtual_element_side_exactly() -> None:
    """Flux weight on the FE side only: the VEM side needs no flux or evidence.

    Both P1 and VEM k = 1 contain the linear field, so the one-sided
    imposition reproduces it on 3 versus 4 nonmatching cells.
    """
    field = _linear_field("linear", 2.0)
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, 3, 1), field)
    right = build_region(RegionSpec("right", "vem", 1.0, 2.0, 4, 1), field)
    coupled = _couple(
        left, right, cpl.NitscheImposition(penalty_factor=2.0, weights=(1.0, 0.0))
    )
    solution = cpl.solve_coupled_problem(coupled.prepared, policy=dense_policy())
    evidence = coupled.prepared.laws[0].evidence

    assert isinstance(evidence, cpl.NitscheEvidence)
    assert evidence.stability[1] is None
    assert evidence.flux_degrees == (0, None)
    assert bool(solution.accepted)
    assert nodal_error(left, solution.field("left", "u"), field) < _EXACT
    assert nodal_error(right, solution.field("right", "u"), field) < _EXACT


def test_owner_without_a_declared_flux_law_is_refused() -> None:
    """A mass-only form defines no conormal flux for the Nitsche terms."""
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, 3, 1), SMOOTH)
    right = build_region(RegionSpec("right", "fe", 1.0, 2.0, 3, 1), SMOOTH)
    owner = right.problem
    assert isinstance(owner, phx.equations.CompiledFiniteElementProblem)
    mass_only = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "mass", "u", (phx.equations.MassAction("u", 1.0),)
        ),
        owner.discretization,
    )
    right = Region(
        right.spec,
        cpl.VariationalComponent("right", mass_only, field="u"),
        mass_only,
        right.interface,
        right.dof_points,
        right.point_rows,
        right.free_rows,
    )

    with pytest.raises(ValueError, match="defines no conormal flux"):
        _couple(left, right, cpl.NitscheImposition(penalty_factor=2.0))
