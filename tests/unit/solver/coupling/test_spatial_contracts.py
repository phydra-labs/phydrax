#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declaration, layout, operator, and interface-quadrature contracts of coupled problems."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import FacetTraceRule
from tests.unit.solver.coupling._cases import (
    build_case,
    build_region,
    interface_binding,
    Method,
    plate_cover,
    Region,
    RegionSpec,
    SMOOTH,
    TransmissionCase,
)


if TYPE_CHECKING:
    from phydrax.solver.coupling import AbstractSpatialComponent
    from phydrax.solver.coupling._interfaces import InterfaceOwner


cpl = phx.solver.coupling


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


class _DeclaredLaw(cpl.AbstractCouplingLaw):
    """Law that only declares bindings; used for plan-level contracts."""

    law_id: str = eqx.field(static=True)
    declared: tuple[cpl.InterfaceBinding, ...]

    @property
    def bindings(self) -> tuple[cpl.InterfaceBinding, ...]:
        return self.declared

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
            state_blocks=(),
            row_blocks=(),
            contributions=(),
            impositions=(),
            certificate=certificate,
            evidence=certificate,
        )


def _parameter(binding_id: str, component: str) -> cpl.ParameterBinding:
    port = phx.ValuePort(
        binding_id, event_shape=(), component_ids=("value",), representation="scalar"
    )
    return cpl.ParameterBinding(
        binding_id,
        port,
        targets=(cpl.RuntimeInput(component, binding_id),),
        role="source",
    )


def _observation(binding_id: str, component: str, x: float) -> cpl.FieldPointObservation:
    measurement = phx.measurement
    contract = phx.SpatialCoordinateContract(phx.units.METER)
    return cpl.FieldPointObservation(
        binding_id,
        component,
        "u",
        quantity=measurement.QuantitySpec(
            "test", "potential", "potential", phx.units.ONE, "potential"
        ),
        support=measurement.PointSampleSupport(
            np.asarray([[x, 0.23, 0.0]]), ("probe",), contract
        ),
        sampling=measurement.SamplingSemantics(measurement.SpatialSamplingKind.POINT),
        field_unit=phx.units.ONE,
    )


@pytest.fixture(scope="module")
def regions() -> tuple[Region, Region]:
    return (
        build_region(RegionSpec("left", "fe", 0.0, 1.0, 3, 1), SMOOTH),
        build_region(RegionSpec("right", "fe", 1.0, 2.0, 4, 1), SMOOTH),
    )


def _binding(regions: tuple[Region, Region], interface_id: str) -> cpl.InterfaceBinding:
    left, right = regions
    return interface_binding(
        plate_cover(),
        left.component.field_space_id("u"),
        right.component.field_space_id("u"),
        interface_id=interface_id,
    )


def test_plan_orders_components_laws_and_bindings_canonically(
    regions: tuple[Region, Region],
) -> None:
    left, right = regions
    first, second = _binding(regions, "cut-a"), _binding(regions, "cut-b")
    laws = (_DeclaredLaw("zeta", (second,)), _DeclaredLaw("alpha", (first,)))
    parameters = (_parameter("source-b", "right"), _parameter("source-a", "left"))
    observations = (
        _observation("probe-b", "right", 1.63),
        _observation("probe-a", "right", 1.37),
    )
    # Every collection is declared against its canonical order.
    bindings = tuple(
        sorted((first, second), key=lambda binding: binding.binding_id, reverse=True)
    )
    plan = cpl.CoupledProblemPlan(
        "ordered",
        components=(right.component, left.component),
        bindings=bindings,
        laws=laws,
        parameters=parameters,
        observations=observations,
    )

    assert [component.name for component in plan.components] == ["left", "right"]
    assert [law.law_id for law in plan.laws] == ["alpha", "zeta"]
    assert [binding.binding_id for binding in plan.bindings] == sorted(
        (first.binding_id, second.binding_id)
    )
    assert [item.binding_id for item in plan.parameters] == ["source-a", "source-b"]
    assert [item.binding_id for item in plan.observations] == ["probe-a", "probe-b"]

    prepared = cpl.prepare_coupled_problem(
        plan,
        interface_owners=(plate_cover(),),
        parameters={"source-a": 0.0, "source-b": 0.0},
    )
    assert prepared.state_space.names == ("left", "right")
    assert [law.law_id for law in prepared.laws] == ["alpha", "zeta"]
    assert [binding.binding_id for binding in prepared.bindings] == sorted(
        (first.binding_id, second.binding_id)
    )
    assert [item.binding_id for item in prepared.parameters.bindings] == [
        "source-a",
        "source-b",
    ]
    assert [item.binding_id for item in prepared.observations] == ["probe-a", "probe-b"]


def test_plan_refuses_duplicate_and_colliding_identities(
    regions: tuple[Region, Region],
) -> None:
    left, _ = regions
    duplicate = build_region(RegionSpec("left", "fe", 1.0, 2.0, 2, 1), SMOOTH)
    with pytest.raises(ValueError, match="identities must be unique"):
        cpl.CoupledProblemPlan(
            "twins",
            components=(left.component, duplicate.component),
            bindings=(),
            laws=(),
        )
    with pytest.raises(ValueError, match="identities must be unique"):
        cpl.CoupledProblemPlan(
            "twin-laws",
            components=(left.component,),
            bindings=(),
            laws=(_DeclaredLaw("alpha", ()), _DeclaredLaw("alpha", ())),
        )
    with pytest.raises(ValueError, match="one namespace"):
        cpl.CoupledProblemPlan(
            "namespace",
            components=(left.component,),
            bindings=(),
            laws=(_DeclaredLaw("left", ()),),
        )
    with pytest.raises(ValueError, match="at least one component"):
        cpl.CoupledProblemPlan("empty", components=(), bindings=(), laws=())


def test_plan_refuses_undeclared_and_unused_bindings(
    regions: tuple[Region, Region],
) -> None:
    left, right = regions
    binding = _binding(regions, "cut")
    components = (left.component, right.component)
    with pytest.raises(ValueError, match="must be declared by the plan"):
        cpl.CoupledProblemPlan(
            "undeclared",
            components=components,
            bindings=(),
            laws=(_DeclaredLaw("alpha", (binding,)),),
        )
    with pytest.raises(ValueError, match="must be used by a law"):
        cpl.CoupledProblemPlan(
            "unused", components=components, bindings=(binding,), laws=()
        )


def test_plan_refuses_parameter_and_observation_bindings_of_unknown_components(
    regions: tuple[Region, Region],
) -> None:
    left, _ = regions
    with pytest.raises(ValueError, match="unknown components"):
        cpl.CoupledProblemPlan(
            "parameters",
            components=(left.component,),
            bindings=(),
            laws=(),
            parameters=(_parameter("source", "right"),),
        )
    with pytest.raises(ValueError, match="unknown components"):
        cpl.CoupledProblemPlan(
            "observations",
            components=(left.component,),
            bindings=(),
            laws=(),
            observations=(_observation("probe", "elsewhere", 0.5),),
        )


_LAYOUT_CASES = (
    TransmissionCase("fe-fe-p1-mortar", "fe", "fe", 3, 4, 1, "mortar-side-trace", SMOOTH),
    TransmissionCase(
        "fe-vem-p2-mortar", "fe", "vem", 2, 3, 2, "mortar-side-trace", SMOOTH
    ),
    TransmissionCase(
        "fe-fe-p1-dp0", "fe", "fe", 3, 4, 1, "mortar-discontinuous", SMOOTH, 0
    ),
    TransmissionCase("fe-fe-p1-matching", "fe", "fe", 3, 3, 1, "matching", SMOOTH),
)


def _free_size(method: Method, cells: int, degree: int) -> int:
    """Solve coordinates of one region: all rows minus Dirichlet rows.

    Dirichlet data holds on the boundary except the interface interior.
    """
    edges = 2 * cells * (cells + 1)
    match method:
        case "fe":
            rows = (degree * cells + 1) ** 2
        case "vem":
            rows = (
                (cells + 1) ** 2
                + (degree - 1) * edges
                + cells**2 * (degree * (degree - 1) // 2)
            )
    boundary = 4 * degree * cells
    return rows - boundary + (degree * cells - 1)


def _assert_layout(case: TransmissionCase, prepared: cpl.PreparedCoupledProblem) -> None:
    """Components by name, then law unknowns; sizes from the case geometry."""
    left = _free_size(case.left_method, case.left_cells, case.degree)
    right = _free_size(case.right_method, case.right_cells, case.degree)
    multiplier = case.expected_multiplier_size()

    if multiplier is None:
        right -= case.expected_eliminated_rows()
        assert prepared.state_space.names == ("left", "right")
        assert prepared.row_space.names == ("left", "right")
        evidence = prepared.laws[0].evidence
        assert isinstance(evidence, cpl.EliminationEvidence)
        assert evidence.eliminated_rows == case.expected_eliminated_rows()
    else:
        assert prepared.state_space.names == ("left", "right", "gamma")
        assert prepared.row_space.names == ("left", "right", "gamma")
        law_state = prepared.state_space.spaces[2]
        law_rows = prepared.row_space.spaces[2]
        assert isinstance(law_state, phx.linalg.BlockSpace)
        assert isinstance(law_rows, phx.linalg.BlockSpace)
        assert law_state.names == ("multiplier",)
        assert law_rows.names == ("constraint",)
        assert law_state.size == law_rows.size == multiplier
        evidence = prepared.laws[0].evidence
        assert isinstance(evidence, cpl.MortarEvidence)
        assert evidence.multiplier_dimension == multiplier
        assert evidence.numerical_rank == multiplier
    for index, (name, size) in enumerate((("left", left), ("right", right))):
        state = prepared.state_space.spaces[index]
        rows = prepared.row_space.spaces[index]
        assert isinstance(state, phx.linalg.BlockSpace)
        assert isinstance(rows, phx.linalg.BlockSpace)
        assert state.names == rows.names == ("u",), name
        assert state.size == rows.size == size, name
    assert prepared.execution == "linear"
    assert prepared.nullspace_policy is None


def _random_state(
    space: phx.linalg.BlockSpace, seed: int
) -> tuple[tuple[Array, ...], ...]:
    rng = np.random.default_rng(seed)
    return space.unflatten(jnp.asarray(rng.standard_normal(space.size)))


def _assert_weak_transpose(prepared: cpl.PreparedCoupledProblem) -> None:
    """``<J x, y> = <x, J^T y>`` and the materialized ``J^T`` equals ``J`` transposed."""
    operator = prepared.weak_operator()
    state = _random_state(prepared.state_space, 0)
    rows = _random_state(prepared.row_space, 1)

    forward = prepared.row_space.flatten(operator.mv(state))
    backward = prepared.state_space.flatten(operator.transpose_mv(rows))
    left = float(jnp.dot(forward, prepared.row_space.flatten(rows)))
    right = float(jnp.dot(prepared.state_space.flatten(state), backward))
    scale = float(
        jnp.linalg.norm(forward) * jnp.linalg.norm(prepared.row_space.flatten(rows))
    )
    assert abs(left - right) <= 1.0e-12 * scale

    size = prepared.state_space.size
    identity = jnp.eye(size, dtype=jnp.float64)
    columns = np.stack(
        [
            np.asarray(
                prepared.row_space.flatten(
                    operator.mv(prepared.state_space.unflatten(identity[index]))
                )
            )
            for index in range(size)
        ],
        axis=1,
    )
    transposed = np.stack(
        [
            np.asarray(
                prepared.state_space.flatten(
                    operator.transpose_mv(prepared.row_space.unflatten(identity[index]))
                )
            )
            for index in range(prepared.row_space.size)
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        transposed, columns.T, rtol=0.0, atol=1.0e-12 * np.max(np.abs(columns))
    )


def _assert_affine_residual(prepared: cpl.PreparedCoupledProblem) -> None:
    """``A z - b`` of the native system is the (Riesz-identified) coupled residual."""
    system, rhs = prepared.linear_system()
    weak = prepared.weak_operator()
    for seed in (2, 3):
        state = _random_state(prepared.state_space, seed)
        residual = prepared.state_space.flatten(
            prepared.state_space.inverse_riesz(prepared.residual(state))
        )
        affine = prepared.state_space.flatten(system.operator.mv(state)) - (
            prepared.state_space.flatten(rhs)
        )
        weak_residual = prepared.row_space.flatten(
            prepared.residual(state)
        ) - prepared.row_space.flatten(weak.mv(state))
        at_zero = prepared.row_space.flatten(
            prepared.residual(prepared.state_space.zeros())
        )
        scale = float(jnp.linalg.norm(residual))
        np.testing.assert_allclose(affine, residual, rtol=0.0, atol=1.0e-11 * scale)
        np.testing.assert_allclose(weak_residual, at_zero, rtol=0.0, atol=1.0e-11 * scale)


@pytest.mark.parametrize("case", _LAYOUT_CASES, ids=lambda case: case.case_id)
def test_prepared_layout_and_block_operator_identities(case: TransmissionCase) -> None:
    """Layout, weak-Jacobian transpose, and residual consistency of one preparation.

    The checks share one expensive two-owner preparation.
    """
    prepared = build_case(case).prepared
    _assert_layout(case, prepared)
    _assert_weak_transpose(prepared)
    _assert_affine_residual(prepared)


@dataclass(frozen=True, slots=True)
class QuadratureCase:
    """Two facet partitions of the cut ``x = 1`` and their common refinement.

    ``segments`` counts the distinct breakpoints ``i / left_cells`` and
    ``j / right_cells`` minus one.
    """

    case_id: str
    left_method: Method
    right_method: Method
    degree: int
    left_cells: int
    right_cells: int
    segments: int


_QUADRATURE_CASES = (
    QuadratureCase("fe-fe-p1-3-4", "fe", "fe", 1, 3, 4, 6),
    QuadratureCase("fe-fe-p2-3-4", "fe", "fe", 2, 3, 4, 6),
    QuadratureCase("fe-vem-k1-3-5", "fe", "vem", 1, 3, 5, 7),
    QuadratureCase("fe-vem-k2-2-5", "fe", "vem", 2, 2, 5, 6),
    QuadratureCase("fe-fe-p2-matching", "fe", "fe", 2, 5, 5, 5),
)

# Interface traces a(y), b(y) of degree <= p for each trace degree p.
_TRACES = {
    1: (np.polynomial.Polynomial([3.0, 1.0]), np.polynomial.Polynomial([0.0, 5.0])),
    2: (
        np.polynomial.Polynomial([3.0, 1.0, -3.0]),
        np.polynomial.Polynomial([0.0, 5.0, 0.5]),
    ),
}


def _coefficients(region: Region, trace: np.polynomial.Polynomial, degree: int) -> Array:
    """Point-value coefficients of ``trace(y) + (x - 1) q(y)`` (degree ``degree``)."""
    rows = region.point_rows
    x, y = region.dof_points[rows, 0], region.dof_points[rows, 1]
    transverse = 2.0 if degree == 1 else 1.0 + y
    values = np.zeros((region.dof_points.shape[0],), dtype=np.float64)
    values[rows] = trace(y) + (x - 1.0) * transverse
    return jnp.asarray(values)


@pytest.mark.parametrize("case", _QUADRATURE_CASES, ids=lambda case: case.case_id)
def test_interface_quadrature_integrates_trace_products_exactly(
    case: QuadratureCase,
) -> None:
    degree = case.degree
    left = build_region(
        RegionSpec("left", case.left_method, 0.0, 1.0, case.left_cells, degree), SMOOTH
    )
    right = build_region(
        RegionSpec("right", case.right_method, 1.0, 2.0, case.right_cells, degree),
        SMOOTH,
    )
    rule = FacetTraceRule("gauss-lobatto-legendre", points=degree + 1)
    quadrature = cpl.prepare_interface_quadrature(
        left.component.prepare_side_trace("u", left.interface, rule=rule),
        right.component.prepare_side_trace("u", right.interface, rule=rule),
        exact_degree=2 * degree,
    )

    evidence = quadrature.evidence
    assert evidence.segment_count == case.segments
    assert evidence.first_measure == pytest.approx(1.0, abs=1.0e-14)
    assert evidence.second_measure == pytest.approx(1.0, abs=1.0e-14)
    assert evidence.common_measure == pytest.approx(1.0, abs=1.0e-14)
    np.testing.assert_allclose(np.asarray(evidence.first_coverage), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(evidence.second_coverage), 1.0, atol=1.0e-12)
    assert evidence.maximum_gap <= 1.0e-12
    assert evidence.maximum_normal_defect <= 1.0e-12
    assert quadrature.exact_degree >= 2 * degree
    points = np.asarray(quadrature.points)
    np.testing.assert_allclose(points[:, 0], 1.0, atol=1.0e-14)
    # Normals point out of the first (minus) side, along +x.
    np.testing.assert_allclose(
        np.asarray(quadrature.normals),
        np.tile([1.0, 0.0], (quadrature.point_count, 1)),
        atol=1.0e-14,
    )
    assert float(jnp.sum(quadrature.weights)) == pytest.approx(1.0, abs=1.0e-14)

    traces = _TRACES[degree]
    values = tuple(
        side.values(_coefficients(region, trace, degree))
        for side, region, trace in zip(
            quadrature.sides, (left, right), traces, strict=True
        )
    )
    for value, trace in zip(values, traces, strict=True):
        np.testing.assert_allclose(np.asarray(value), trace(points[:, 1]), atol=1.0e-12)
    product = (traces[0] * traces[1]).integ()
    measured = float(jnp.sum(quadrature.weights * values[0] * values[1]))
    assert measured == pytest.approx(product(1.0) - product(0.0), rel=1.0e-13)


def test_interface_resampling_pullback_is_the_exact_transpose() -> None:
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, 3, 2), SMOOTH)
    right = build_region(RegionSpec("right", "vem", 1.0, 2.0, 4, 2), SMOOTH)
    rule = FacetTraceRule("gauss-lobatto-legendre", points=3)
    quadrature = cpl.prepare_interface_quadrature(
        left.component.prepare_side_trace("u", left.interface, rule=rule),
        right.component.prepare_side_trace("u", right.interface, rule=rule),
        exact_degree=4,
    )
    rng = np.random.default_rng(7)
    for side, region in zip(quadrature.sides, (left, right), strict=True):
        coefficients = jnp.asarray(rng.standard_normal(region.dof_points.shape[0]))
        covector = jnp.asarray(rng.standard_normal(quadrature.point_count))
        forward = float(jnp.dot(side.values(coefficients), covector))
        backward = float(jnp.dot(coefficients, side.pullback(covector)))
        assert forward == pytest.approx(backward, rel=1.0e-13, abs=1.0e-13)
        pulled = np.asarray(side.pullback(covector))
        # Only rows whose basis functions have a trace on x = 1 receive weight.
        off_interface = ~np.isclose(region.dof_points[:, 0], 1.0)
        assert np.all(pulled[off_interface] == 0.0)
        assert np.any(pulled[~off_interface] != 0.0)


def _p1_traces() -> tuple[
    phx.discretization.PreparedTraceAction, phx.discretization.PreparedTraceAction
]:
    """Left trace facets ``[0, 1/2], [1/2, 1]``; right ``[k/4, (k+1)/4]`` on ``x = 1``."""
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, 2, 1), SMOOTH)
    right = build_region(RegionSpec("right", "fe", 1.0, 2.0, 4, 1), SMOOTH)
    rule = FacetTraceRule("gauss-lobatto-legendre", points=2)
    return (
        left.component.prepare_side_trace("u", left.interface, rule=rule),
        right.component.prepare_side_trace("u", right.interface, rule=rule),
    )


def _facet_at(sites: np.ndarray, low: float) -> int:
    (index,) = np.flatnonzero(np.isclose(np.min(sites[:, :, 1], axis=1), low))
    return int(index)


# Right facets rewritten as (y_low, y_high) of one facet at x = 1, keyed by the
# y_low of the original facet they replace.
_BROKEN_PARTITIONS = {
    # Duplicate [0, 1/4] in place of [1/4, 1/2]: the overlap and the gap have
    # equal length, so the summed covered length of every facet is exact.
    "overlap-and-gap": {0.25: (0.0, 0.25)},
    "gap": {0.0: (0.1, 0.25)},
    "overlap": {0.0: (0.0, 0.5)},
}


@pytest.mark.parametrize("edits", _BROKEN_PARTITIONS.values(), ids=_BROKEN_PARTITIONS)
def test_interface_quadrature_refuses_a_partition_that_is_not_a_tiling(
    edits: dict[float, tuple[float, float]],
) -> None:
    left, right = _p1_traces()
    sites = np.asarray(right.sites, dtype=np.float64).copy()
    for replaced, (low, high) in edits.items():
        sites[_facet_at(sites, replaced)] = [[1.0, high], [1.0, low]]
    broken = eqx.tree_at(lambda trace: trace.sites, right, jnp.asarray(sites))

    with pytest.raises(ValueError, match="do not cover each other"):
        cpl.prepare_interface_quadrature(left, broken, exact_degree=2)


def test_boundary_panels_that_are_not_a_tiling_are_refused() -> None:
    from phydrax.solver.coupling._interface_quadrature import (
        prepare_boundary_panel_resampling,
    )

    _, volume = _p1_traces()
    # Panels [0, 1/4] twice, none on [1/4, 1/2]: total panel length is exact.
    starts = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 0.75]])
    directions = np.tile([0.0, 0.25], (4, 1))
    normals = np.tile([1.0, 0.0], (4, 1))
    points = (
        starts[:, None, :]
        + np.asarray([0.25, 0.75])[None, :, None] * directions[:, None, :]
    )

    with pytest.raises(ValueError, match="do not cover each other"):
        prepare_boundary_panel_resampling(volume, (starts, directions, normals), points)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64], ids=["f32", "c64"])
def test_interface_resampling_evaluates_in_the_trace_precision(dtype: type) -> None:
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, 3, 2), SMOOTH)
    right = build_region(RegionSpec("right", "fe", 1.0, 2.0, 4, 2), SMOOTH)
    rule = FacetTraceRule("gauss-lobatto-legendre", points=3)
    quadrature = cpl.prepare_interface_quadrature(
        left.component.prepare_side_trace("u", left.interface, rule=rule),
        right.component.prepare_side_trace("u", right.interface, rule=rule),
        exact_degree=4,
    )
    real, imaginary = _TRACES[2]
    y = np.asarray(quadrature.points)[:, 1]
    expected = real(y) + (1j * imaginary(y) if dtype is jnp.complex64 else 0.0)
    rng = np.random.default_rng(3)
    covector = rng.standard_normal(quadrature.point_count)
    for side in quadrature.sides:
        site_y = np.asarray(side.trace.sites)[:, :, 1]
        exact = real(site_y) + (1j * imaginary(site_y) if dtype is jnp.complex64 else 0.0)
        site_values = jnp.asarray(exact, dtype=dtype)
        values = side.resampling.apply(site_values)
        pulled = side.resampling.transpose(jnp.asarray(covector, dtype=dtype))

        assert values.dtype == dtype
        assert pulled.dtype == dtype
        # The site data is a degree-2 polynomial of y; resampling reproduces it.
        np.testing.assert_allclose(np.asarray(values), expected, rtol=0.0, atol=2.0e-6)
        forward = np.vdot(np.asarray(values, dtype=np.complex128), covector)
        backward = np.vdot(np.asarray(site_values, dtype=np.complex128), pulled)
        assert forward == pytest.approx(backward, rel=1.0e-5)


def test_single_component_plan_reproduces_the_native_solve(
    regions: tuple[Region, Region],
) -> None:
    left, _ = regions
    plan = cpl.CoupledProblemPlan(
        "alone", components=(left.component,), bindings=(), laws=()
    )
    prepared = cpl.prepare_coupled_problem(plan)
    policy = phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    solution = cpl.solve_coupled_problem(prepared, policy=policy)
    system, rhs = left.problem.linear_system(None)
    native = phx.linalg.solve(system, rhs, policy=policy)

    assert prepared.state_space.names == ("left",)
    assert bool(native.successful)
    assert bool(solution.native_successful)
    assert bool(solution.accepted)
    assert solution.interfaces == ()
    np.testing.assert_allclose(
        np.asarray(solution.field("left", "u")),
        np.asarray(left.problem.expand(native.value, None)),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_galerkin_boundary_component_publishes_the_exterior_relation() -> None:
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.4, 1.3], [0.0, 1.0]])
    galerkin = phx.operators.prepare_scalar_laplace_galerkin_2d(
        phx.operators.ClosedPolygonalCurve2D(vertices, source_id="pentagon")
    )
    component = cpl.GalerkinBoundaryComponent("exterior", galerkin)
    relation = galerkin.exterior_relation
    trace_space, conormal_space, constant_space = relation.source.spaces
    rng = np.random.default_rng(11)

    def sample(space: phx.linalg.AbstractVectorSpace) -> Array:
        return space.unflatten(jnp.asarray(rng.standard_normal(space.size)))

    trace, conormal, constant = (
        sample(trace_space),
        sample(conormal_space),
        sample(constant_space),
    )
    expected = relation.mv((trace_space.zeros(), conormal, constant))
    published = component.linear_operator(None).mv((conormal, constant))
    residual = component.residual((conormal, constant), None)
    dirichlet = relation.mv((trace, conormal_space.zeros(), constant_space.zeros()))

    assert component.state_space.names == ("conormal", "far_field_constant")
    assert component.row_space.names == ("exterior_boundary_equation", "total_conormal")
    for block, reference in zip(published, expected, strict=True):
        np.testing.assert_allclose(np.asarray(block), np.asarray(reference), atol=1.0e-13)
    for block, reference in zip(residual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(block), np.asarray(reference), atol=1.0e-13)
    np.testing.assert_allclose(
        np.asarray(component.dirichlet_operator().mv(trace)),
        np.asarray(dirichlet[0]),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(np.asarray(dirichlet[1]), 0.0, atol=1.0e-13)
    assert component.nullspace(None) is None
    assert component.boundary_impositions() == ()
