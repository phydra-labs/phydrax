#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.conditions import Periodic
from phydrax.discretization import (
    point_sbp_report,
    PointBoundaryCondition,
    PointBoundaryKind,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCloudPoissonPlan,
    PointCollocationPlan,
    PointDiffusionOperator,
    PreparedPointCloudDiscretization,
    SBPDerivativePlan,
    SBPGridNorm,
    TensorGridPlan,
    UniformAxisSpec,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    PointGhostLayerPlan,
    PointSBPDerivatives,
    prepare_point_sbp_derivatives,
    prepare_tensor_point_sbp,
)
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.domain import HyperRectangle, Interval1d, PeriodicIdentification
from phydrax.linalg import FailurePolicy, GMRES, LinearSolvePolicy, TolerancePolicy


_LINE_SEAM = PeriodicIdentification(Interval1d(0.0, 1.0), "x")
_LINE_ADDRESS = MortonAddressPlan.from_periodic_identifications(
    (_LINE_SEAM,), maximum_depth=12
)


_ENDS = np.asarray([0, 10])
_UNPRECONDITIONED = LinearSolvePolicy(
    GMRES(restart=11),
    tolerance=TolerancePolicy(relative=1e-9, absolute=1e-10, max_steps=1000),
    failure=FailurePolicy("error"),
)


def _cloud(count: int = 11) -> tuple[Array, PreparedPointCloudDiscretization]:
    x = jnp.linspace(0.0, 1.0, count)
    boundary = (jnp.arange(count) == 0) | (jnp.arange(count) == count - 1)
    normals = jnp.zeros((count, 1)).at[0, 0].set(-1.0).at[-1, 0].set(1.0)
    cloud = PointCloudPlan(
        x[:, None],
        jnp.linspace(0.5, 1.5, count) / count,
        boundary_mask=boundary,
        boundary_normals=normals,
        boundary_quadrature_weights=boundary.astype(jnp.float64),
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=5,
    ).prepare()
    return x, cloud


def _ends(
    kind: PointBoundaryKind, values: Array, *, robin: float | None = None
) -> PointBoundaryPlan:
    return PointBoundaryPlan(
        (
            PointBoundaryCondition(
                kind,
                _ENDS,
                values[_ENDS],
                label="ends",
                normals=None if kind == "dirichlet" else np.asarray([[-1.0], [1.0]]),
                robin_coefficient=robin,
            ),
        ),
        row_count=values.shape[0],
    )


def test_collocated_variable_coefficient_sign_and_dirichlet_lift() -> None:
    x, cloud = _cloud()
    exact = 1 + x + x * x
    k = 1 + x
    # -((1+x)(1+2x))' = -3-4x.
    source = -3 - 4 * x
    prepared = PointCloudPoissonPlan(
        cloud, _ends("dirichlet", exact), linear_policy=_UNPRECONDITIONED
    ).prepare(k)
    result = prepared.solve(source)
    np.testing.assert_allclose(result.values, exact, atol=2e-8)
    assert float(result.residual_norm) < 1e-8
    assert float(result.boundary_residual_norm) < 1e-9
    assert int(result.status) == 0
    assert bool(result.diagnostics.converged)
    assert bool(result.compatible)
    assert bool(result.successful)
    assert bool(result.diffusivity_evidence.successful)


def test_reused_default_preconditioner_with_changed_dirichlet_data() -> None:
    x, cloud = _cloud()
    prepared = PointCloudPoissonPlan(cloud, _ends("dirichlet", x * x)).prepare()
    source = jnp.full_like(x, -2.0)
    initial = prepared.solve(source)
    np.testing.assert_allclose(initial.values, x * x, atol=2e-8)
    shifted_field = x * x + x + 2.0
    shifted = prepared.solve(source, boundary_values={"ends": shifted_field[_ENDS]})
    np.testing.assert_allclose(shifted.values, shifted_field, atol=2e-8)
    assert bool(shifted.successful)


def test_dissipative_weighted_energy_and_numeric_refresh() -> None:
    x, cloud = _cloud()
    value = jnp.stack((jnp.sin(3 * x), jnp.cos(5 * x)), axis=1)
    diffusion = PointDiffusionOperator(cloud, 1 + x, form="dissipative")
    gradients = cloud.partial_derivative(value, axis=0)
    expected = -jnp.sum(
        cloud.quadrature_weights[:, None] * (1 + x)[:, None] * gradients**2
    )
    np.testing.assert_allclose(diffusion.energy_rate(value), expected, atol=1e-10)
    u = jnp.sin(2 * x) + x
    prepared = PointCloudPoissonPlan(
        cloud, _ends("dirichlet", u), form="dissipative"
    ).prepare(1 + x)
    source = -diffusion.mv(u)
    result = prepared.solve(source)
    assert bool(result.linear_result.successful)
    assert not result.continuum_consistent and not bool(result.successful)
    np.testing.assert_allclose(result.values, u, atol=2e-8)
    refreshed = prepared.refresh(2 * (1 + x))
    refreshed_result = refreshed.solve(2 * source)
    assert bool(refreshed_result.linear_result.successful)
    assert not refreshed_result.continuum_consistent and not bool(
        refreshed_result.successful
    )
    np.testing.assert_allclose(refreshed_result.values, u, atol=2e-8)
    # Reusing old coefficients cannot solve the changed physical problem.
    assert float(jnp.max(jnp.abs(prepared.solve(2 * source).values - u))) > 1e-3


def test_robin_conormal_and_neumann_algebraic_compatibility() -> None:
    x, cloud = _cloud()
    u = 1 + x + x * x
    conormal = jnp.zeros_like(x).at[0].set(-1.0).at[-1].set(3.0)
    robin = PointCloudPoissonPlan(
        cloud, _ends("robin", conormal + u, robin=1.0), linear_policy=_UNPRECONDITIONED
    ).prepare()
    np.testing.assert_allclose(robin.solve(jnp.full_like(x, -2.0)).values, u, atol=2e-8)
    neumann = PointCloudPoissonPlan(
        cloud,
        _ends("neumann", conormal),
        linear_policy=_UNPRECONDITIONED,
        compatibility="refuse",
    ).prepare()
    (gauge,) = neumann.plan.gauges
    result = neumann.solve(jnp.full_like(x, -2.0))
    np.testing.assert_allclose(result.values, u - u[gauge], atol=2e-8)
    assert float(result.gauge_residual) < 1e-10
    with pytest.raises(Exception, match="incompatible Neumann"):
        neumann.solve(jnp.full_like(x, -1.0))
    projected = PointCloudPoissonPlan(
        cloud,
        _ends("neumann", conormal),
        linear_policy=_UNPRECONDITIONED,
        compatibility="project",
    ).prepare()
    corrected = projected.solve(jnp.full_like(x, -1.0))
    assert not bool(corrected.compatible)
    np.testing.assert_allclose(corrected.source_correction[1:-1], -1.0, atol=2e-8)
    np.testing.assert_allclose(
        corrected.source_correction[jnp.asarray([0, -1])], 0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        corrected.values, u - u[projected.plan.gauges[0]], atol=2e-8
    )
    assert float(corrected.residual_norm) < 1e-8
    assert bool(corrected.successful)


type SBPLine = tuple[Array, PreparedPointCloudDiscretization, PointSBPDerivatives]


@pytest.fixture(scope="module")
def sbp_line() -> SBPLine:
    x = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
    boundary = jnp.asarray([True, False, True], dtype=jnp.bool_)
    normals = jnp.asarray([[-1.0], [0.0], [1.0]], dtype=jnp.float64)
    cloud = PointCloudPlan(
        x[:, None],
        jnp.asarray([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=jnp.float64),
        boundary_mask=boundary,
        boundary_normals=normals,
        boundary_quadrature_weights=boundary.astype(jnp.float64),
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=3,
    ).prepare()
    # Global quadratic interpolation on three Lobatto nodes and Simpson mass
    # obeys the full SBP identity, not merely its constant-vector contraction.
    assert point_sbp_report(cloud).passed
    sbp = prepare_point_sbp_derivatives(cloud, reproduction_degree=2, tolerance=1e-10)
    assert bool(sbp.successful)
    assert not replace(sbp, native_binding_id="native-tensor-sbp").stable_realization
    assert not sbp.stable_realization
    return x, cloud, sbp


@pytest.mark.parametrize("kind", ["neumann", "robin"], ids=["flux", "flux-and-value"])
@pytest.mark.parametrize("degree", [1, 2], ids=["linear", "quadratic-nonzero-source"])
def test_dissipative_natural_boundary_retains_volume_source(
    kind: PointBoundaryKind,
    degree: int,
    sbp_line: SBPLine,
) -> None:
    x, cloud, sbp = sbp_line
    normals = cloud.plan.boundary_normals
    exact = x**degree
    alpha = 2.0 if kind == "robin" else 0.0
    conditions = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "dirichlet", np.asarray([0], dtype=np.intp), 0.0, label="left"
            ),
            PointBoundaryCondition(
                kind,
                np.asarray([2], dtype=np.intp),
                degree + alpha,
                label="right",
                normals=normals[-1:],
                robin_coefficient=alpha if kind == "robin" else None,
            ),
        ),
        row_count=3,
    )
    prepared = PointCloudPoissonPlan(
        cloud,
        conditions,
        form="dissipative",
        linear_policy=_UNPRECONDITIONED,
        sbp=sbp,
    ).prepare()
    result = prepared.solve(jnp.full_like(x, -degree * (degree - 1)))
    assert not result.continuum_consistent
    assert not bool(result.successful)
    assert bool(result.algebraically_successful)
    # Replacing the natural row by a pointwise derivative equation, or dropping
    # its nonzero volume source, fails this independent quadratic oracle.
    np.testing.assert_allclose(result.values, exact, rtol=0.0, atol=2e-9)


def test_generic_q1_sbp_does_not_authorize_the_unstable_continuum_claim() -> None:
    x = jnp.linspace(-1.0, 1.0, 6, dtype=jnp.float64)
    measure = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=jnp.float64)
    normals = jnp.asarray([[-1.0], [0.0], [0.0], [0.0], [0.0], [1.0]], dtype=jnp.float64)
    cloud = PointCloudPlan(
        x[:, None],
        (2.0 / 5.0) * jnp.asarray([0.5, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=jnp.float64),
        boundary_mask=measure > 0.0,
        boundary_normals=normals,
        boundary_quadrature_weights=measure,
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=6,
    ).prepare()
    sbp = prepare_point_sbp_derivatives(cloud, reproduction_degree=1)
    exact = jnp.exp(x)
    diffusivity = 2.0 + 0.1 * x
    endpoints = np.asarray([0, 5], dtype=np.int32)
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "robin",
                endpoints,
                exact[endpoints] * (1.0 + diffusivity[endpoints] * normals[endpoints, 0]),
                normals=normals[endpoints],
                robin_coefficient=1.0,
                label="ends",
            ),
        ),
        row_count=6,
    )
    result = (
        PointCloudPoissonPlan(
            cloud,
            boundary,
            form="dissipative",
            sbp=sbp,
            linear_policy=_UNPRECONDITIONED,
        )
        .prepare(diffusivity)
        .solve(-exact * (diffusivity + 0.1))
    )
    assert bool(sbp.successful) and not sbp.stable_realization
    assert bool(result.algebraically_successful)
    assert not result.continuum_consistent and not bool(result.successful)
    # Independent continuum data exposes what reproduction/Green identities
    # and a tiny linear residual cannot certify.
    assert float(jnp.max(jnp.abs(result.values - exact))) > 0.1


@pytest.fixture(scope="module")
def tensor_sbp_line() -> SBPLine:
    grid = TensorGridPlan((UniformAxisSpec(6),), axis_names=("x",)).prepare(
        jnp.asarray([[0.0], [1.0]], dtype=jnp.float64)
    )
    derivative = SBPDerivativePlan(grid, "x", interior_order=2).prepare()
    norm = SBPGridNorm((derivative,))
    # Point IDs, not incidental cloud ordering, declare the tensor flatten map.
    rows = np.asarray([5, 2, 0, 4, 1, 3], dtype=np.int32)
    points = grid.points[rows]
    boundary = (rows == 0) | (rows == 5)
    normals = np.asarray(derivative.boundary_diagonal)[rows, None]
    cloud = PointCloudPlan(
        points,
        norm.weights.reshape(-1)[rows],
        point_ids=rows,
        boundary_mask=boundary,
        boundary_normals=normals,
        boundary_quadrature_weights=boundary.astype(np.float64),
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=5,
    ).prepare()
    return points[:, 0], cloud, prepare_tensor_point_sbp(cloud, (derivative,))


def test_tensor_sbp_row_map_authorizes_the_actual_native_realization(
    tensor_sbp_line: SBPLine,
) -> None:
    x, cloud, sbp = tensor_sbp_line
    assert sbp.stable_realization
    native = sbp.native_derivatives[0]
    rows = np.asarray(cloud.plan.point_ids)
    values = jnp.exp(x)
    native_values = jnp.zeros_like(values).at[rows].set(values)
    actual = sbp.bind(cloud).apply(values, (1,))
    np.testing.assert_allclose(
        actual, native.operator.mv(native_values)[rows], atol=1e-13
    )
    left = np.flatnonzero(rows == 0)
    right = np.flatnonzero(rows == 5)
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition("dirichlet", left, 0.0, label="left"),
            PointBoundaryCondition(
                "robin",
                right,
                3.0,
                label="right",
                normals=cloud.plan.boundary_normals[right],
                robin_coefficient=2.0,
            ),
        ),
        row_count=6,
    )
    result = (
        PointCloudPoissonPlan(
            cloud,
            boundary,
            form="dissipative",
            sbp=sbp,
            linear_policy=_UNPRECONDITIONED,
        )
        .prepare()
        .solve(jnp.zeros_like(x))
    )
    assert result.continuum_consistent and bool(result.successful)
    np.testing.assert_allclose(result.values, x, atol=2e-9)


def test_tensor_sbp_refuses_stale_and_foreign_owner_bindings(
    tensor_sbp_line: SBPLine,
) -> None:
    _, cloud, sbp = tensor_sbp_line
    stale = eqx.tree_at(
        lambda value: value.weights, sbp, tuple(value * 1.01 for value in sbp.weights)
    )
    with pytest.raises(ValueError, match="different cloud or numerical revision"):
        stale.bind(cloud)
    grid = TensorGridPlan((UniformAxisSpec(6),), axis_names=("x",)).prepare(
        jnp.asarray([[1.0], [2.0]], dtype=jnp.float64)
    )
    foreign = SBPDerivativePlan(grid, "x", interior_order=2).prepare()
    with pytest.raises(ValueError, match="geometry"):
        prepare_tensor_point_sbp(cloud, (foreign,))
    rebound = eqx.tree_at(lambda value: value.native_derivatives, sbp, (foreign,))
    assert not rebound.stable_realization
    with pytest.raises(ValueError, match="different numerical revision"):
        rebound.bind(cloud)


def test_sbp_derivatives_refuse_a_cloud_with_changed_volume_cubature(
    sbp_line: SBPLine,
) -> None:
    _, cloud, sbp = sbp_line
    changed = PointCloudPlan(
        cloud.points,
        2.0 * cloud.quadrature_weights,
        boundary_mask=cloud.plan.boundary_mask,
        boundary_normals=cloud.plan.boundary_normals,
        boundary_quadrature_weights=cloud.plan.boundary_quadrature_weights,
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=3,
    ).prepare()
    with pytest.raises(ValueError, match="different cloud or numerical revision"):
        PointDiffusionOperator(changed, form="dissipative", sbp=sbp)


def test_sbp_derivatives_refuse_the_collocated_diffusion_route(
    sbp_line: SBPLine,
) -> None:
    _, cloud, sbp = sbp_line
    with pytest.raises(ValueError, match="single-sided dissipative diffusion"):
        PointDiffusionOperator(cloud, form="collocated", sbp=sbp)


def test_sbp_exact_identity_reports_non_sbp_and_refuses_budget() -> None:
    _, cloud = _cloud()
    report = point_sbp_report(cloud)
    assert not report.passed
    assert report.maximum_green_residual > 1e-4
    with pytest.raises(ValueError, match="maximum_coefficients"):
        point_sbp_report(cloud, maximum_coefficients=1)


def test_invalid_robin_and_nonpositive_or_nonsymmetric_diffusivity_are_refused() -> None:
    x, cloud = _cloud()
    with pytest.raises(ValueError, match="Zero-coefficient Robin"):
        _ends("robin", x, robin=0.0)
    with pytest.raises(Exception, match="positive"):
        PointDiffusionOperator(cloud, 0.0)
    _, plane = _square_cloud(2, 5, 0)
    with pytest.raises(Exception, match="symmetric"):
        PointDiffusionOperator(
            plane, jnp.asarray([[2.0, 0.5], [0.1, 2.0]]), kind="tensor"
        )
    with pytest.raises(Exception, match="positive definite"):
        PointDiffusionOperator(
            plane, jnp.asarray([[1.0, 2.0], [2.0, 1.0]]), kind="tensor"
        )


def test_status_failure_mode_publishes_refusal_instead_of_raising() -> None:
    x, cloud = _cloud()
    conormal = jnp.zeros_like(x).at[0].set(-1.0).at[-1].set(3.0)
    status_policy = LinearSolvePolicy(
        GMRES(restart=11),
        tolerance=TolerancePolicy(relative=1e-9, absolute=1e-10, max_steps=1000),
        failure=FailurePolicy("status"),
    )
    prepared = PointCloudPoissonPlan(
        cloud, _ends("neumann", conormal), linear_policy=status_policy
    ).prepare()
    accepted = prepared.solve(jnp.full_like(x, -2.0))
    assert bool(accepted.successful)
    # Incompatible data: the same refusal the error mode raises is published.
    refused = prepared.solve(jnp.full_like(x, -1.0))
    assert not bool(refused.compatible)
    assert not bool(refused.successful)
    assert float(refused.residual_norm) > float(refused.residual_tolerance)


def _boundary_free_periodic(count: int) -> float:
    rng = np.random.default_rng(2)
    x = (np.arange(count) + rng.uniform(-0.2, 0.2, count)) / count
    cloud = PointCloudPlan(
        x[:, None],
        np.full(count, 1.0 / count),
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
        neighbors=7,
        address=_LINE_ADDRESS,
    ).prepare()
    plan = PointCloudPoissonPlan(
        cloud, PointBoundaryPlan((), row_count=count), compatibility="project"
    )
    (gauge,) = plan.gauges
    exact = jnp.cos(2 * jnp.pi * jnp.asarray(x))
    result = plan.prepare().solve((2 * jnp.pi) ** 2 * exact)
    assert bool(result.successful)
    return float(jnp.max(jnp.abs(result.values - (exact - exact[gauge]))))


def test_boundary_free_periodic_cloud_gauges_its_floating_component() -> None:
    coarse = _boundary_free_periodic(24)
    fine = _boundary_free_periodic(48)
    assert fine < coarse / 3 and fine < 1e-2
    # Without boundary rows, collocated rows on a bounded cloud have affine
    # null modes and are refused.
    x = jnp.linspace(0.0, 1.0, 11) ** 1.3
    open_cloud = PointCloudPlan(
        x[:, None],
        jnp.full((11,), 1.0 / 11),
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=5,
    ).prepare()
    with pytest.raises(ValueError, match="fully periodic"):
        PointCloudPoissonPlan(open_cloud, PointBoundaryPlan((), row_count=11))
    _, bounded = _cloud()
    with pytest.raises(ValueError, match="declared boundary rows"):
        PointCloudPoissonPlan(bounded, PointBoundaryPlan((), row_count=11))


def test_periodic_address_keeps_one_representative_per_seam_orbit() -> None:
    # The closed box carries both endpoints; the half-open periodic address
    # identifies them, so the upper-face copy is a duplicate node.
    closed = np.linspace(0.0, 1.0, 11)[:, None]
    with pytest.raises(ValueError, match="one representative per seam orbit"):
        PointCloudPlan(
            closed,
            np.full(11, 1.0 / 11),
            stencil=LocalStencilPolicy(polynomial_degree=2),
            neighbors=5,
            address=_LINE_ADDRESS,
        )


def test_boundary_free_dissipative_form_imposes_natural_boundary() -> None:
    x = jnp.linspace(0.0, 1.0, 11) ** 1.3
    cloud = PointCloudPlan(
        x[:, None],
        jnp.full((11,), 1.0 / 11),
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=5,
    ).prepare()
    plan = PointCloudPoissonPlan(
        cloud,
        PointBoundaryPlan((), row_count=11),
        form="dissipative",
        linear_policy=_UNPRECONDITIONED,
    )
    (gauge,) = plan.gauges
    k = 1.0 + x
    u = jnp.sin(3 * x)
    # The source lies in the range of the symmetric quadrature-adjoint
    # operator, so the discrete problem is exactly compatible; the solution
    # is unique up to the declared gauge only if ker(G) is the constants.
    source = -PointDiffusionOperator(cloud, k, form="dissipative").mv(u)
    result = plan.prepare(k).solve(source)
    assert bool(result.compatible)
    assert bool(result.linear_result.successful)
    assert not result.continuum_consistent and not bool(result.successful)
    np.testing.assert_allclose(result.values, u - u[gauge], atol=1e-8)


def test_boundary_rows_have_single_explicit_owner() -> None:
    x, cloud = _cloud()
    first = PointBoundaryCondition("dirichlet", _ENDS, 0.0, label="left-and-right")
    corner = PointBoundaryCondition(
        "neumann", np.asarray([10]), 0.0, label="right", normals=np.asarray([[1.0]])
    )
    with pytest.raises(ValueError, match="corner ownership"):
        PointBoundaryPlan((first, corner), row_count=11)
    partial = PointBoundaryPlan(
        (PointBoundaryCondition("dirichlet", np.asarray([0]), 0.0, label="left"),),
        row_count=11,
    )
    with pytest.raises(ValueError, match="declared boundary rows"):
        PointCloudPoissonPlan(cloud, partial)
    del x


def test_dirichlet_lift_does_not_weaken_original_equation_tolerance() -> None:
    from examples.point_cloud_poisson import run_workflow

    record = run_workflow(
        size=128,
        dimension=2,
        seed=0,
        manufactured="exponential",
        stencil=LocalStencilPolicy(approximation="gmls", polynomial_degree=2),
        neighbors=18,
    )
    residual = record["true_original_residual"]
    tolerance = record["residual_tolerance"]
    assert isinstance(residual, float)
    assert isinstance(tolerance, float)
    assert residual <= tolerance
    assert record["successful"] is True


def test_independent_neumann_refinement_and_reported_source_projection() -> None:
    from examples.point_cloud_poisson import run_workflow

    coarse = run_workflow(
        size=64, dimension=2, seed=0, manufactured="exponential", boundary_kind="neumann"
    )
    fine = run_workflow(
        size=128, dimension=2, seed=0, manufactured="exponential", boundary_kind="neumann"
    )
    coarse_error = coarse["maximum_solution_error"]
    fine_error = fine["maximum_solution_error"]
    coarse_correction = coarse["maximum_source_correction"]
    fine_correction = fine["maximum_source_correction"]
    assert isinstance(coarse_error, float) and isinstance(fine_error, float)
    assert isinstance(coarse_correction, float) and isinstance(fine_correction, float)
    assert fine_error < coarse_error
    assert 0.0 < fine_correction < coarse_correction
    assert fine["compatible"] is False
    assert fine["successful"] is True


# Irregular unit-cube clouds; references below are independent analytic fields
# differentiated by jax autodiff, never the stencils under test.


def _square_cloud(
    dimension: int, side: int, seed: int, *, degree: int = 3
) -> tuple[np.ndarray, PreparedPointCloudDiscretization]:
    grid = np.meshgrid(*([np.linspace(0.0, 1.0, side)] * dimension), indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in grid], axis=1)
    on_face = np.isclose(points, 0.0) | np.isclose(points, 1.0)
    boundary = np.any(on_face, axis=1)
    rng = np.random.default_rng(seed)
    points[~boundary] += rng.uniform(
        -0.15, 0.15, (np.count_nonzero(~boundary), dimension)
    ) / (side - 1)
    normals = np.where(np.isclose(points, 1.0), 1.0, 0.0) - np.where(
        np.isclose(points, 0.0), 1.0, 0.0
    )
    lengths = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0)
    spacing = 1.0 / (side - 1)
    cloud = PointCloudPlan(
        points,
        np.full(points.shape[0], 1.0 / points.shape[0]),
        boundary_mask=boundary,
        boundary_normals=normals / lengths,
        boundary_quadrature_weights=np.where(boundary, spacing ** (dimension - 1), 0.0),
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=degree),
    ).prepare()
    return points, cloud


def _tensor(x: Array) -> Array:
    dimension = x.shape[0]
    base = jnp.diag(1.0 + 0.5 * x**2)
    coupling = 0.3 + 0.1 * jnp.sum(x)
    off = (
        coupling
        * (jnp.ones((dimension, dimension)) - jnp.eye(dimension))
        / (dimension - 1)
    )
    return base + off


def _exact(x: Array) -> Array:
    return jnp.exp(0.5 * x[0]) * jnp.sin(1.0 + jnp.sum(x[1:]))


def _flux_source(
    tensor: Callable[[Array], Array], exact: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    def flux(x: Array) -> Array:
        return tensor(x) @ jax.grad(exact)(x)

    return lambda x: -jnp.trace(jax.jacfwd(flux)(x))


@pytest.mark.parametrize("dimension,side", ((2, 11), (3, 6)), ids=("2d", "3d"))
def test_anisotropic_variable_tensor_diffusion_irregular_cloud(
    dimension: int, side: int
) -> None:
    points, cloud = _square_cloud(dimension, side, 3)
    x = jnp.asarray(points)
    tensor = jax.vmap(_tensor)(x)
    exact = jax.vmap(_exact)(x)
    source = jax.vmap(_flux_source(_tensor, _exact))(x)
    rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    boundary = PointBoundaryPlan(
        (PointBoundaryCondition("dirichlet", rows, exact[rows], label="walls"),),
        row_count=points.shape[0],
    )
    plan = PointCloudPoissonPlan(cloud, boundary, diffusivity="tensor")
    result = plan.prepare(tensor).solve(source)
    error = float(jnp.max(jnp.abs(result.values - exact)))
    assert bool(result.successful)
    assert error < 5e-3
    evidence = result.diffusivity_evidence
    assert float(evidence.minimum_eigenvalue) > 0.0
    assert float(evidence.symmetry_defect) == 0.0
    # Fault adequacy: discarding the off-diagonal coupling is visibly wrong.
    diagonal = tensor * jnp.eye(dimension)
    dropped = plan.prepare(diagonal).solve(source)
    dropped_error = float(jnp.max(jnp.abs(dropped.values - exact)))
    assert dropped_error > 10 * error


def test_mixed_rows_corner_ownership_and_normal_orientation() -> None:
    points, cloud = _square_cloud(2, 12, 5)
    x = jnp.asarray(points)
    scalar = lambda p: (1.0 + 0.3 * p[0] + 0.2 * p[1]) * jnp.eye(2)
    k = jax.vmap(lambda p: scalar(p)[0, 0])(x)
    exact = jax.vmap(_exact)(x)
    source = jax.vmap(_flux_source(scalar, _exact))(x)
    gradient = jax.vmap(jax.grad(_exact))(x)
    left = np.flatnonzero(np.isclose(points[:, 0], 0.0))
    right = np.flatnonzero(np.isclose(points[:, 0], 1.0))
    bottom_top = np.flatnonzero(
        (np.isclose(points[:, 1], 0.0) | np.isclose(points[:, 1], 1.0))
        & ~np.isclose(points[:, 0], 0.0)
        & ~np.isclose(points[:, 0], 1.0)
    )
    right_normal = np.tile([[1.0, 0.0]], (right.size, 1))
    vertical = np.where(points[bottom_top, 1:2] > 0.5, 1.0, -1.0)
    face_normal = np.concatenate((np.zeros_like(vertical), vertical), axis=1)

    def conormal(rows: np.ndarray, normals: np.ndarray) -> Array:
        return k[rows] * jnp.sum(gradient[rows] * normals, axis=1)

    def boundary(flip: float) -> PointBoundaryPlan:
        return PointBoundaryPlan(
            (
                # Corners on x=0 and x=1 belong to the x faces explicitly.
                PointBoundaryCondition("dirichlet", left, exact[left], label="x0"),
                PointBoundaryCondition(
                    "neumann",
                    right,
                    conormal(right, right_normal),
                    label="x1",
                    normals=flip * right_normal,
                ),
                PointBoundaryCondition(
                    "robin",
                    bottom_top,
                    conormal(bottom_top, face_normal) + 2.0 * exact[bottom_top],
                    label="y",
                    normals=face_normal,
                    robin_coefficient=2.0,
                ),
            ),
            row_count=points.shape[0],
        )

    # Flux rows use the ghost-layer PDE+BC route: square flux-only rows are
    # spectrally unstable here and refused by the default assessment.
    declared = boundary(1.0)
    plan = PointCloudPoissonPlan(
        cloud, declared, ghosts=PointGhostLayerPlan(declared).prepare(cloud)
    )
    result = plan.prepare(k).solve(source)
    assert bool(result.successful)
    assert plan.gauges == ()
    assert float(jnp.max(jnp.abs(result.values - exact))) < 2e-3
    # A flipped normal places its ghosts among the cloud's own samples.
    with pytest.raises(ValueError, match="minimum_separation"):
        PointGhostLayerPlan(boundary(-1.0)).prepare(cloud)


def _two_segments() -> tuple[Array, PreparedPointCloudDiscretization]:
    first = np.linspace(0.0, 1.0, 11)
    x = np.concatenate((first, first + 3.0))
    boundary = np.isin(np.arange(22), (0, 10, 11, 21))
    normals = np.zeros((22, 1))
    normals[[0, 11], 0] = -1.0
    normals[[10, 21], 0] = 1.0
    cloud = PointCloudPlan(
        x[:, None],
        np.full(22, 0.1),
        boundary_mask=boundary,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=5,
    ).prepare()
    return jnp.asarray(x), cloud


def test_disconnected_neumann_components_have_own_gauges_and_compatibility() -> None:
    x, cloud = _two_segments()
    rows = np.asarray([0, 10, 11, 21])
    normals = np.asarray([[-1.0], [1.0], [-1.0], [1.0]])
    # u = x² on each segment: -u'' = -2, conormal u' n.
    flux = np.asarray([0.0, 2.0, -6.0, 8.0])
    boundary = PointBoundaryPlan(
        (PointBoundaryCondition("neumann", rows, flux, label="ends", normals=normals),),
        row_count=22,
    )
    plan = PointCloudPoissonPlan(
        cloud, boundary, linear_policy=_UNPRECONDITIONED, compatibility="project"
    )
    assert len(plan.gauges) == 2
    assert len(set(np.asarray(plan.component_labels)[list(plan.gauges)].tolist())) == 2
    prepared = plan.prepare()
    exact = x * x
    result = prepared.solve(jnp.full_like(x, -2.0))
    # Gauges are ordered by component; the first owns the segment at x=0.
    first, second = plan.gauges
    assert first < 11 <= second
    reference = jnp.where(x < 2.0, exact - exact[first], exact - exact[second])
    np.testing.assert_allclose(result.values, reference, atol=2e-8)
    assert bool(result.compatible)
    # Incompatible data on the second segment only is corrected there only.
    source = jnp.full_like(x, -2.0).at[12:21].set(-1.0)
    corrected = prepared.solve(source)
    residuals = np.asarray(corrected.component_compatibility_residual)
    assert residuals[0] < 1e-8 < residuals[1]
    np.testing.assert_allclose(corrected.source_correction[1:10], 0.0, atol=1e-8)
    assert float(jnp.min(jnp.abs(corrected.source_correction[12:21]))) > 1e-2
    assert bool(corrected.successful)
    with pytest.raises(ValueError, match="one interior row per floating component"):
        PointCloudPoissonPlan(cloud, boundary, gauges=(3, 5))


def _seam_cloud(
    x: np.ndarray, *, address: MortonAddressPlan | None = None
) -> PreparedPointCloudDiscretization:
    count = x.size
    boundary = np.isin(np.arange(count), (0, count - 1))
    normals = np.zeros((count, 1))
    normals[0, 0], normals[-1, 0] = -1.0, 1.0
    return PointCloudPlan(
        x[:, None],
        np.full(count, 1.0 / count),
        boundary_mask=boundary,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
        neighbors=7,
        address=address,
    ).prepare()


def _seam_rows(
    count: int,
    seam: PeriodicIdentification | Periodic,
    values: float | None = None,
    *,
    rows: int = -1,
    partners: int = 0,
) -> PointBoundaryPlan:
    return PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "periodic",
                np.asarray([rows % count]),
                values,
                label="seam",
                seam=seam,
                partners=np.asarray([partners % count]),
            ),
        ),
        row_count=count,
    )


def _periodic(
    count: int, boundary: PointBoundaryPlan, jump: float
) -> tuple[float, float, float]:
    """Max error against ``cos(2 pi x) + g x``, compatibility correction, and seam jump.

    The oracle solves ``-u'' = (2 pi)^2 cos(2 pi x)`` with the canonical seam
    relation ``u(1) - u(0) = g`` and continuous flux ``u'(1) = u'(0)``.
    """
    x = np.linspace(0.0, 1.0, count)
    plan = PointCloudPoissonPlan(_seam_cloud(x), boundary, compatibility="project")
    assert len(plan.gauges) == 1
    wave = jnp.cos(2 * jnp.pi * jnp.asarray(x))
    result = plan.prepare().solve((2 * jnp.pi) ** 2 * wave)
    assert bool(result.successful)
    exact = wave + jump * jnp.asarray(x)
    reference = exact - exact[plan.gauges[0]]
    return (
        float(jnp.max(jnp.abs(result.values - reference))),
        float(jnp.max(jnp.abs(result.source_correction))),
        float(result.values[-1] - result.values[0]),
    )


def test_periodic_identification_converges_on_translation_periodic_problem() -> None:
    coarse, coarse_correction, _ = _periodic(21, _seam_rows(21, _LINE_SEAM), 0.0)
    fine, fine_correction, _ = _periodic(41, _seam_rows(41, _LINE_SEAM), 0.0)
    assert fine < coarse / 3 and fine < 2e-2
    # The reported discrete compatibility correction vanishes under refinement.
    assert fine_correction < coarse_correction / 3


def test_periodic_seam_jump_is_canonical_upper_minus_lower() -> None:
    jump = 0.7
    relation = Periodic("u", _LINE_SEAM.pairing(), target=jump)
    error, _, seam = _periodic(41, _seam_rows(41, relation), jump)
    # The reversed convention lower - upper = g would miss the oracle by O(g).
    assert error < 2e-2
    assert seam == pytest.approx(jump, abs=1e-7)
    explicit = _seam_rows(41, _LINE_SEAM, jump).conditions[0]
    related = _seam_rows(41, relation).conditions[0]
    np.testing.assert_array_equal(explicit.values, related.values)
    assert explicit.condition_id != related.condition_id


def test_periodic_seam_rows_refuse_noncanonical_realizations() -> None:
    count = 11
    closed = _seam_cloud(np.linspace(0.0, 1.0, count))
    with pytest.raises(ValueError, match=r"target \(upper\) face"):
        PointCloudPoissonPlan(closed, _seam_rows(count, _LINE_SEAM, rows=0, partners=-1))
    # A periodic address already identifies the seam half-open.
    half_open = _seam_cloud(np.arange(count) / count, address=_LINE_ADDRESS)
    with pytest.raises(ValueError, match="already identifies this seam"):
        PointCloudPoissonPlan(half_open, _seam_rows(count, _LINE_SEAM))
    antiperiodic = Periodic("u", _LINE_SEAM.pairing(), transport=-1.0)
    with pytest.raises(ValueError, match="identity seam transport"):
        _seam_rows(count, antiperiodic)
    with pytest.raises(ValueError, match="omit values"):
        _seam_rows(count, Periodic("u", _LINE_SEAM.pairing(), target=0.5), 0.5)
    with pytest.raises(ValueError, match="omit normals"):
        PointBoundaryCondition(
            "periodic",
            np.asarray([count - 1]),
            label="seam",
            seam=_LINE_SEAM,
            partners=np.asarray([0]),
            normals=np.asarray([[1.0]]),
        )


def test_periodic_seam_partners_must_be_face_map_images() -> None:
    side = 5
    points, cloud = _square_cloud(2, side, 0)
    seam = PeriodicIdentification(
        HyperRectangle(np.zeros(2), np.ones(2)), "x", component=0
    )
    transverse = np.arange(1, side - 1)
    rows = (side - 1) * side + transverse
    partners = transverse
    walls = np.setdiff1d(
        np.flatnonzero(np.asarray(cloud.plan.boundary_mask)),
        np.concatenate((rows, partners)),
    )

    def boundary(images: np.ndarray) -> PointBoundaryPlan:
        return PointBoundaryPlan(
            (
                PointBoundaryCondition(
                    "periodic", rows, label="seam", seam=seam, partners=images
                ),
                PointBoundaryCondition("dirichlet", walls, label="walls"),
            ),
            row_count=points.shape[0],
        )

    with pytest.raises(ValueError, match="face-map image"):
        PointCloudPoissonPlan(cloud, boundary(np.roll(partners, 1)))
    matched = PointCloudPoissonPlan(cloud, boundary(partners), stability="diagnostic")
    # Dirichlet walls anchor the seam-identified cylinder: no floating component.
    assert matched.gauges == ()


def _oversampled(side: int) -> float:
    points, cloud = _square_cloud(2, side, 1, degree=3)
    targets_side = 2 * side - 1
    grid = np.meshgrid(*([np.linspace(0.0, 1.0, targets_side)] * 2), indexing="ij")
    targets = np.stack([axis.reshape(-1) for axis in grid], axis=1)
    on_boundary = np.any(np.isclose(targets, 0.0) | np.isclose(targets, 1.0), axis=1)
    collocation = PointCollocationPlan(
        targets,
        np.full(targets.shape[0], 1.0 / targets.shape[0]),
        boundary_mask=on_boundary,
        boundary_weight=float(targets_side),
    ).prepare(cloud)
    k = lambda p: (1.0 + 0.2 * p[0]) * jnp.eye(2)
    rows = np.flatnonzero(on_boundary)
    target = jnp.asarray(targets)
    exact_targets = jax.vmap(_exact)(target)
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "dirichlet",
                rows,
                exact_targets[rows],
                label="walls",
                measure=np.full(rows.size, 1.0 / (targets_side - 1)),
            ),
        ),
        row_count=targets.shape[0],
    )
    plan = PointCloudPoissonPlan(cloud, boundary, collocation=collocation)
    assert plan.route == "oversampled-least-squares"
    diffusivity = jax.vmap(lambda p: k(p)[0, 0])(jnp.asarray(points))
    result = plan.prepare(diffusivity).solve(jax.vmap(_flux_source(k, _exact))(target))
    assert bool(result.successful)
    exact = jax.vmap(_exact)(jnp.asarray(points))
    return float(jnp.max(jnp.abs(result.values - exact)))


def test_oversampled_least_squares_converges_on_nonpolynomial_oracle() -> None:
    coarse = _oversampled(7)
    fine = _oversampled(11)
    assert fine < 0.5 * coarse
    assert fine < 5e-3


def test_oversampled_route_refuses_bad_row_weighting_and_undersampling() -> None:
    points, cloud = _square_cloud(2, 7, 1)
    on_boundary = np.asarray(cloud.plan.boundary_mask)
    with pytest.raises(ValueError, match="positive physical measure"):
        PointCollocationPlan(
            points,
            np.zeros(points.shape[0]),
            boundary_mask=on_boundary,
            boundary_weight=1.0,
        )
    with pytest.raises(ValueError, match="at least as many target rows"):
        PointCollocationPlan(
            points[:20],
            np.ones(20),
            boundary_mask=on_boundary[:20],
            boundary_weight=1.0,
        ).prepare(cloud)
    collocation = PointCollocationPlan(
        points,
        np.full(points.shape[0], 1.0 / points.shape[0]),
        boundary_mask=on_boundary,
        boundary_weight=1e12,
    ).prepare(cloud)
    rows = np.flatnonzero(on_boundary)
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "dirichlet", rows, 0.0, label="walls", measure=np.ones(rows.size)
            ),
        ),
        row_count=points.shape[0],
    )
    with pytest.raises(ValueError, match="maximum_weight_ratio"):
        PointCloudPoissonPlan(cloud, boundary, collocation=collocation)
