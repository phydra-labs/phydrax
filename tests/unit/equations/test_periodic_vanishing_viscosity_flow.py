from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    AxisDomain,
    FourierBasisPlan,
    HermitianSpectralCoordinates,
    OversamplingDealiasingPlan,
    PaddingDealiasingPlan,
    PseudospectralMethodPlan,
    SpectralVanishingViscosityPlan,
    TensorSpectralDiscretization,
    TensorSpectralPlan,
)
from phydrax.equations import (
    compile_periodic_incompressible_flow,
    CompiledIncompressibleSpectralDynamics,
    IncompressibleFlowProblem,
    PeriodicAdvectiveCourantRestriction,
    PeriodicAlgebraicLESPlan,
    PeriodicFourierGridFilterPlan,
)
from phydrax.equations._barotropic_beta_plane import BarotropicBetaPlane
from phydrax.equations._dynamic_les import (
    DynamicLESProvenance,
    DynamicSmagorinskyPlan,
    ExactDenominatorRegularization,
    GlobalDynamicLESAveraging,
    NonnegativeBackscatterClip,
)
from phydrax.equations._les_closures import (
    LESParameterProvenance,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)
from phydrax.equations._periodic_dynamic_les import (
    PeriodicDynamicLESPlan,
    PeriodicFourierTestFilterPlan,
)
from phydrax.solver import (
    CourantGuardedETDRKMethod,
    ETDRKMethod,
    PreparedCourantGuardedETDRKMethod,
)
from phydrax.statistical_dynamics import (
    BetaPlaneCumulantSystem,
    ForcingCovariance,
    InteractionPartition,
)


def _space(dimension: int, count: int = 8) -> TensorSpectralDiscretization:
    return TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for _ in range(dimension)),
        axis_names=("x", "y", "z")[:dimension],
        field_name="velocity",
    ).prepare((AxisDomain.periodic(0.0, 2.0 * np.pi),) * dimension)


def _svv() -> SpectralVanishingViscosityPlan:
    return SpectralVanishingViscosityPlan(
        strength=0.3,
        order=1.0,
        cutoff_scale=0.5,
        cutoff_exponent=0.4,
        kernel="tadmor_step",
    )


def _compiled(
    dimension: int, *, viscosity: float = 0.0, svv: bool = True
) -> CompiledIncompressibleSpectralDynamics:
    return compile_periodic_incompressible_flow(
        IncompressibleFlowProblem(dimension, viscosity),
        _space(dimension),
        PseudospectralMethodPlan(dealiasing=PaddingDealiasingPlan(2)),
        spectral_viscosity=_svv() if svv else None,
    )


def _shear(dynamics: CompiledIncompressibleSpectralDynamics) -> Array:
    grid = jnp.meshgrid(
        *(axis.nodes for axis in dynamics.discretization.axes), indexing="ij"
    )
    first = jnp.sin(2.0 * grid[1])
    return dynamics.project_state(
        jnp.stack(
            (first,)
            + (jnp.zeros_like(first),) * (dynamics.problem.spatial_dimension - 1),
            axis=-1,
        )
    )


@pytest.mark.parametrize("dimension", [2, 3], ids=["2d", "3d"])
def test_euler_shear_mode_etdrk4_exact_svv_decay(dimension: int) -> None:
    dynamics = _compiled(dimension)
    initial = _shear(dynamics)
    coordinates = HermitianSpectralCoordinates(
        dynamics.discretization, component_shape=(dimension,)
    )
    method = ETDRKMethod(4).prepare(dynamics.semilinear_drift, coordinates=coordinates)
    step = eqx.filter_jit(method.step)
    state = initial
    for index in range(3):
        result = step(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(0.1 * index),
            state,
            jnp.asarray(0.1),
            None,
        )
        assert bool(result.successful)
        state = result.accepted_state
    # K=3, epsilon_K=0.3/3, |k|=2 and Q=1 on the 2pi torus.
    expected = np.exp(-0.3 / 3.0 * 4.0 * 0.3) * initial
    np.testing.assert_allclose(state, expected, atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("dimension", [2, 3], ids=["2d", "3d"])
def test_molecular_and_svv_energy_identity(dimension: int) -> None:
    dynamics = _compiled(dimension, viscosity=0.02)
    state = _shear(dynamics)
    stage = dynamics.stage(0.0, state)
    diagnostics = dynamics.diagnostics(0.0, state)
    assert stage.rates.spectral_viscosity_rate is not None
    np.testing.assert_allclose(
        stage.rates.molecular_rate, -0.02 * 4.0 * state, atol=1e-13
    )
    np.testing.assert_allclose(
        stage.rates.spectral_viscosity_rate, -0.4 * state, atol=1e-13
    )
    np.testing.assert_allclose(
        stage.rates.total_rate, dynamics(jnp.asarray(0.0), state, None), atol=1e-13
    )
    np.testing.assert_allclose(
        diagnostics.spectral_viscosity_energy_rate,
        -diagnostics.spectral_viscosity_dissipation,
        atol=1e-13,
    )
    assert float(diagnostics.spectral_viscosity_dissipation) > 0.0
    assert dynamics.spectral_viscosity is not None
    assert diagnostics.spectral_viscosity_id == dynamics.spectral_viscosity.prepared_id
    np.testing.assert_allclose(diagnostics.energy_balance_defect, 0.0, atol=1e-12)
    assert bool(diagnostics.finite)


def test_absent_svv_has_zero_diagnostics_and_no_stage_operator() -> None:
    dynamics = _compiled(2, svv=False)
    state = _shear(dynamics)
    diagnostics = dynamics.diagnostics(0.0, state)
    assert dynamics.stage(0.0, state).rates.spectral_viscosity_rate is None
    assert diagnostics.spectral_viscosity_id is None
    assert diagnostics.spectral_viscosity_energy_rate == 0.0
    assert diagnostics.spectral_viscosity_dissipation == 0.0
    active = _compiled(2)
    assert active.compilation_id != dynamics.compilation_id
    assert active.semilinear_drift.drift_id != dynamics.semilinear_drift.drift_id
    assert (
        active.discretization_bundle.bundle_id != dynamics.discretization_bundle.bundle_id
    )


@pytest.mark.parametrize(
    "courant", [0.2, 2.0 * np.sqrt(2.0) / np.pi], ids=["small", "rk4-limit"]
)
def test_retained_grid_courant_bound(courant: float) -> None:
    dynamics = _compiled(2)
    state = _shear(dynamics)
    restriction = dynamics.advective_courant_restriction(
        state, maximum_courant_number=courant
    )
    velocity = np.asarray(dynamics.discretization.reconstruct(state))
    widths = np.full((2,), 2.0 * np.pi / 8.0, dtype=np.float64)
    frequency = np.max(np.sum(np.abs(velocity) / widths, axis=-1))
    np.testing.assert_allclose(restriction.courant_frequency, frequency, rtol=1e-14)
    np.testing.assert_allclose(restriction.maximum_step, courant / frequency, rtol=1e-14)
    assert restriction.reference_model == "frozen_uniform_advection_rk4"
    assert not restriction.nonlinear_stability_certified
    assert restriction.compilation_id == dynamics.compilation_id
    assert bool(restriction.finite)


def test_zero_velocity_courant_bound_is_infinite() -> None:
    dynamics = _compiled(2)
    restriction = dynamics.advective_courant_restriction(
        jnp.zeros(dynamics.state_shape, dtype=jnp.complex128), maximum_courant_number=0.5
    )
    assert float(restriction.maximum_step) == np.inf
    assert bool(restriction.finite)


@pytest.mark.parametrize("accepted", [False, True], ids=["reject-above", "accept-below"])
def test_courant_guard_transaction(accepted: bool) -> None:
    dynamics = _compiled(2)
    state = _shear(dynamics)
    method = CourantGuardedETDRKMethod(
        ETDRKMethod(4), maximum_courant_number=0.5
    ).prepare(
        dynamics,
        coordinates=HermitianSpectralCoordinates(
            dynamics.discretization, component_shape=(2,)
        ),
    )
    bound = method.step_restriction(state).maximum_step
    duration = bound * (0.5 if accepted else 1.1)
    result = eqx.filter_jit(method.step)(
        jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0), state, duration, None
    )
    assert bool(result.successful) == accepted
    if accepted:
        expected = (
            ETDRKMethod(4)
            .prepare(
                dynamics.semilinear_drift, coordinates=method.base_method.coordinates
            )
            .step(
                jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0), state, duration, None
            )
        )
        np.testing.assert_array_equal(result.accepted_state, expected.accepted_state)
    else:
        np.testing.assert_array_equal(result.accepted_state, state)
        np.testing.assert_array_equal(result.candidate_state, state)
        assert float(result.residual) > 0.0
        assert int(result.iterations) == 0


def test_courant_guard_rejects_nonfinite_evidence() -> None:
    dynamics = _compiled(2)
    method = CourantGuardedETDRKMethod(
        ETDRKMethod(4), maximum_courant_number=0.5
    ).prepare(
        dynamics,
        coordinates=HermitianSpectralCoordinates(
            dynamics.discretization, component_shape=(2,)
        ),
    )
    state = _shear(dynamics).at[0, 0, 0].set(jnp.inf)
    result = eqx.filter_jit(method.step)(
        jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0), state, jnp.asarray(0.01), None
    )
    np.testing.assert_array_equal(result.accepted_state, state)
    assert not bool(result.successful)
    assert float(result.residual) == np.inf


@pytest.mark.parametrize(
    "courant,error",
    [
        (0.0, ValueError),
        (-0.1, ValueError),
        (1.0, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
        ("0.5", TypeError),
    ],
    ids=["zero", "negative", "above-reference", "nan", "inf", "bool", "string"],
)
def test_courant_number_refusals(courant: float | str, error: type[Exception]) -> None:
    with pytest.raises(error):
        CourantGuardedETDRKMethod(
            ETDRKMethod(4),
            maximum_courant_number=courant,  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(error):
        _compiled(2).advective_courant_restriction(
            jnp.zeros((8, 8, 2), dtype=jnp.complex128),
            maximum_courant_number=courant,  # ty: ignore[invalid-argument-type]
        )


def test_courant_guard_refuses_etdrk2() -> None:
    with pytest.raises(ValueError, match="ETDRK4"):
        CourantGuardedETDRKMethod(ETDRKMethod(2), maximum_courant_number=0.5)


@pytest.mark.parametrize(
    "compilation_id,error",
    [
        ("", ValueError),
        (" ", ValueError),
        (" dns ", ValueError),
        (True, TypeError),
        (1, TypeError),
    ],
    ids=["empty", "blank", "padded", "bool", "integer"],
)
def test_courant_restriction_requires_compilation_identity(
    compilation_id: str | int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        PeriodicAdvectiveCourantRestriction(
            courant_frequency=jnp.asarray(1.0),
            maximum_step=jnp.asarray(0.5),
            maximum_courant_number=jnp.asarray(0.5),
            finite=jnp.asarray(True),
            compilation_id=compilation_id,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_model", "other-reference"),
        ("nonlinear_stability_certified", True),
    ],
    ids=["reference_model", "nonlinear_stability_certified"],
)
def test_courant_restriction_fixed_reference_evidence(
    field: str, value: str | bool
) -> None:
    with pytest.raises(TypeError):
        PeriodicAdvectiveCourantRestriction(
            courant_frequency=jnp.asarray(1.0),
            maximum_step=jnp.asarray(0.5),
            maximum_courant_number=jnp.asarray(0.5),
            finite=jnp.asarray(True),
            compilation_id="dns",
            **{field: value},
        )


def test_courant_restriction_requires_explicit_identity() -> None:
    with pytest.raises(TypeError):
        PeriodicAdvectiveCourantRestriction(  # ty: ignore[missing-argument]
            courant_frequency=jnp.asarray(1.0),
            maximum_step=jnp.asarray(0.5),
            maximum_courant_number=jnp.asarray(0.5),
            finite=jnp.asarray(True),
        )


@pytest.mark.parametrize("argument", ["dynamics", "coordinates", "both"])
def test_courant_prepare_rejects_wrong_kinds(
    argument: Literal["dynamics", "coordinates", "both"],
) -> None:
    dynamics = _compiled(2)
    coordinates = HermitianSpectralCoordinates(
        dynamics.discretization, component_shape=(2,)
    )
    method = CourantGuardedETDRKMethod(ETDRKMethod(4), maximum_courant_number=0.5)
    with pytest.raises(TypeError):
        method.prepare(
            None if argument in ("dynamics", "both") else dynamics,  # ty: ignore[invalid-argument-type]
            coordinates=None if argument in ("coordinates", "both") else coordinates,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("argument", ["plan", "base_method", "dynamics"])
def test_prepared_courant_guard_rejects_wrong_kinds(
    argument: Literal["plan", "base_method", "dynamics"],
) -> None:
    dynamics = _compiled(2)
    coordinates = HermitianSpectralCoordinates(
        dynamics.discretization, component_shape=(2,)
    )
    plan = CourantGuardedETDRKMethod(ETDRKMethod(4), maximum_courant_number=0.5)
    base = plan.base_method.prepare(dynamics.semilinear_drift, coordinates=coordinates)
    with pytest.raises(TypeError):
        PreparedCourantGuardedETDRKMethod(
            None if argument == "plan" else plan,  # ty: ignore[invalid-argument-type]
            None if argument == "base_method" else base,  # ty: ignore[invalid-argument-type]
            None if argument == "dynamics" else dynamics,  # ty: ignore[invalid-argument-type]
        )


def _filter(name: str) -> ResolvedLESFilter:
    return ResolvedLESFilter(
        name,
        family="sharp-fourier-projection",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )


@pytest.mark.parametrize("dynamic", [False, True], ids=["algebraic", "dynamic"])
def test_svv_and_courant_guard_refuse_les(dynamic: bool) -> None:
    space = _space(3)
    resolved_filter = _filter("resolved")
    parameters = LESParameterProvenance(
        resolved_filter,
        space.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    method = PseudospectralMethodPlan(dealiasing=PaddingDealiasingPlan(2))
    grid_filter = PeriodicFourierGridFilterPlan(resolved_filter)
    algebraic = None
    dynamic_plan = None
    test = None
    if dynamic:
        test_filter = _filter("test")
        provenance = DynamicLESProvenance(
            parameters, test_filter, np.full((3,), 2.0, dtype=np.float64)
        )
        model = DynamicSmagorinskyPlan(
            GlobalDynamicLESAveraging(),
            ExactDenominatorRegularization(),
            NonnegativeBackscatterClip(),
        ).prepare(provenance)
        dynamic_plan = PeriodicDynamicLESPlan(
            model,
            grid_filter,
            PeriodicFourierTestFilterPlan(test_filter),
            PseudospectralMethodPlan(dealiasing=OversamplingDealiasingPlan(2.0)),
        )
        test = _space(3, 4)
    else:
        algebraic = PeriodicAlgebraicLESPlan(
            SmagorinskyLESPlan(0.16).prepare(parameters),
            grid_filter,
            PseudospectralMethodPlan(dealiasing=OversamplingDealiasingPlan(1.5)),
        )
    with pytest.raises(ValueError, match="LES"):
        compile_periodic_incompressible_flow(
            IncompressibleFlowProblem(3, 0.01),
            space,
            method,
            algebraic_les=algebraic,
            dynamic_les=dynamic_plan,
            dynamic_test_discretization=test,
            spectral_viscosity=_svv(),
        )
    dynamics = compile_periodic_incompressible_flow(
        IncompressibleFlowProblem(3, 0.01),
        space,
        method,
        algebraic_les=algebraic,
        dynamic_les=dynamic_plan,
        dynamic_test_discretization=test,
    )
    with pytest.raises(ValueError, match="LES"):
        CourantGuardedETDRKMethod(ETDRKMethod(4), maximum_courant_number=0.5).prepare(
            dynamics,
            coordinates=HermitianSpectralCoordinates(space, component_shape=(3,)),
        )


def test_beta_plane_svv_decay_and_per_volume_budget() -> None:
    space = _space(2)
    problem = BarotropicBetaPlane(space, beta=0.0, spectral_viscosity=_svv())
    x, y = jnp.meshgrid(*(axis.nodes for axis in space.axes), indexing="ij")
    state = problem.project_state(space.project(jnp.sin(2.0 * y) + 0.0 * x))
    budgets = problem.budgets(state)
    # lambda=-0.4; area-normalized sine square integral=1/2; |k|^2=4.
    np.testing.assert_allclose(
        budgets.spectral_viscosity_energy_rate, -0.4 * 0.5 / 4.0, atol=2e-15
    )
    np.testing.assert_allclose(
        budgets.spectral_viscosity_enstrophy_rate, -0.4 * 0.5, atol=2e-15
    )
    np.testing.assert_allclose(
        budgets.energy_rate, budgets.spectral_viscosity_energy_rate, atol=2e-15
    )
    np.testing.assert_allclose(
        budgets.enstrophy_rate, budgets.spectral_viscosity_enstrophy_rate, atol=2e-15
    )
    result = problem.prepare_etdrk().step(
        jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0), state, jnp.asarray(0.2), None
    )
    np.testing.assert_allclose(
        result.accepted_state, np.exp(-0.4 * 0.2) * state, atol=2e-14
    )
    assert bool(result.successful) and bool(budgets.successful)
    absent = BarotropicBetaPlane(space, beta=0.0).budgets(state)
    assert absent.spectral_viscosity_energy_rate == 0.0
    assert absent.spectral_viscosity_enstrophy_rate == 0.0


@pytest.mark.parametrize("closure", ["ce2", "gce2"])
def test_beta_plane_cumulants_include_svv_linear_term(closure: str) -> None:
    problem = BarotropicBetaPlane(_space(2, 6), beta=0.0, spectral_viscosity=_svv())
    partition = InteractionPartition.from_wavenumber_cutoff(
        problem.discretization,
        1,
        axes=(0, 1),
        admissibility_mask=problem.admissibility_mask,
    )
    system = BetaPlaneCumulantSystem(problem, partition)
    forcing = ForcingCovariance(jnp.zeros((system.layout.eddy_dimension,) * 2))
    prepared = system.prepare(forcing, closure=closure, time_step=1e-3)
    values = jnp.linspace(-0.2, 0.3, system.coordinates.coordinate_size)
    vorticity = system.coordinates.from_coordinates(values)
    expected = system.coordinates.to_coordinates(problem.linear_tendency(vorticity))
    np.testing.assert_allclose(system.linear @ values, expected, atol=1e-13)
    assert float(jnp.linalg.norm(expected)) > 0.0
    assert prepared.plan.dynamics is not None
