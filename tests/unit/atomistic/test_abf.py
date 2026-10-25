"""ABF: entropic estimators, constrained measure, bins, force-port sign and integration."""

from __future__ import annotations

import math
from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

from phydrax import AbstractArrayModel, config as pxconfig
from phydrax._fingerprint import array_tree_fingerprint
from phydrax._identity import IdentityWords
from phydrax.atomistic import (
    DistanceConstraintPlan,
    MolecularTopologyPlan,
    PreparedAtomisticSystem,
)
from phydrax.atomistic._force_port import AtomisticForcePortGeometry
from phydrax.atomistic.sampling import (
    abf_profile,
    ABFBaseMeasure,
    AtomisticABFForcePort,
    AtomisticABFPlan,
    AtomisticABFState,
    CollectiveVariableGeometry,
    CollectiveVariableKind,
    CollectiveVariablePlan,
    CollectiveVariableProgram,
    IntegratedProfileBiasForm,
    ModelCollectiveVariableProgram,
    PreparedABF,
)
from phydrax.typing import PRNGKey
from tests._support.atomistic_dynamics_fixtures import atomistic_system
from tests._support.atomistic_sampling_fixtures import cv_program


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


KT = 0.5


def _abf(
    *,
    bins: int = 15,
    min_samples: int = 1,
    full_samples: int = 1,
    max_force: float | None = None,
) -> tuple[PreparedAtomisticSystem, PreparedABF]:
    system = atomistic_system([0, 1, 2], [1.0] * 3)
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
    plan = AtomisticABFPlan(
        variables,
        minimum=0.5,
        maximum=2.0,
        bins=bins,
        temperature=KT,
        min_samples=min_samples,
        full_samples=full_samples,
        max_force=max_force,
    )
    return system, plan.prepare(system)


def _pair(r: float) -> jnp.ndarray:
    return jnp.asarray([[0.0, 0.0, 0.0], [r, 0.0, 0.0], [0.0, 3.0, 0.0]])


def _linear_forces(slope: float) -> jnp.ndarray:
    # U = a |x0 - x1| with x1 = r e_x: F0 = +a e_x, F1 = -a e_x.
    return jnp.zeros((3, 3)).at[0, 0].set(slope).at[1, 0].set(-slope)


def test_pair_distance_sample_is_the_radial_mean_force_with_entropic_term() -> None:
    _, abf = _abf()
    r, slope = 1.2, 1.0
    sample = abf.sample(CollectiveVariableGeometry(_pair(r)), _linear_forces(slope))
    # Independent reference A(r) = U(r) - 2 kT log r  =>  dA/dr = a - 2 kT / r.
    np.testing.assert_allclose(
        float(sample.mean_force[0]), slope - 2.0 * KT / r, rtol=1e-10
    )
    assert bool(sample.regular) and bool(sample.in_range)


def test_bins_are_half_open_and_out_of_range_is_not_recorded() -> None:
    # Baseline defect 3: an out-of-range sample was clipped into the edge bin.
    _, abf = _abf()
    width = 1.5 / 15
    for r, expected in (
        (0.5, 0),
        (0.5 + width - 1e-12, 0),
        (np.nextafter(0.5 + width, -np.inf), 0),
        (0.5 + width, 1),
    ):
        flat, inside = abf.bin_index(jnp.asarray([r]))
        assert bool(inside) and int(flat) == expected
    _, outside = abf.bin_index(jnp.asarray([2.0]))
    assert not bool(outside)
    state = abf.accumulate(
        abf.initialize(),
        abf.sample(CollectiveVariableGeometry(_pair(3.0)), _linear_forces(1.0)),
    )
    assert int(jnp.sum(state.counts)) == 0 and int(state.out_of_range) == 1


def test_force_port_opposes_the_mean_force_in_the_sampled_bin() -> None:
    # Baseline defect 3: the old bias had the wrong sign and acted in another cell
    # (nonzero at the sample point, zero at the centre of the sampled bin).
    # 16 bins as in the baseline keep r = 1.2 strictly inside bin 7 = [1.15625, 1.25).
    _, abf = _abf(bins=16, max_force=10.0)
    r, width = 1.2, 1.5 / 16
    sampled = math.floor((r - 0.5) / width)
    flat, inside = abf.bin_index(jnp.asarray([r]))
    assert sampled == 7 and bool(inside) and int(flat) == sampled
    state = abf.accumulate(
        abf.initialize(),
        abf.sample(CollectiveVariableGeometry(_pair(r)), _linear_forces(1.0)),
    )
    port = AtomisticABFForcePort(abf, name="pair")

    def forces_at(distance: float) -> np.ndarray:
        positions = _pair(distance)
        return np.asarray(
            port.forces(
                AtomisticForcePortGeometry(
                    positions=positions,
                    unwrapped_positions=positions,
                    cell_vectors=None,
                    image_counts=None,
                    species=jnp.zeros((3,), dtype=jnp.int32),
                ),
                state,
            ).forces
        )

    gradient = 1.0 - 2.0 * KT / r
    # V_bias = -A  =>  F_bias = +A'(r) grad r; grad_x1 r = +e_x, grad_x0 r = -e_x.
    # The sampled bin's estimate applies throughout that bin, sample point and centre.
    for distance in (r, 0.5 + (sampled + 0.5) * width):
        forces = forces_at(distance)
        np.testing.assert_allclose(forces[1], [gradient, 0.0, 0.0], rtol=1e-10)
        np.testing.assert_allclose(forces[0], [-gradient, 0.0, 0.0], rtol=1e-10)
        np.testing.assert_allclose(forces[2], 0.0)
    # Unvisited neighbouring bins carry no bias at all.
    for neighbour in (sampled - 1, sampled + 1):
        np.testing.assert_array_equal(forces_at(0.5 + (neighbour + 0.5) * width), 0.0)
    assert port.derivative_contract is None


class _OutputDtype(AbstractArrayModel):
    """Identity CV model whose output dtype differs from the coordinates."""

    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    def __init__(self, dtype: str) -> None:
        self.in_size = 1
        self.out_size = 1
        self.dtype = dtype

    def __call__(self, value: ArrayLike, /, *, key: PRNGKey | None = None) -> Array:
        del key
        return jnp.asarray(value).astype(self.dtype)


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    ("coordinate_dtype", "cv_dtype", "slope"),
    [("float64", "float32", 1.0e40), ("float32", "float64", 1.0)],
)
def test_force_port_applies_a_mixed_dtype_bias_in_the_coordinate_dtype(
    coordinate_dtype: str, cv_dtype: str, slope: float
) -> None:
    # A CV program of another dtype must not narrow the applied gradient: casting
    # the finite 1e40 gradient to the float32 CV dtype would overflow it.
    system = atomistic_system([0, 1, 2], [1.0] * 3)
    variables = ModelCollectiveVariableProgram(
        cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1]),
        _OutputDtype(cv_dtype),
        model_id=f"distance-as-{cv_dtype}",
        names=("distance",),
    )
    abf = AtomisticABFPlan(
        variables,
        minimum=0.5,
        maximum=2.0,
        bins=16,
        temperature=KT,
        min_samples=1,
        full_samples=1,
    ).prepare(system)
    positions = _pair(1.2).astype(coordinate_dtype)
    sample = abf.sample(
        CollectiveVariableGeometry(positions),
        _linear_forces(slope).astype(coordinate_dtype),
    )
    assert bool(sample.regular) and bool(sample.in_range)
    assert sample.values.dtype == np.dtype(cv_dtype)
    port = AtomisticABFForcePort(abf, name="mixed")
    state = abf.accumulate(port.initial_term_state(positions.dtype), sample)
    flat, _ = abf.bin_index(sample.values)
    gradient = float(abf.applied_gradient(state)[flat, 0])
    if slope > 1.0:
        assert abs(gradient) > float(np.finfo(np.float32).max)
    evaluation = port.forces(
        AtomisticForcePortGeometry(
            positions=positions,
            unwrapped_positions=positions,
            cell_vectors=None,
            image_counts=None,
            species=jnp.zeros((3,), dtype=jnp.int32),
        ),
        state,
    )
    forces = np.asarray(evaluation.forces)
    assert forces.dtype == np.dtype(coordinate_dtype)
    assert np.all(np.isfinite(forces))
    # grad_x1 r = +e_x and grad_x0 r = -e_x; the third atom carries no bias.
    expected = np.zeros((3, 3))
    expected[1, 0], expected[0, 0] = gradient, -gradient
    np.testing.assert_allclose(
        forces, expected, rtol=1e-12 if coordinate_dtype == "float64" else 1e-6
    )
    assert bool(evaluation.successful) and int(evaluation.status) == 0


def test_ramp_and_cap_follow_the_declared_thresholds() -> None:
    _, abf = _abf(min_samples=2, full_samples=4, max_force=0.1)
    state = abf.initialize()
    geometry = CollectiveVariableGeometry(_pair(1.2))
    applied = []
    for _ in range(4):
        state = abf.accumulate(state, abf.sample(geometry, _linear_forces(1.0)))
        flat, _ = abf.bin_index(jnp.asarray([1.2]))
        applied.append(float(abf.applied_gradient(state)[int(flat), 0]))
    np.testing.assert_allclose(applied, [0.0, 0.0, 0.05, 0.1])


def test_integrated_profile_recovers_the_radial_reference() -> None:
    _, abf = _abf()
    state = abf.initialize()
    centers = np.asarray(abf.plan.centers()[0])
    for r in centers:
        state = abf.accumulate(
            state,
            abf.sample(CollectiveVariableGeometry(_pair(float(r))), _linear_forces(1.0)),
        )
    profile = abf_profile(abf, state)
    assert profile.formula_admitted and profile.connected
    assert profile.integration_solve is not None
    assert bool(profile.integration_solve.successful)
    reference = centers - 2.0 * KT * np.log(centers)
    np.testing.assert_allclose(
        np.asarray(profile.free_energy) - np.asarray(profile.free_energy)[0],
        reference - reference[0],
        atol=5e-3,
    )
    form = IntegratedProfileBiasForm(profile)
    assert form.dimension == 1
    assert form.periods == (None,)
    evaluation = form.evaluate(
        jnp.asarray([1.0]),
        abf.plan.variables.metrics,
        jnp.zeros((0,)),
        None,
        jnp.int32(0),
    )
    assert bool(evaluation.successful) and math.isfinite(float(evaluation.energy))


def _coarea_rows(
    positions: np.ndarray,
    cv_pairs: tuple[tuple[int, int], ...],
    constrained: bool,
) -> np.ndarray:
    """Analytic distance/bond covectors, independent of the JAX CV implementation."""
    rows = []
    for left, right in cv_pairs:
        delta = positions[right] - positions[left]
        row = np.zeros_like(positions)
        row[right] = delta / np.linalg.norm(delta)
        row[left] = -row[right]
        rows.append(row.reshape(-1))
    if constrained:
        for left, right in ((0, 1), (1, 2)):
            row = np.zeros_like(positions)
            row[right] = 2.0 * (positions[right] - positions[left])
            row[left] = -row[right]
            rows.append(row.reshape(-1))
    return np.stack(rows)


@pytest.mark.parametrize("constrained", [False, True], ids=["joint-cvs", "fixman"])
def test_joint_coarea_actions_match_independent_finite_differences(
    constrained: bool,
) -> None:
    angle = 1.1
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0 + math.cos(angle), math.sin(angle), 0.0]],
        dtype=np.float64,
    )
    masses = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    pairs = ((0, 2),) if constrained else ((0, 1), (1, 2))
    topology = (
        MolecularTopologyPlan(
            constraints=((0, 1), (1, 2)), constraint_distances=(1.0, 1.0)
        )
        if constrained
        else None
    )
    system = atomistic_system([0, 1, 2], masses, topology=topology)
    variables = CollectiveVariableProgram(
        tuple(
            CollectiveVariablePlan(CollectiveVariableKind.DISTANCE, pair).prepare(system)
            for pair in pairs
        )
    )
    constraints = DistanceConstraintPlan().prepare(system) if constrained else None
    abf = AtomisticABFPlan(
        variables,
        minimum=np.full(len(pairs), 0.5, dtype=np.float64),
        maximum=np.full(len(pairs), 2.0, dtype=np.float64),
        bins=4,
        temperature=KT,
        measure=ABFBaseMeasure.CONSTRAINED_DYNAMICS
        if constrained
        else ABFBaseMeasure.CONFIGURATIONAL,
    ).prepare(system, constraints=constraints)

    def fields(point: np.ndarray) -> np.ndarray:
        rows = _coarea_rows(point.reshape((-1, 3)), pairs, constrained)
        return np.linalg.solve(rows @ rows.T, rows)[: len(pairs)].T

    flat = positions.reshape(-1)
    basis = fields(flat)
    step = 2.0e-5
    divergence = np.zeros(len(pairs), dtype=np.float64)
    for coordinate in range(flat.size):
        direction = np.zeros_like(flat)
        direction[coordinate] = step
        divergence += (
            fields(flat + direction)[coordinate] - fields(flat - direction)[coordinate]
        ) / (2.0 * step)
    forces = -0.2 * positions
    expected = -(forces.reshape(-1) @ basis) - KT * divergence
    if constrained:
        inverse_masses = np.repeat(1.0 / masses, 3)

        def log_metric(point: np.ndarray) -> float:
            rows = _coarea_rows(point.reshape((-1, 3)), pairs, True)[len(pairs) :]
            sign, value = np.linalg.slogdet((rows * inverse_masses) @ rows.T)
            assert sign == 1.0
            return float(value)

        correction = (
            log_metric(flat + step * basis[:, 0]) - log_metric(flat - step * basis[:, 0])
        ) / (2.0 * step)
        assert abs(correction) > 1.0e-3
        # Canonical constrained density delta(sigma) sqrt(det G) dq: w = det(G)^{1/2}.
        expected -= 0.5 * KT * correction
    sample = abf.sample(
        CollectiveVariableGeometry(jnp.asarray(positions)), jnp.asarray(forces)
    )
    assert bool(sample.regular) and bool(sample.dual_solve.successful)
    np.testing.assert_allclose(sample.mean_force, expected, rtol=2.0e-7, atol=1.0e-9)
    if constrained:
        assert sample.fixman_solve is not None
        assert bool(sample.fixman_solve.successful)


def test_dependent_collective_variables_refuse_without_recording_a_zero_force() -> None:
    system = atomistic_system([0, 1, 2], [1.0] * 3)
    variables = CollectiveVariableProgram(
        tuple(
            CollectiveVariablePlan(CollectiveVariableKind.DISTANCE, pair).prepare(system)
            for pair in ((0, 1), (1, 0))
        )
    )
    abf = AtomisticABFPlan(
        variables,
        minimum=np.asarray([0.5, 0.5], dtype=np.float64),
        maximum=np.asarray([2.0, 2.0], dtype=np.float64),
        bins=4,
        temperature=KT,
    ).prepare(system)
    sample = abf.sample(CollectiveVariableGeometry(_pair(1.2)), _linear_forces(1.0))
    assert not bool(sample.regular)
    assert not bool(sample.dual_solve.successful)
    assert np.all(np.isnan(sample.mean_force))
    state = abf.accumulate(abf.initialize(), sample)
    assert int(jnp.sum(state.counts)) == 0
    assert int(state.irregular) == 1


def _rigid_triangle(theta: float) -> np.ndarray:
    half = 0.5 * theta
    return np.asarray(
        [
            [-math.sin(half), math.cos(half), 0.0],
            [0.0, 0.0, 0.0],
            [math.sin(half), math.cos(half), 0.0],
        ],
        dtype=np.float64,
    )


def _rigid_abf(min_samples: int = 200) -> PreparedABF:
    topology = MolecularTopologyPlan(
        constraints=((0, 1), (1, 2)), constraint_distances=(1.0, 1.0)
    )
    rigid = atomistic_system([0, 1, 2], [1.0] * 3, topology=topology)
    end_to_end = cv_program(rigid, CollectiveVariableKind.DISTANCE, [0, 2])
    return AtomisticABFPlan(
        end_to_end,
        minimum=0.5,
        maximum=1.9,
        bins=8,
        temperature=KT,
        min_samples=min_samples,
        full_samples=min_samples,
        measure=ABFBaseMeasure.CONSTRAINED_DYNAMICS,
    ).prepare(rigid, constraints=DistanceConstraintPlan().prepare(rigid))


@pytest.mark.parametrize("theta", [0.7, 1.1, 1.9, 2.5])
def test_constrained_estimator_is_the_canonical_rigid_end_to_end_marginal(
    theta: float,
) -> None:
    # Independent control: three equal masses, two unit rigid bonds, no energy.
    # RATTLE with a projected thermostat samples delta(sigma) sqrt(det G) dq;
    # integrating the rigid-body orientation leaves sin(theta) dtheta times
    # sqrt(det G) = 4 sqrt(4 - cos^2 theta) / m.  With r^2 = 2 - 2 cos theta,
    # r dr = sin theta dtheta, so p(r) ~ r sqrt(4 - cos^2 theta) and
    # A'(r) = -kT/r - kT cos(theta) r / (4 - cos^2 theta).
    abf = _rigid_abf()
    point = _rigid_triangle(theta)
    sample = abf.sample(CollectiveVariableGeometry(jnp.asarray(point)), jnp.zeros((3, 3)))
    radius = float(np.linalg.norm(point[2] - point[0]))
    cosine = math.cos(theta)
    np.testing.assert_allclose(radius**2, 2.0 - 2.0 * cosine, rtol=1e-12)

    def log_marginal(r: float) -> float:
        c = 1.0 - 0.5 * r * r
        return math.log(r) + 0.5 * math.log(4.0 - c * c)

    step = 1.0e-6
    finite_difference = (
        -KT * (log_marginal(radius + step) - log_marginal(radius - step)) / (2.0 * step)
    )
    closed_form = -KT / radius - KT * cosine * radius / (4.0 - cosine * cosine)
    np.testing.assert_allclose(finite_difference, closed_form, rtol=1e-8)
    assert bool(sample.regular) and bool(sample.in_range)
    assert sample.fixman_solve is not None and bool(sample.fixman_solve.successful)
    np.testing.assert_allclose(float(sample.mean_force[0]), closed_form, rtol=1e-9)


def test_constrained_plan_binds_its_canonical_density() -> None:
    abf = _rigid_abf(min_samples=1)
    density = "delta(sigma) sqrt(det(J_sigma M^-1 J_sigma^T)) dq"
    assert ABFBaseMeasure.CONSTRAINED_DYNAMICS.density == density
    state = abf.accumulate(
        abf.initialize(),
        abf.sample(
            CollectiveVariableGeometry(jnp.asarray(_rigid_triangle(1.1))),
            jnp.zeros((3, 3)),
        ),
    )
    assert int(jnp.sum(state.counts)) == 1
    profile = abf_profile(abf, state)
    assert f"base-density:{density}" in profile.assumptions
    # A history accumulated under a constrained declaration that did not bind
    # the density (the former, wrong-sign estimator) is another owner: it
    # neither resumes accumulation nor integrates under the corrected plan.
    plan = abf.plan
    unbound = IdentityWords.of(
        {
            "kind": "abf-plan",
            "variables": plan.variables.program_id,
            "bounds": array_tree_fingerprint(
                {"minimum": np.asarray(plan.minimum), "maximum": np.asarray(plan.maximum)}
            ),
            "bins": list(plan.bins),
            "temperature": float(np.asarray(plan.temperature)).hex(),
            "min_samples": int(np.asarray(plan.min_samples)),
            "full_samples": int(np.asarray(plan.full_samples)),
            "max_force": None,
            "measure": plan.measure.value,
        }
    )
    assert unbound.fingerprint() != plan.abf_id
    stale = eqx.tree_at(lambda value: value.owner_identity, state, unbound)
    resumed = abf.accumulate(
        stale,
        abf.sample(
            CollectiveVariableGeometry(jnp.asarray(_rigid_triangle(1.1))),
            jnp.zeros((3, 3)),
        ),
    )
    assert int(jnp.sum(resumed.counts)) == 1 and not bool(resumed.successful)
    with pytest.raises(ValueError, match="another ABF plan"):
        abf_profile(abf, stale)


def _torsion_system() -> PreparedAtomisticSystem:
    return atomistic_system([0, 1, 2, 3], [1.0] * 4)


def _filled(abf: PreparedABF, means: np.ndarray, counts: np.ndarray) -> AtomisticABFState:
    """Frozen bin statistics with the given per-bin means (no sampling)."""
    return eqx.tree_at(
        lambda state: (state.counts, state.force_sums),
        abf.initialize(),
        (
            jnp.asarray(counts, dtype=jnp.int32),
            jnp.asarray(means * counts[:, None], dtype=jnp.float64),
        ),
    )


@pytest.mark.parametrize("upper", [False, True], ids=["lower", "upper"])
def test_nonperiodic_profile_constant_extension_has_one_sided_forces(upper: bool) -> None:
    _, abf = _abf(bins=8)
    profile = abf_profile(
        abf,
        _filled(
            abf,
            np.ones((8, 1), dtype=np.float64),
            np.ones(8, dtype=np.int32),
        ),
    )
    assert profile.formula_admitted
    form = IntegratedProfileBiasForm(profile)

    def energy(point: jax.Array) -> jax.Array:
        evaluation = form.evaluate(
            jnp.reshape(point, (1,)),
            abf.plan.variables.metrics,
            jnp.zeros((0,), dtype=jnp.float64),
            None,
            jnp.int32(0),
        )
        return evaluation.energy

    nodes = profile.nodes[0]
    endpoint = nodes[-1] if upper else nodes[0]
    step = (nodes[1] - nodes[0]) * (-1e-3 if upper else 1e-3)
    inside, outside = endpoint + step, endpoint - step
    gradient = jax.grad(energy)
    # A'=1 gives V_bias'=-1 inside and a constant continuation outside.
    # Their unequal one-sided limits make the endpoint derivative nonunique.
    np.testing.assert_allclose(gradient(inside), -1.0, atol=1e-10)
    np.testing.assert_array_equal(gradient(outside), 0.0)
    np.testing.assert_allclose(energy(inside) - energy(endpoint), -step, atol=1e-12)
    np.testing.assert_array_equal(energy(outside), energy(endpoint))


def _torsion_abf(*, minimum: float, maximum: float, bins: int) -> PreparedABF:
    system = _torsion_system()
    variables = cv_program(system, CollectiveVariableKind.TORSION, [0, 1, 2, 3])
    return AtomisticABFPlan(
        variables,
        minimum=minimum,
        maximum=maximum,
        bins=bins,
        temperature=KT,
        min_samples=1,
        full_samples=1,
    ).prepare(system)


def test_periodic_integrated_profile_is_continuous_through_the_seam() -> None:
    abf = _torsion_abf(minimum=-math.pi, maximum=math.pi, bins=8)
    centers = np.asarray(abf.plan.centers()[0])
    width = 2.0 * math.pi / 8
    gradient = 0.4 + np.cos(centers)
    profile = abf_profile(
        abf, _filled(abf, gradient[:, None], np.ones(8, dtype=np.int32))
    )
    assert profile.periodic_axes == (True,)
    assert profile.formula_admitted and profile.connected
    free_energy = np.asarray(profile.free_energy)
    # Unequal end-node values: a clamped (nonperiodic) seam would jump by this.
    assert abs(free_energy[0] - free_energy[-1]) > 0.1
    form = IntegratedProfileBiasForm(profile)
    metrics = abf.plan.variables.metrics

    def value(phi: jax.Array) -> jax.Array:
        return form.evaluate(
            jnp.reshape(phi, (1,)), metrics, jnp.zeros((0,)), None, jnp.int32(0)
        ).energy

    def energy(phi: float) -> jax.Array:
        evaluation = form.evaluate(
            jnp.asarray([phi]), metrics, jnp.zeros((0,)), None, jnp.int32(0)
        )
        assert bool(evaluation.successful)
        return evaluation.energy

    force = jax.grad(value)
    for epsilon in (1.0e-9, 1.0e-6):
        np.testing.assert_allclose(
            float(energy(math.pi - epsilon)), float(energy(-math.pi + epsilon)), atol=1e-5
        )
        np.testing.assert_allclose(
            float(force(math.pi - epsilon)), float(force(-math.pi + epsilon)), atol=1e-5
        )
    for phi in (-2.9, 0.3, 3.0):
        np.testing.assert_allclose(
            float(energy(phi)), float(energy(phi + 2.0 * math.pi)), atol=1e-12
        )
    # Seam interval [c_7, c_0 + 2 pi] is the Hermite segment of the node data
    # V = -A, V' = -A'; its midpoint is the seam phi = pi.
    y0, y1 = -free_energy[-1], -free_energy[0]
    s0, s1 = -gradient[-1], -gradient[0]
    np.testing.assert_allclose(
        float(energy(math.pi)),
        0.5 * (y0 + y1) + width * (s0 - s1) / 8.0,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        float(force(math.pi)),
        (1.5 * (y1 - y0) - 0.25 * width * (s0 + s1)) / width,
        rtol=1e-10,
        atol=1e-12,
    )


def test_periodic_two_cv_profile_interpolates_through_the_seam() -> None:
    system = _torsion_system()
    variables = CollectiveVariableProgram(
        tuple(
            CollectiveVariablePlan(CollectiveVariableKind.TORSION, quartet).prepare(
                system
            )
            for quartet in ((0, 1, 2, 3), (1, 2, 3, 0))
        )
    )
    bins = 6
    abf = AtomisticABFPlan(
        variables,
        minimum=np.full(2, -math.pi),
        maximum=np.full(2, math.pi),
        bins=bins,
        temperature=KT,
        min_samples=1,
        full_samples=1,
    ).prepare(system)
    first, second = (np.asarray(axis) for axis in abf.plan.centers())
    grid_first, grid_second = np.meshgrid(first, second, indexing="ij")
    means = np.stack((0.4 + np.cos(grid_first), 0.2 * np.ones_like(grid_second)), -1)
    profile = abf_profile(
        abf, _filled(abf, means.reshape((-1, 2)), np.ones(bins * bins, dtype=np.int32))
    )
    assert profile.periodic_axes == (True, True) and profile.formula_admitted
    form = IntegratedProfileBiasForm(profile)
    assert form.periods == (2.0 * math.pi, 2.0 * math.pi)
    metrics = variables.metrics

    def energy(point: list[float]) -> jax.Array:
        evaluation = form.evaluate(
            jnp.asarray(point), metrics, jnp.zeros((0,)), None, jnp.int32(0)
        )
        assert bool(evaluation.successful)
        return evaluation.energy

    epsilon = 1.0e-9
    for column in range(bins):
        y = float(second[column])
        np.testing.assert_allclose(
            float(energy([math.pi - epsilon, y])),
            float(energy([-math.pi + epsilon, y])),
            atol=1e-7,
        )
        # Multilinear midpoint of the wrapping interval between c_last and c_0.
        values = -np.asarray(profile.free_energy)
        np.testing.assert_allclose(
            float(energy([math.pi, y])),
            0.5 * (values[-1, column] + values[0, column]),
            rtol=1e-12,
            atol=1e-12,
        )


def test_periodic_profile_refuses_unqualified_support_or_range() -> None:
    abf = _torsion_abf(minimum=-math.pi, maximum=math.pi, bins=8)
    centers = np.asarray(abf.plan.centers()[0])
    counts = np.ones(8, dtype=np.int32)
    counts[3] = 0
    profile = abf_profile(abf, _filled(abf, np.cos(centers)[:, None], counts))
    # The supported run wraps the seam (bins 4..7, 0..2) and is connected.
    assert profile.connected and profile.periodic_axes == (True,)
    with pytest.raises(ValueError, match="every bin of the period"):
        IntegratedProfileBiasForm(profile)
    partial = _torsion_abf(minimum=-0.5 * math.pi, maximum=0.5 * math.pi, bins=8)
    centers = np.asarray(partial.plan.centers()[0])
    profile = abf_profile(
        partial, _filled(partial, np.cos(centers)[:, None], np.ones(8, dtype=np.int32))
    )
    assert profile.periodic_axes == (False,)
    with pytest.raises(ValueError, match="close its period"):
        IntegratedProfileBiasForm(profile)


def test_profile_integration_is_a_native_matrix_free_solve() -> None:
    system = _torsion_system()
    variables = CollectiveVariableProgram(
        tuple(
            CollectiveVariablePlan(CollectiveVariableKind.TORSION, quartet).prepare(
                system
            )
            for quartet in ((0, 1, 2, 3), (1, 2, 3, 0))
        )
    )
    bins = 40
    abf = AtomisticABFPlan(
        variables,
        minimum=np.full(2, -math.pi),
        maximum=np.full(2, math.pi),
        bins=bins,
        temperature=KT,
        min_samples=1,
        full_samples=1,
    ).prepare(system)
    first, second = (np.asarray(axis) for axis in abf.plan.centers())
    grid_first, grid_second = np.meshgrid(first, second, indexing="ij")
    # Exactly integrable periodic field of A = sin(x) + cos(y).
    means = np.stack((np.cos(grid_first), -np.sin(grid_second)), -1)
    profile = abf_profile(
        abf, _filled(abf, means.reshape((-1, 2)), np.ones(bins * bins, dtype=np.int32))
    )
    evidence = profile.integration_solve
    assert evidence is not None and bool(evidence.successful)
    diagnostics = evidence.diagnostics
    assert bool(np.all(np.asarray(diagnostics.converged)))
    # Iterative operator actions certify the true residual at the declared tolerance.
    iterations = np.asarray(diagnostics.iterations)
    assert bool(np.all(iterations > 0))
    assert bool(np.all(np.asarray(diagnostics.matvec_count) >= iterations))
    assert float(np.max(np.asarray(diagnostics.relative_residual))) <= (
        np.finfo(np.float64).eps ** 0.625
    )
    assert profile.formula_admitted and profile.periodic_axes == (True, True)
    reference = np.sin(grid_first) + np.cos(grid_second)
    free_energy = np.asarray(profile.free_energy)
    width = 2.0 * math.pi / bins
    # Trapezoid edge targets carry an O(w^2 / 12) truncation per edge.
    np.testing.assert_allclose(
        free_energy - free_energy[0, 0],
        reference - reference[0, 0],
        atol=width**2,
    )
    assert float(profile.integrability_residual) < width**2
