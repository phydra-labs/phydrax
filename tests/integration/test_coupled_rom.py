#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""A Galerkin reduced-order model swapped into one region of a coupled plate.

The P1 triangles of the FE-VEM plate (``tests._support.coupled_plate``) are
replaced by ``ReducedComponent("triangles", FullResidualGalerkin(...))`` built
offline from full-order coupled snapshots at seeded training parameters: the
truncated ROMs on the leading modes of an uncentered physical POD, the spanning
ROM on the host SVD basis of the snapshots' manifold. The plan is otherwise
identical (same interface binding, law, parameter and observation bindings).
The truth of the ROM is the full-order coupled solution at held-out
parameters; the host analytic temperature (``exact_temperature``) is the
discretization reference.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.solver import coupling as cpl
from tests._support import coupled_plate as cp


M = phx.measurement

LEVEL = 0
FE_H = 1.0 / (4 * 2**LEVEL)
REFERENCE = {"conductivity-left": 1.3, "conductivity-right": 0.7, "heat-flux": 0.5}
# Training parameters (kappa_left, kappa_right, g) in a bounded box, and
# held-out parameters inside the box that are not training samples.
TRAINING_SEED = 20260928
TRAINING_COUNT = 12
LOWER = np.asarray([0.8, 0.5, -0.5])
UPPER = np.asarray([1.6, 1.2, 1.0])
HELD_OUT = np.asarray([[1.1, 0.9, 0.2], [1.45, 0.6, 0.8]])
SOURCES = ("plate-training-snapshots",)
# POD ranks below the dimension of the triangles' parametric solution manifold.
# The triangles' block solves (1 / kappa_left) K0 u = f + B lambda, so the manifold
# has at most 1 + (multiplier size) = 5 directions. Its fifth singular value is
# 1.8e-8 of the first, below the resolution floor sqrt((N + sqrt(m)) eps) of the
# method of snapshots (eigenvalues of the Gram matrix), so the POD stops short of it.
# The spanning basis is therefore the leading MANIFOLD_DIMENSION left
# singular vectors of a host SVD of the same snapshots (accurate to eps
# relative), whose trailing singular values certify the dimension.
TRUNCATED_RANKS = (1, 2, 3, 4)
MANIFOLD_DIMENSION = 5
# Observed at the spanning rank: field, multiplier, observation, and flux-balance
# discrepancies of 1e-16 to 2e-15 relative.
SPANNING_RTOL = 1e-10
WEAK_RTOL = 1e-12

CONTRACT = phx.SpatialCoordinateContract(phx.units.METER)
TEMPERATURE = M.QuantitySpec(
    "test", "temperature", "temperature", phx.units.KELVIN, "temperature"
)
SENSORS = np.asarray([[0.3, 0.4, 0.0], [0.7, 0.6, 0.0], [0.5, 0.9, 0.0]])
POINT = M.SamplingSemantics(M.SpatialSamplingKind.POINT)
AVERAGE = M.SamplingSemantics(M.SpatialSamplingKind.SURFACE_AVERAGE)
RULE = phx.discretization.FacetTraceRule(points=2)
OBSERVATIONS = ("left-sensors", "interface-mean")


def _observations(plate: cp.CoupledPlate) -> tuple[cpl.AbstractObservationBinding, ...]:
    """Point values and the interface-side mean temperature of the triangles."""
    return (
        cpl.FieldPointObservation(
            "left-sensors",
            "triangles",
            "u",
            quantity=TEMPERATURE,
            support=M.PointSampleSupport(SENSORS, ("a", "b", "c"), CONTRACT),
            sampling=POINT,
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldBoundaryObservation(
            "interface-mean",
            "triangles",
            "u",
            plate.law.sides[0].domain,
            statistic="average",
            rule=RULE,
            quantity=TEMPERATURE,
            support=M.IndexSampleSupport((1,), ("interface",)),
            sampling=AVERAGE,
            field_unit=phx.units.KELVIN,
        ),
    )


def _prepare(
    plate: cp.CoupledPlate, triangles: cpl.AbstractTraceComponent
) -> cpl.PreparedCoupledProblem:
    """The plate plan with ``triangles`` as its left region."""
    plan = cpl.CoupledProblemPlan(
        "plate",
        components=(triangles, plate.polygons),
        bindings=(plate.binding,),
        laws=(plate.law,),
        parameters=cp.parameter_bindings(),
        observations=_observations(plate),
    )
    return cpl.prepare_coupled_problem(
        plan,
        interface_owners=(plate.cover,),
        parameters={name: jnp.asarray(value) for name, value in REFERENCE.items()},
    )


def _parameters(theta: Array) -> dict[str, Array]:
    return {
        "conductivity-left": theta[0],
        "conductivity-right": theta[1],
        "heat-flux": theta[2],
    }


def _solve(prepared: cpl.PreparedCoupledProblem, theta: Array) -> cpl.CoupledSolution:
    return cpl.solve_coupled_problem(
        prepared, parameters=_parameters(theta), policy=cp.dense_policy()
    )


@dataclass(frozen=True)
class _Offline:
    """Full-order problem, held-out solutions, the POD, and the host manifold basis."""

    plate: cp.CoupledPlate
    full: cpl.PreparedCoupledProblem
    provider: cpl.ComponentResidualProvider
    pod: phx.ml.decomposition.PhysicalPODResult
    host_singular_values: np.ndarray
    manifold: np.ndarray
    held_out: tuple[cpl.CoupledSolution, ...]

    def reduced(self, rank: int) -> cpl.ReducedComponent:
        """The ROM on the leading ``rank`` POD modes."""
        return _reduced_component(self.provider, self.pod.subspace.basis[:, :rank])

    def spanning_reduced(self) -> cpl.ReducedComponent:
        """The ROM on the host basis of the snapshots' manifold."""
        return _reduced_component(self.provider, jnp.asarray(self.manifold))


def _reduced_component(
    provider: cpl.ComponentResidualProvider, modes: Array
) -> cpl.ReducedComponent:
    reduction = phx.rom.trial_test_reduction_from_bases(_artifact(provider, modes))
    return cpl.ReducedComponent(
        "triangles", phx.rom.FullResidualGalerkin(reduction, provider)
    )


def _artifact(
    provider: cpl.ComponentResidualProvider, modes: Array
) -> phx.rom.ReducedBasisArtifact:
    return phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(provider.state_space, modes, orthonormal=True),
        role="state",
        state_contract_id=provider.residual_id,
        support_id=provider.support_id,
        measure_id="euclidean-coordinates",
        geometry_id=provider.geometry_id,
        source_artifact_ids=SOURCES,
    )


@pytest.fixture(scope="module")
def plate() -> cp.CoupledPlate:
    return cp.build_plate(LEVEL)


@pytest.fixture(scope="module")
def offline(plate: cp.CoupledPlate) -> _Offline:
    """Full-order training solves, the POD of their triangle blocks, and held-out truths."""
    full = _prepare(plate, plate.triangles)
    solve = jax.jit(lambda theta: _solve(full, theta))
    rng = np.random.default_rng(TRAINING_SEED)
    training = LOWER + (UPPER - LOWER) * rng.uniform(size=(TRAINING_COUNT, 3))
    index = [component.name for component in full.components].index("triangles")
    snapshots = []
    for theta in training:
        solution = solve(jnp.asarray(theta))
        assert bool(solution.accepted), theta
        snapshots.append(solution.state[index][0])
    provider = cpl.ComponentResidualProvider(plate.triangles, field="u")
    pod = phx.ml.decomposition.PhysicalPODPlan(TRAINING_COUNT, centered=False).fit(
        provider.state_space, jnp.stack(snapshots), source_artifact_ids=SOURCES
    )
    # Uncentered snapshot matrix (columns are snapshots) scaled like the POD's
    # uniform sample weights, so its singular values are comparable.
    left, singular, _ = np.linalg.svd(
        np.stack([np.asarray(item) for item in snapshots], axis=1)
        / np.sqrt(TRAINING_COUNT),
        full_matrices=False,
    )
    held_out = tuple(solve(jnp.asarray(theta)) for theta in HELD_OUT)
    return _Offline(
        plate,
        full,
        provider,
        pod,
        singular,
        left[:, :MANIFOLD_DIMENSION],
        held_out,
    )


@dataclass(frozen=True)
class _Rank:
    """One ROM-coupled problem and its held-out solutions."""

    rank: int
    prepared: cpl.PreparedCoupledProblem
    solutions: tuple[cpl.CoupledSolution, ...]


def _reduced_problem(
    offline: _Offline, rank: int, component: cpl.ReducedComponent
) -> _Rank:
    """A separate preparation of the plate with one rank-``rank`` ROM region."""
    prepared = _prepare(offline.plate, component)
    solve = jax.jit(lambda theta: _solve(prepared, theta))
    return _Rank(rank, prepared, tuple(solve(jnp.asarray(theta)) for theta in HELD_OUT))


@pytest.fixture(scope="module")
def spanning(offline: _Offline) -> _Rank:
    return _reduced_problem(offline, MANIFOLD_DIMENSION, offline.spanning_reduced())


@pytest.fixture(scope="module")
def truncated(offline: _Offline) -> tuple[_Rank, ...]:
    return tuple(
        _reduced_problem(offline, rank, offline.reduced(rank)) for rank in TRUNCATED_RANKS
    )


def _relative(value: Array, reference: Array) -> float:
    difference = np.linalg.norm(np.asarray(value) - np.asarray(reference))
    return float(difference / np.linalg.norm(np.asarray(reference)))


def _errors(solution: cpl.CoupledSolution, truth: cpl.CoupledSolution) -> np.ndarray:
    """Relative errors of both fields, the multiplier, and the observations."""
    return np.asarray(
        [
            _relative(solution.field("triangles", "u"), truth.field("triangles", "u")),
            _relative(solution.field("polygons", "u"), truth.field("polygons", "u")),
            _relative(
                solution.law_state("transmission")[0], truth.law_state("transmission")[0]
            ),
            *(
                _relative(
                    solution.observation(name).values, truth.observation(name).values
                )
                for name in OBSERVATIONS
            ),
        ]
    )


def _defect(report: cpl.InterfaceDefectReport, name: str) -> float:
    """One interface defect relative to its reference magnitude."""
    index = report.names.index(name)
    return float(report.values[index] / report.scales[index])


def _identity(field: M.PreparedQuantityField) -> tuple[str, ...]:
    return (
        field.quantity_id,
        field.compatibility_id,
        field.layout_id,
        field.support_id,
        field.sampling_id,
        field.unit_id,
        field.field_id,
    )


# --- Retained identities and held-out accuracy ------------------------------------------


def test_reduced_region_keeps_the_plan_bindings_and_observation_identities(
    offline: _Offline, spanning: _Rank
) -> None:
    full = offline.full
    names = [component.name for component in full.components]
    assert [component.name for component in spanning.prepared.components] == names
    assert spanning.prepared.bindings == full.bindings
    reduced = spanning.prepared.components[names.index("triangles")]
    assert isinstance(reduced, cpl.ReducedComponent)
    # The reduced region publishes the full owner's field space and pointwise
    # reconstruction, so its observations keep their identities and exact FE
    # evaluation of the reconstructed field.
    assert reduced.field_space_id("u") == offline.plate.triangles.field_space_id("u")
    for name in OBSERVATIONS:
        assert spanning.prepared.observation(name).approximation == "exact"
        assert full.observation(name).approximation == "exact"
        for solution, truth in zip(spanning.solutions, offline.held_out, strict=True):
            assert _identity(solution.observation(name)) == _identity(
                truth.observation(name)
            )


def test_rank_sweep_converges_to_the_full_order_plate_with_its_interface_residual(
    offline: _Offline, truncated: tuple[_Rank, ...], spanning: _Rank
) -> None:
    # The training snapshots span the manifold: the host SVD (accurate to eps
    # relative) has MANIFOLD_DIMENSION singular values above roundoff and none
    # past it. The POD's method of snapshots agrees on them within its
    # sqrt(eps) relative floor.
    host = offline.host_singular_values
    assert host[MANIFOLD_DIMENSION - 1] > 1e-9 * host[0], host
    assert np.all(host[MANIFOLD_DIMENSION:] < 1e-13 * host[0]), host
    singular = np.asarray(offline.pod.singular_values)[:MANIFOLD_DIMENSION]
    np.testing.assert_allclose(
        singular,
        host[:MANIFOLD_DIMENSION],
        rtol=0.0,
        atol=4.0 * np.sqrt(np.finfo(np.float64).eps) * host[0],
    )
    assert truncated[-1].rank < MANIFOLD_DIMENSION == spanning.rank
    sweep = (*truncated, spanning)
    errors = []
    balances = []
    for entry in sweep:
        worst = np.zeros((3 + len(OBSERVATIONS),))
        balance = 0.0
        for index, (solution, truth) in enumerate(
            zip(entry.solutions, offline.held_out, strict=True)
        ):
            worst = np.maximum(worst, _errors(solution, truth))
            report = solution.interface("transmission")
            # The mortar rows are solved exactly at every rank: weak continuity
            # holds to roundoff whatever the reduced field is.
            weak = _defect(report, "weak-continuity")
            assert weak <= WEAK_RTOL, (entry.rank, index, weak)
            balance = max(balance, _defect(report, "flux-balance"))
        errors.append(worst)
        balances.append(balance)
    # Every rank is a separate prepared problem.
    assert len({entry.prepared.problem_id for entry in sweep}) == len(sweep)
    # The held-out field errors of both regions and the flux balance (the full
    # owner's reaction of the reconstructed field against the multiplier, i.e.
    # the interface residual of the ROM) decrease with the rank. The multiplier
    # is a dual variable of the saddle-point coupling with no such monotonicity
    # (observed: 1.2e-3 at rank 1, 1.6e-3 at rank 2); it converges with the rest.
    for index in range(len(sweep) - 1):
        row = (sweep[index].rank, sweep[index + 1].rank)
        assert np.all(errors[index + 1][:2] < errors[index][:2]), (row, errors)
        assert balances[index + 1] < balances[index], (row, balances)
    for entry in truncated:
        assert not any(bool(solution.accepted) for solution in entry.solutions), (
            entry.rank
        )
    # At the rank that spans the training snapshots' manifold the ROM-coupled
    # plate is the full-order plate.
    assert np.max(errors[-1]) < SPANNING_RTOL, errors[-1]
    assert balances[-1] < SPANNING_RTOL, balances
    assert all(bool(solution.accepted) for solution in spanning.solutions)


def test_reduced_plate_has_the_full_order_discretization_error(
    offline: _Offline, spanning: _Rank
) -> None:
    nodes = offline.plate.triangle_nodes
    for theta, solution, truth in zip(
        HELD_OUT, spanning.solutions, offline.held_out, strict=True
    ):
        exact = cp.exact_temperature(nodes, *theta)
        reduced = np.max(np.abs(np.asarray(solution.field("triangles", "u")) - exact))
        full = np.max(np.abs(np.asarray(truth.field("triangles", "u")) - exact))
        # The exact u is quadratic in x: the P1 (and mortar) nodal error is
        # O(h^2 |u''|) with |u''| = s / kappa_left; twice the interpolation bound
        # h^2 |u''| / 8 is admitted. The spanning ROM adds only its reduction
        # error, far below that discretization error.
        bound = FE_H**2 * cp.SOURCE / theta[0] / 4.0
        assert full <= bound, (theta, full, bound)
        np.testing.assert_allclose(reduced, full, rtol=1e-6)


# --- Parameters flow explicitly through the reduced region ------------------------------


def _observed(prepared: cpl.PreparedCoupledProblem, theta: Array) -> Array:
    solution = _solve(prepared, theta)
    return jnp.concatenate([solution.observation(name).values for name in OBSERVATIONS])


def test_parameter_derivatives_through_the_reduced_solve_match_full_order(
    offline: _Offline, spanning: _Rank
) -> None:
    theta = jnp.asarray(HELD_OUT[0])

    def reduced(value: Array) -> Array:
        return _observed(spanning.prepared, value)

    def full(value: Array) -> Array:
        return _observed(offline.full, value)

    reduced_jacobian = np.asarray(jax.jit(jax.jacrev(reduced))(theta))
    full_jacobian = np.asarray(jax.jit(jax.jacrev(full))(theta))
    # The triangles' temperatures scale like 1 / kappa_left: every observation
    # has a nonzero implicit derivative in the conductivity the ROM region reads.
    assert np.all(np.abs(full_jacobian[:, 0]) > 1e-2), full_jacobian
    np.testing.assert_allclose(
        reduced_jacobian,
        full_jacobian,
        rtol=0.0,
        atol=SPANNING_RTOL * np.max(np.abs(full_jacobian)),
    )


# --- A basis change is a new preparation, never an online derivative -------------------


def test_basis_is_fixed_structure_and_a_change_is_a_new_problem(
    offline: _Offline, spanning: _Rank
) -> None:
    reduced = [
        *(offline.reduced(rank) for rank in TRUNCATED_RANKS),
        offline.spanning_reduced(),
    ]
    owners = {component.owner_id for component in reduced}
    assert len(owners) == len(reduced)
    assert offline.plate.triangles.owner_id not in owners
    assert spanning.prepared.problem_id != offline.full.problem_id
    # Every array leaf of the reduced region (basis, provider, owner) is FIXED:
    # a training step can never move it.
    resolution = phx.resolve_array_roles(reduced[-1])
    assert resolution.violations == ()
    assert set(resolution.roles) == {phx.ArrayRole.FIXED}


# --- Refusals ---------------------------------------------------------------------------


def _coordinate_basis(
    provider: cpl.ComponentResidualProvider, columns: tuple[int, ...]
) -> phx.rom.ReducedBasisArtifact:
    """Orthonormal coordinate vectors of the provider's state space."""
    identity = np.eye(provider.state_space.size, dtype=np.float64)
    return _artifact(provider, jnp.asarray(identity[:, list(columns)]))


class _HiddenConductivity(phx.StrictModule):
    """A learned conductivity held inside the owner instead of a parameter binding."""

    network: phx.nn.models.MLP

    def __call__(self, points: Array, context: object) -> Array:
        del context
        flat = points.reshape(-1, points.shape[-1])[:, :2]
        values = jax.vmap(self.network)(flat).reshape(points.shape[:-1])
        return 1.0 + values**2


def test_petrov_galerkin_reduction_is_refused(plate: cp.CoupledPlate) -> None:
    provider = cpl.ComponentResidualProvider(plate.triangles, field="u")
    reduction = phx.rom.trial_test_reduction_from_bases(
        _coordinate_basis(provider, (0, 1)), _coordinate_basis(provider, (1, 2))
    )
    galerkin = phx.rom.FullResidualGalerkin(reduction, provider)
    with pytest.raises(ValueError, match="Petrov-Galerkin"):
        cpl.ReducedComponent("triangles", galerkin)


def test_basis_of_another_component_is_refused(plate: cp.CoupledPlate) -> None:
    triangles = cpl.ComponentResidualProvider(plate.triangles, field="u")
    polygons = cpl.ComponentResidualProvider(plate.polygons, field="u")
    reduction = phx.rom.trial_test_reduction_from_bases(
        _coordinate_basis(polygons, (0, 1))
    )
    with pytest.raises(ValueError, match="state space must match"):
        phx.rom.FullResidualGalerkin(reduction, triangles)


def test_trainable_model_hidden_in_the_provider_is_refused() -> None:
    network = phx.nn.models.MLP(
        in_size=2, out_size="scalar", width_size=4, depth=1, key=jr.key(0)
    )
    plate = cp.build_plate(LEVEL, triangle_conductivity=_HiddenConductivity(network))
    provider = cpl.ComponentResidualProvider(plate.triangles, field="u")
    reduction = phx.rom.trial_test_reduction_from_bases(_coordinate_basis(provider, (0,)))
    galerkin = phx.rom.FullResidualGalerkin(reduction, provider)
    violations = phx.resolve_array_roles(galerkin).violations
    assert {kind for _, kind, _ in violations} == {"parameter-under-fixed-ancestor"}
    with pytest.raises(ValueError, match="frozen silently"):
        cpl.ReducedComponent("triangles", galerkin)


class _OwnerResidual(phx.rom.AbstractResidualProvider):
    """A hand-written provider of the triangles' residual: no traces, no fluxes."""

    component: cpl.VariationalComponent
    state_space: phx.linalg.AbstractVectorSpace
    residual_space: phx.linalg.AbstractVectorSpace
    residual_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(self, provider: cpl.ComponentResidualProvider) -> None:
        component = provider.component
        if not isinstance(component, cpl.VariationalComponent):
            raise TypeError("The plate's triangles are a variational component.")
        self.component = component
        self.state_space = provider.state_space
        self.residual_space = provider.residual_space
        self.residual_id = "hand-written-triangle-residual"
        self.support_id = provider.support_id
        self.geometry_id = provider.geometry_id

    def residual(
        self, coordinate: Array, state: Array, state_rate: Array | None, inputs: object, /
    ) -> Array:
        del coordinate, state_rate
        return self.component.residual((state,), inputs)[0]


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("reduction", id="bare-reduction"),
        pytest.param("provider", id="foreign-provider"),
    ],
)
def test_reduced_component_requires_a_component_galerkin_model(
    plate: cp.CoupledPlate, kind: str
) -> None:
    provider = cpl.ComponentResidualProvider(plate.triangles, field="u")
    reduction = phx.rom.trial_test_reduction_from_bases(_coordinate_basis(provider, (0,)))
    match kind:
        case "reduction":
            with pytest.raises(TypeError, match="FullResidualGalerkin"):
                cpl.ReducedComponent("triangles", reduction)  # ty: ignore[invalid-argument-type]
        case "provider":
            galerkin = phx.rom.FullResidualGalerkin(reduction, _OwnerResidual(provider))
            with pytest.raises(TypeError, match="ComponentResidualProvider"):
                cpl.ReducedComponent("triangles", galerkin)
        case _:
            raise ValueError(f"Unknown case {kind!r}.")
