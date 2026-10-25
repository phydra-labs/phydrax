#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from phydrax._admissibility import AdmissibilityReason
from phydrax._differentiation import (
    ComponentAuthority,
    DERIVATIVE_UNSUPPORTED,
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    GradientLevel,
    RegularityPolicy,
    SurfaceDerivative,
)
from phydrax._strict import StrictModule
from phydrax._trainable import (
    combine_parameters,
    fixed_field,
    NonTrainableState,
    partition_parameters,
)
from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticConditionField,
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticGraphExecutionPlan,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    TOTAL_CHARGE_PORT,
)
from phydrax.atomistic._graph import AtomisticGraphLayout, AtomisticGraphPayload
from phydrax.atomistic._hessian import (
    atomistic_edge_distance_locus,
    ATOMISTIC_HESSIAN_REQUEST,
    ATOMISTIC_NONCOINCIDENT_SUPPORT,
    AtomisticHessianBlockSelection,
    bind_atomistic_hessian,
    prepare_atomistic_hessian,
)
from phydrax.atomistic._model_inputs import AtomisticGraphInputs
from phydrax.atomistic._potential import (
    AbstractAtomisticModel,
    AbstractAtomisticPotential,
    AtomisticPotentialCapabilities,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.linalg import LinearCapabilityError, MaterializationPolicy, materialize
from phydrax.nn.atomistic import (
    MACEArchitecture,
    MACEPotential,
    NequIPPotential,
    PaiNNPotential,
)
from phydrax.units import ANGSTROM, ELECTRONVOLT, ELEMENTARY_CHARGE


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
INPUT = DerivativeSurface.INPUT
PARAMETER = DerivativeSurface.MODEL_PARAMETER
MIXED = DifferentiationRequest(
    (INPUT, PARAMETER), order=3, authority=ComponentAuthority.MODEL
)
COORDINATE_THIRD = DifferentiationRequest(
    (INPUT,), order=3, authority=ComponentAuthority.MODEL
)
AE = RegularityPolicy(allow_almost_everywhere=True)
_EVALUATIONS = [0]


def _payload(
    batch: AtomisticBatch, execution: Any, *, cutoff: float
) -> AtomisticGraphPayload:
    return AtomisticGraphLayout.for_batch(batch, execution, cutoff=cutoff).prepare(batch)


def _request(*derivatives: DifferentiationRequest, blocks: bool = False) -> Any:
    observables = (AtomisticObservable.HESSIAN_ACTION,) + (
        (AtomisticObservable.HESSIAN_BLOCKS,) if blocks else ()
    )
    return AtomisticObservableRequest(
        observables, derivatives=(ATOMISTIC_HESSIAN_REQUEST, *derivatives)
    )


class _Configuration(StrictModule, NonTrainableState):
    cutoff: float = eqx.field(static=True)


def _smooth_contract(*, support: str | None = None) -> DerivativeContract:
    return DerivativeContract(
        (
            SurfaceDerivative(INPUT, GradientLevel.SMOOTH),
            SurfaceDerivative(PARAMETER, GradientLevel.SMOOTH),
        ),
        route=DerivativeRoute.DIRECT,
        regularity=DerivativeRegularity.smooth(support=support),
    )


type _QuadraticContract = Literal["declared", "undeclared", "nonconservative"]


class _QuadraticPotential(AbstractAtomisticPotential):
    """``E_case = stiffness * x^T K x / 2`` with one fixed SPD ``K`` per case.

    ``contract`` selects the declaration over the same numerics: ``declared``
    is a smooth quadratic, ``undeclared`` declares no derivative contract and
    ``nonconservative`` declares a nonconservative energy.
    """

    stiffness: Array
    coupling: Array = fixed_field()
    configuration: _Configuration
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    contract: _QuadraticContract = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        stiffness: float,
        coupling: np.ndarray,
        *,
        contract: _QuadraticContract = "declared",
    ) -> None:
        self.stiffness = jnp.asarray(stiffness, dtype=jnp.float64)
        self.coupling = jnp.asarray(coupling, dtype=jnp.float64)
        self.configuration = _Configuration(cutoff=5.0)
        self.scale = SCALE
        self.precision = AtomisticPrecisionPolicy()
        self.contract = contract
        self.architecture_id = "test-quadratic"
        self.method_id = "negative-position-gradient"

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        return AtomisticPotentialCapabilities(
            conservative_energy=self.contract != "nonconservative"
        )

    @property
    def derivative_contract(self) -> DerivativeContract | None:
        match self.contract:
            case "declared":
                return DerivativeContract(
                    (
                        SurfaceDerivative(INPUT, GradientLevel.SMOOTH),
                        SurfaceDerivative(PARAMETER, GradientLevel.SMOOTH),
                    ),
                    route=DerivativeRoute.DIRECT,
                    regularity=DerivativeRegularity.smooth(degree_bound=2),
                )
            case "undeclared":
                return None
            case "nonconservative":
                return _smooth_contract()

    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        if batch.scale.scale_id != self.scale.scale_id:
            raise ValueError("scale mismatch")

    def evaluate_graph(
        self, inputs: AtomisticGraphInputs, request: Any = None, /
    ) -> AtomisticModelOutputs:
        _EVALUATIONS[0] += 1
        cases, atoms = inputs.case_count, inputs.atom_capacity
        flat = inputs.positions.reshape((cases, atoms * 3))
        coupled = flat @ self.coupling
        atom_energy = (
            0.5 * self.stiffness * (flat * coupled).reshape((cases, atoms, 3)).sum(-1)
        ) * inputs.atom_mask.reshape((cases, atoms))
        return AtomisticModelOutputs(
            atom_energy.sum(axis=1), atom_energy, inputs.successful
        )


class _PropertyModel(AbstractAtomisticModel):
    configuration: _Configuration
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.configuration = _Configuration(cutoff=5.0)
        self.scale = SCALE
        self.precision = AtomisticPrecisionPolicy()
        self.architecture_id = "test-property"
        self.method_id = "report-only"

    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        del batch

    def evaluate_graph(
        self, inputs: AtomisticGraphInputs, request: Any = None, /
    ) -> AtomisticModelOutputs:
        raise AssertionError("A refused property model must not be evaluated.")


class _MorsePotential(AbstractAtomisticPotential):
    """All-pair Morse energy computed directly from positions (no graph edges)."""

    depth: Array
    width: Array
    configuration: _Configuration
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    equilibrium: float = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, depth: float, width: float, equilibrium: float) -> None:
        self.depth = jnp.asarray(depth, dtype=jnp.float64)
        self.width = jnp.asarray(width, dtype=jnp.float64)
        self.configuration = _Configuration(cutoff=6.0)
        self.scale = SCALE
        self.precision = AtomisticPrecisionPolicy()
        self.equilibrium = float(equilibrium)
        self.architecture_id = "test-morse"
        self.method_id = "negative-position-gradient"

    @property
    def derivative_contract(self) -> DerivativeContract | None:
        return _smooth_contract(support=ATOMISTIC_NONCOINCIDENT_SUPPORT)

    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        del batch

    def evaluate_graph(
        self, inputs: AtomisticGraphInputs, request: Any = None, /
    ) -> AtomisticModelOutputs:
        cases, atoms = inputs.case_count, inputs.atom_capacity
        positions = inputs.positions.reshape((cases, atoms, 3))
        mask = inputs.atom_mask.reshape((cases, atoms))
        pair = (
            mask[:, :, None] & mask[:, None, :] & ~jnp.eye(atoms, dtype=jnp.bool_)[None]
        )
        difference = positions[:, :, None, :] - positions[:, None, :, :]
        squared = jnp.sum(difference * difference, axis=-1)
        radius = jnp.sqrt(jnp.where(pair, squared, 1.0))
        decay = jnp.exp(-self.width * (radius - self.equilibrium))
        energy = jnp.where(pair, self.depth * (1.0 - decay) ** 2, 0.0)
        atom_energy = 0.5 * jnp.sum(energy, axis=2)
        return AtomisticModelOutputs(
            atom_energy.sum(axis=1), atom_energy, inputs.successful
        )


class _EnvelopedPairPotential(AbstractAtomisticPotential):
    """Graph-edge harmonic pairs under a C1 cosine cutoff envelope."""

    stiffness: Array
    configuration: _Configuration
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, stiffness: float, cutoff: float) -> None:
        self.stiffness = jnp.asarray(stiffness, dtype=jnp.float64)
        self.configuration = _Configuration(cutoff=cutoff)
        self.scale = SCALE
        self.precision = AtomisticPrecisionPolicy()
        self.architecture_id = "test-enveloped-pair"
        self.method_id = "negative-position-gradient"

    @property
    def derivative_contract(self) -> DerivativeContract | None:
        return DerivativeContract(
            (
                SurfaceDerivative(INPUT, GradientLevel.SMOOTH),
                SurfaceDerivative(PARAMETER, GradientLevel.SMOOTH),
            ),
            route=DerivativeRoute.DIRECT,
            regularity=DerivativeRegularity.piecewise_smooth(
                continuity=1,
                conditions=(atomistic_edge_distance_locus(self.configuration.cutoff),),
                support=ATOMISTIC_NONCOINCIDENT_SUPPORT,
            ),
        )

    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        del batch

    def evaluate_graph(
        self, inputs: AtomisticGraphInputs, request: Any = None, /
    ) -> AtomisticModelOutputs:
        ir = inputs.graph.graph
        if ir.receivers is None or ir.edge_mask is None:
            raise ValueError("The pair potential needs the bound graph's explicit edges.")
        distance = jnp.asarray(ir.edges["distance"])[:, 0]
        scaled = distance / self.configuration.cutoff
        envelope = jnp.where(scaled < 1.0, 0.5 * (jnp.cos(jnp.pi * scaled) + 1.0), 0.0)
        edge_energy = jnp.where(
            ir.edge_mask,
            0.25 * self.stiffness * (distance - 1.0) ** 2 * envelope,
            0.0,
        )
        atoms = inputs.case_count * inputs.atom_capacity
        atom_energy = (
            jnp.zeros((atoms,), dtype=distance.dtype).at[ir.receivers].add(edge_energy)
        )
        atom_energy = atom_energy.reshape((inputs.case_count, inputs.atom_capacity))
        return AtomisticModelOutputs(
            atom_energy.sum(axis=1), atom_energy, inputs.successful
        )


def _execution(maximum_neighbors: int = 4) -> AtomisticGraphExecutionPlan:
    return AtomisticGraphExecutionPlan(maximum_neighbors, maximum_dense_atoms=4)


def _coupling(atoms: int) -> np.ndarray:
    raw = np.random.default_rng(3).normal(size=(3 * atoms, 3 * atoms))
    return raw @ raw.T / (3 * atoms) + np.eye(3 * atoms)


def _quadratic_batch() -> AtomisticBatch:
    positions = np.random.default_rng(4).normal(size=(2, 3, 3))
    return AtomisticBatch(
        np.ones((2, 3), dtype=np.int32), positions, np.ones((2, 3)), SCALE
    )


def _morse_batch() -> AtomisticBatch:
    positions = np.asarray(
        [[[0.0, 0.0, 0.0], [1.2, 0.1, -0.1], [0.3, 1.1, 0.2], [9.0, 9.0, 9.0]]]
    )
    return AtomisticBatch(
        np.asarray([[1, 1, 1, 0]], dtype=np.int32), positions, np.ones((1, 4)), SCALE
    )


def _morse_hessian(
    positions: Array, depth: Array, width: Array, equilibrium: float
) -> Array:
    """Independent closed-form pair Hessian ``(atom, 3, atom, 3)``."""
    atoms = positions.shape[0]
    hessian = jnp.zeros((atoms, 3, atoms, 3), dtype=positions.dtype)
    identity = jnp.eye(3, dtype=positions.dtype)
    for left in range(atoms):
        for right in range(left + 1, atoms):
            difference = positions[left] - positions[right]
            radius = jnp.sqrt(jnp.sum(difference * difference))
            unit = difference / radius
            decay = jnp.exp(-width * (radius - equilibrium))
            first = 2.0 * depth * width * decay * (1.0 - decay)
            second = 2.0 * depth * width**2 * decay * (2.0 * decay - 1.0)
            outer = jnp.outer(unit, unit)
            block = second * outer + first / radius * (identity - outer)
            hessian = (
                hessian.at[left, :, left, :]
                .add(block)
                .at[right, :, right, :]
                .add(block)
                .at[left, :, right, :]
                .add(-block)
                .at[right, :, left, :]
                .add(-block)
            )
    return hessian


def _vector(shape: tuple[int, ...], seed: int) -> Array:
    return jnp.asarray(np.random.default_rng(seed).normal(size=shape))


def test_quadratic_sign_reused_actions_and_bounded_dense_blocks() -> None:
    coupling = _coupling(3)
    model = _QuadraticPotential(1.7, coupling)
    batch = _quadratic_batch()
    execution = _execution()
    prepared = prepare_atomistic_hessian(
        model, batch, execution, request=_request(blocks=True)
    )
    operator = prepared.operator
    flat = np.asarray(batch.positions).reshape((2, 9))
    np.testing.assert_allclose(
        operator.forces.reshape((2, 9)), -1.7 * flat @ coupling, rtol=1e-12
    )
    assert bool(jnp.all(operator.action_valid))
    assert operator.properties.self_adjoint
    direction = _vector((2, 3, 3), 5)
    before = _EVALUATIONS[0]
    first = operator.mv(direction)
    second = operator.mv(direction)
    transposed = operator.transpose_mv(direction)
    adjoint = operator.adjoint_mv(direction)
    blocks = operator.blocks(
        AtomisticHessianBlockSelection((0, 1, 2), (0, 1, 2)), MaterializationPolicy()
    )
    assert _EVALUATIONS[0] == before, "actions must reuse the prepared linearization"
    expected = 1.7 * np.asarray(direction).reshape((2, 9)) @ coupling
    np.testing.assert_allclose(first.reshape((2, 9)), expected, rtol=1e-12)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(transposed, first, rtol=1e-12)
    np.testing.assert_allclose(adjoint, first, rtol=1e-12)
    # Positive curvature of a positive-definite quadratic: H = +K, F = -K x.
    assert float(jnp.sum(direction * first)) > 0.0
    assert blocks.observable is AtomisticObservable.HESSIAN_BLOCKS
    assert blocks.unit is not None
    assert bool(jnp.all(blocks.available))
    np.testing.assert_allclose(
        np.asarray(blocks.values).reshape((2, 9, 9)),
        np.broadcast_to(1.7 * coupling, (2, 9, 9)),
        rtol=1e-12,
    )
    with pytest.raises(LinearCapabilityError):
        materialize(operator, MaterializationPolicy())
    with pytest.raises(LinearCapabilityError):
        operator.blocks(
            AtomisticHessianBlockSelection((0, 1, 2), (0, 1, 2)),
            MaterializationPolicy(max_entries=10),
        )
    with pytest.raises(ValueError, match="atom capacity"):
        operator.blocks(
            AtomisticHessianBlockSelection((3,), (0,)), MaterializationPolicy()
        )


def test_nonlinear_hessian_matches_independent_closed_form_and_duality() -> None:
    model = _MorsePotential(0.8, 1.3, 1.1)
    batch = _morse_batch()
    prepared = prepare_atomistic_hessian(
        model, batch, _execution(), request=_request(blocks=True)
    )
    operator = prepared.operator
    reference = _morse_hessian(batch.positions[0, :3], model.depth, model.width, 1.1)
    blocks = operator.blocks(
        AtomisticHessianBlockSelection((0, 1, 2), (0, 1, 2)), MaterializationPolicy()
    )
    np.testing.assert_allclose(blocks.values[0], reference, rtol=1e-10, atol=1e-12)
    padded = operator.blocks(
        AtomisticHessianBlockSelection((0, 3), (0,)), MaterializationPolicy()
    )
    assert not bool(padded.available[0])
    assert bool(jnp.all(jnp.isnan(padded.values)))
    left, right = _vector((1, 4, 3), 11), _vector((1, 4, 3), 12)
    image = operator.mv(right)
    np.testing.assert_allclose(
        jnp.vdot(left, image), jnp.vdot(operator.transpose_mv(left), right), rtol=1e-12
    )
    np.testing.assert_allclose(operator.adjoint_mv(left), operator.transpose_mv(left))
    np.testing.assert_allclose(operator.mv(left), operator.transpose_mv(left), rtol=1e-10)
    np.testing.assert_allclose(image[0, 3], 0.0, atol=0.0)
    translation = jnp.zeros((1, 4, 3)).at[0, :3].set(jnp.asarray([0.3, -0.2, 0.5]))
    np.testing.assert_allclose(operator.mv(translation), 0.0, atol=1e-12)


def test_mixed_parameter_derivative_of_hvp_objective() -> None:
    coupling = _coupling(3)
    model = _QuadraticPotential(1.7, coupling)
    batch = _quadratic_batch()
    execution = _execution()
    payload = _payload(batch, execution, cutoff=5.0)
    probe, target = _vector((2, 3, 3), 21), _vector((2, 3, 3), 22)

    def objective(stiffness: Array) -> Array:
        student = eqx.tree_at(lambda value: value.stiffness, model, stiffness)
        operator = bind_atomistic_hessian(
            student,
            payload,
            batch.positions,
            request=_request(MIXED),
        )
        return jnp.sum((operator.mv(probe) - target) ** 2)

    gradient = jax.jit(jax.grad(objective))(model.stiffness)
    action = np.asarray(probe).reshape((2, 9)) @ coupling
    residual = 1.7 * action - np.asarray(target).reshape((2, 9))
    np.testing.assert_allclose(gradient, 2.0 * np.sum(residual * action), rtol=1e-12)

    morse = _MorsePotential(0.8, 1.3, 1.1)
    morse_batch = _morse_batch()
    morse_payload = _payload(morse_batch, execution, cutoff=6.0)
    morse_probe = jnp.zeros((1, 4, 3)).at[0, :3].set(_vector((3, 3), 23))
    morse_target = _vector((3, 3), 24)

    def morse_objective(parameters: tuple[Array, Array]) -> Array:
        student = eqx.tree_at(lambda value: (value.depth, value.width), morse, parameters)
        operator = bind_atomistic_hessian(
            student,
            morse_payload,
            morse_batch.positions,
            request=_request(MIXED),
        )
        return jnp.sum((operator.mv(morse_probe)[0, :3] - morse_target) ** 2)

    def independent(parameters: tuple[Array, Array]) -> Array:
        hessian = _morse_hessian(morse_batch.positions[0, :3], *parameters, 1.1)
        action = jnp.einsum("iajb,jb->ia", hessian, morse_probe[0, :3])
        return jnp.sum((action - morse_target) ** 2)

    parameters = (morse.depth, morse.width)
    observed = jax.grad(morse_objective)(parameters)
    expected = jax.grad(independent)(parameters)
    for left, right in zip(observed, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-9)
    step = 1e-5
    depth, width = parameters
    shifted = (
        ((depth + step, width), (depth - step, width)),
        ((depth, width + step), (depth, width - step)),
    )
    for index, (plus, minus) in enumerate(shifted):
        difference = (morse_objective(plus) - morse_objective(minus)) / (2.0 * step)
        np.testing.assert_allclose(observed[index], difference, rtol=1e-6)


def test_unadmitted_enclosing_derivatives_refuse_and_third_order_is_admitted() -> None:
    model = _MorsePotential(0.8, 1.3, 1.1)
    batch = _morse_batch()
    execution = _execution()
    payload = _payload(batch, execution, cutoff=6.0)
    probe = jnp.zeros((1, 4, 3)).at[0, :3].set(_vector((3, 3), 31))
    weight = jnp.zeros((1, 4, 3)).at[0, :3].set(_vector((3, 3), 32))

    def parameter_objective(depth: Array) -> Array:
        student = eqx.tree_at(lambda value: value.depth, model, depth)
        operator = bind_atomistic_hessian(student, payload, batch.positions)
        return jnp.sum(operator.mv(probe) ** 2)

    with pytest.raises(ValueError, match=DERIVATIVE_UNSUPPORTED):
        jax.grad(parameter_objective)(model.depth)

    def coordinate_objective(positions: Array, request: Any) -> Array:
        operator = bind_atomistic_hessian(model, payload, positions, request=request)
        return jnp.vdot(weight, operator.mv(probe))

    with pytest.raises(ValueError, match=DERIVATIVE_UNSUPPORTED):
        jax.grad(coordinate_objective)(batch.positions, _request())
    observed = jax.grad(coordinate_objective)(batch.positions, _request(COORDINATE_THIRD))

    def independent(active: Array) -> Array:
        hessian = _morse_hessian(active, model.depth, model.width, 1.1)
        return jnp.einsum("ia,iajb,jb->", weight[0, :3], hessian, probe[0, :3])

    np.testing.assert_allclose(
        observed[0, :3], jax.grad(independent)(batch.positions[0, :3]), rtol=1e-9
    )
    np.testing.assert_allclose(observed[0, 3], 0.0, atol=0.0)


def _conditioning(charge: float) -> AtomisticConditioning:
    spec = AtomisticConditioningSpec(
        (
            AtomisticConditionField(
                "charge",
                TOTAL_CHARGE_PORT,
                unit=ELEMENTARY_CHARGE,
                minimum=-1.0,
                maximum=1.0,
            ),
        )
    )
    return AtomisticConditioning(
        spec,
        {"charge": np.full((2,), charge, dtype=np.float64)},
        case_count=2,
        atom_capacity=3,
    )


def test_stale_condition_key_revision_geometry_and_resource_epochs_refuse() -> None:
    model = _QuadraticPotential(1.7, _coupling(3))
    batch = _quadratic_batch()
    execution = _execution()
    payload = _payload(batch, execution, cutoff=5.0)
    bound: dict[str, Any] = {
        "payload": payload,
        "conditioning": _conditioning(0.0),
        "key": jr.key(0),
        "resource_epochs": np.asarray([3, 3], dtype=np.int64),
    }
    prepared = prepare_atomistic_hessian(model, batch, execution, **bound)
    assert prepared.require_current(model, batch, execution, **bound) is prepared.operator
    changes: dict[str, tuple[Any, dict[str, Any]]] = {
        "conditioning": (model, {"conditioning": _conditioning(0.5)}),
        "key": (model, {"key": jr.key(1)}),
        "resource-epochs": (
            model,
            {"resource_epochs": np.asarray([3, 4], dtype=np.int64)},
        ),
        "geometry": (model, {"positions": batch.positions + 0.01}),
        "model-revision": (
            eqx.tree_at(lambda value: value.stiffness, model, jnp.asarray(1.8)),
            {},
        ),
    }
    for component, (potential, change) in changes.items():
        with pytest.raises(
            ValueError, match=f"Stale atomistic Hessian binding: {component}"
        ):
            prepared.require_current(potential, batch, execution, **{**bound, **change})


def test_property_nonconservative_undeclared_and_unsupported_requests_refuse() -> None:
    batch = _quadratic_batch()
    execution = _execution()
    coupling = _coupling(3)
    with pytest.raises(ValueError, match="property-only"):
        prepare_atomistic_hessian(_PropertyModel(), batch, execution)
    with pytest.raises(ValueError, match="conservative"):
        prepare_atomistic_hessian(
            _QuadraticPotential(1.0, coupling, contract="nonconservative"),
            batch,
            execution,
        )
    with pytest.raises(ValueError, match="no derivative contract"):
        prepare_atomistic_hessian(
            _QuadraticPotential(1.0, coupling, contract="undeclared"), batch, execution
        )
    model = _QuadraticPotential(1.0, coupling)
    strain = DifferentiationRequest(
        (INPUT, DerivativeSurface.PHYSICAL_PARAMETER),
        order=3,
        authority=ComponentAuthority.MODEL,
    )
    with pytest.raises(ValueError, match=DERIVATIVE_UNSUPPORTED):
        prepare_atomistic_hessian(model, batch, execution, request=_request(strain))
    with pytest.raises(ValueError, match="coordinate Hessian derivative request"):
        prepare_atomistic_hessian(
            model,
            batch,
            execution,
            request=AtomisticObservableRequest(
                (AtomisticObservable.HESSIAN_ACTION,), derivatives=(COORDINATE_THIRD,)
            ),
        )
    with pytest.raises(ValueError, match="direct-forces"):
        prepare_atomistic_hessian(
            model,
            batch,
            execution,
            request=AtomisticObservableRequest(
                (AtomisticObservable.HESSIAN_ACTION, AtomisticObservable.DIRECT_FORCES),
                derivatives=(ATOMISTIC_HESSIAN_REQUEST,),
            ),
        )
    operator = prepare_atomistic_hessian(model, batch, execution).operator
    with pytest.raises(ValueError, match="not requested"):
        operator.blocks(
            AtomisticHessianBlockSelection((0,), (0,)), MaterializationPolicy()
        )


def test_graph_capacity_failure_is_refused_and_invalidates_pure_actions() -> None:
    model = _QuadraticPotential(1.0, _coupling(3))
    batch = _quadratic_batch()
    execution = _execution(maximum_neighbors=1)
    with pytest.raises(ValueError, match="capacity"):
        prepare_atomistic_hessian(model, batch, execution)
    operator = bind_atomistic_hessian(
        model, _payload(batch, execution, cutoff=5.0), batch.positions
    )
    assert bool(jnp.all(operator.overflow))
    assert not bool(jnp.any(operator.action_valid))
    status = operator.admission(ATOMISTIC_HESSIAN_REQUEST).runtime_status
    assert status is not None
    assert bool(jnp.all(status & int(AdmissibilityReason.CAPACITY_INSUFFICIENT)))
    assert bool(jnp.all(jnp.isnan(operator.mv(_vector((2, 3, 3), 41)))))


def _pair_batch(positions: Any) -> AtomisticBatch:
    return AtomisticBatch(
        np.ones((1, 3), dtype=np.int32), np.asarray([positions]), np.ones((1, 3)), SCALE
    )


def test_c1_cutoff_needs_explicit_almost_everywhere_policy_and_point_margin() -> None:
    model = _EnvelopedPairPotential(2.0, 2.5)
    execution = _execution()
    generic = _pair_batch([[0.0, 0.0, 0.0], [1.3, 0.2, 0.0], [0.1, 1.4, 0.3]])
    with pytest.raises(ValueError, match="almost-everywhere-not-allowed"):
        prepare_atomistic_hessian(model, generic, execution)
    operator = prepare_atomistic_hessian(model, generic, execution, policy=AE).operator
    assert operator.unresolved_conditions == ()
    assert not operator.properties.self_adjoint
    assert operator.admission(ATOMISTIC_HESSIAN_REQUEST).levels == (
        GradientLevel.ALMOST_EVERYWHERE,
    )
    assert bool(jnp.all(operator.action_valid))
    direction = _vector((1, 3, 3), 51)
    payload = _payload(generic, execution, cutoff=2.5)

    def energy(positions: Array) -> Array:
        return jnp.sum(
            model.energy(generic, execution, positions=positions, payload=payload)
        )

    expected = jax.jvp(jax.grad(energy), (generic.positions,), (direction,))[1]
    np.testing.assert_allclose(operator.mv(direction), expected, rtol=1e-10, atol=1e-12)
    on_join = _pair_batch([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    joined = prepare_atomistic_hessian(model, on_join, execution, policy=AE).operator
    assert not bool(joined.action_valid[0])
    status = joined.admission(ATOMISTIC_HESSIAN_REQUEST).runtime_status
    assert status is not None
    assert int(status[0, 0]) & int(AdmissibilityReason.OUTSIDE_SUPPORT)
    assert bool(jnp.all(jnp.isnan(joined.mv(direction))))
    coincident = _pair_batch([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    collapsed = bind_atomistic_hessian(
        model,
        _payload(coincident, execution, cutoff=2.5),
        coincident.positions,
        policy=AE,
    )
    assert not bool(collapsed.action_valid[0])


def _molecule() -> AtomisticBatch:
    return AtomisticBatch.from_structure(
        AtomicStructure(
            np.asarray([8, 1, 1], dtype=np.int64),
            np.asarray(
                [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 0.8, 0.2]], dtype=np.float64
            ),
            np.asarray([15.999, 1.008, 1.008], dtype=np.float64),
            SCALE,
        )
    )


def _reference_hvp(
    model: Any, batch: AtomisticBatch, execution: Any, payload: Any, direction: Array
) -> Array:
    def energy(positions: Array) -> Array:
        return jnp.sum(
            model.energy(batch, execution, positions=positions, payload=payload)
        )

    return jax.jvp(jax.grad(energy), (batch.positions,), (direction,))[1]


def _parameter_hvp_check(
    model: Any,
    batch: AtomisticBatch,
    execution: Any,
    payload: AtomisticGraphPayload,
    policy: RegularityPolicy,
    seed: int,
) -> None:
    probe = _vector(batch.positions.shape, seed)
    target = _vector(batch.positions.shape, seed + 1)
    parameters, state, fixed = partition_parameters(model)

    def objective(values: Any) -> Array:
        operator = bind_atomistic_hessian(
            combine_parameters(values, state, fixed),
            payload,
            batch.positions,
            request=_request(MIXED),
            policy=policy,
        )
        return jnp.sum((operator.mv(probe) - target) ** 2)

    gradient = jax.jit(jax.grad(objective))(parameters)
    leaves, treedef = jax.tree.flatten(parameters)
    rng = np.random.default_rng(seed + 2)
    direction = jax.tree.unflatten(
        treedef, [jnp.asarray(rng.normal(size=leaf.shape)) for leaf in leaves]
    )
    directional = sum(
        jnp.vdot(left, right)
        for left, right in zip(
            jax.tree.leaves(gradient), jax.tree.leaves(direction), strict=True
        )
    )
    step = 1e-6

    def shifted(sign: float) -> Any:
        return jax.tree.map(
            lambda value, delta: value + sign * step * delta, parameters, direction
        )

    evaluate = jax.jit(objective)
    difference = (evaluate(shifted(1.0)) - evaluate(shifted(-1.0))) / (2.0 * step)
    np.testing.assert_allclose(directional, difference, rtol=1e-5)


def test_painn_hessian_is_almost_everywhere_with_explicit_unresolved_cone() -> None:
    model = PaiNNPotential(
        SCALE,
        cutoff=2.5,
        feature_count=8,
        interaction_count=2,
        radial_basis_count=6,
        key=jr.key(7),
    )
    batch = _molecule()
    execution = _execution(3)
    payload = _payload(batch, execution, cutoff=2.5)
    with pytest.raises(ValueError, match="almost-everywhere-not-allowed"):
        prepare_atomistic_hessian(model, batch, execution, payload=payload)
    operator = prepare_atomistic_hessian(
        model, batch, execution, payload=payload, policy=AE
    ).operator
    assert (
        "atomistic-nonsmooth-locus:painn-vector-norm-zero"
        in operator.unresolved_conditions
    )
    assert "singular-part-ignored" in operator.unresolved_conditions
    assert bool(jnp.all(operator.action_valid))
    direction = _vector(batch.positions.shape, 61)
    np.testing.assert_allclose(
        operator.mv(direction),
        _reference_hvp(model, batch, execution, payload, direction),
        rtol=1e-9,
        atol=1e-11,
    )
    _parameter_hvp_check(model, batch, execution, payload, AE, 62)


def test_periodic_painn_binds_cell_and_refuses_stale_cell() -> None:
    model = PaiNNPotential(
        SCALE,
        cutoff=2.5,
        feature_count=8,
        interaction_count=2,
        radial_basis_count=6,
        key=jr.key(7),
    )
    cell = np.asarray([[2.8, 0.0, 0.0], [0.3, 2.9, 0.0], [0.2, 0.1, 3.0]])
    batch = AtomisticBatch.from_structure(
        AtomicStructure(
            np.asarray([1, 8], dtype=np.int64),
            np.asarray([[0.0, 0.0, 0.0], [1.1, 0.3, 0.2]], dtype=np.float64),
            np.asarray([1.008, 15.999], dtype=np.float64),
            SCALE,
            cell=cell,
            periodic_axes=np.asarray([True, True, True], dtype=np.bool_),
        )
    )
    execution = AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=2048,
            maximum_degree=64,
            maximum_images=125,
        ),
    )
    payload = _payload(batch, execution, cutoff=2.5)
    prepared = prepare_atomistic_hessian(
        model, batch, execution, payload=payload, policy=AE
    )
    direction = _vector(batch.positions.shape, 71)
    np.testing.assert_allclose(
        prepared.operator.mv(direction),
        _reference_hvp(model, batch, execution, payload, direction),
        rtol=1e-9,
        atol=1e-11,
    )
    with pytest.raises(ValueError, match="Stale atomistic Hessian binding: cell"):
        prepared.require_current(
            model,
            batch,
            execution,
            payload=payload,
            cell_vectors=np.asarray([cell * 1.01]),
        )


def test_nequip_hessian_and_mixed_parameter_hvp() -> None:
    model = NequIPPotential(
        SCALE,
        cutoff=2.5,
        feature_count=3,
        interaction_count=2,
        radial_basis_count=4,
        key=jr.key(27),
    )
    batch = _molecule()
    execution = _execution(3)
    payload = _payload(batch, execution, cutoff=2.5)
    operator = prepare_atomistic_hessian(
        model, batch, execution, payload=payload, policy=AE
    ).operator
    assert operator.unresolved_conditions == ()
    direction = _vector(batch.positions.shape, 81)
    np.testing.assert_allclose(
        operator.mv(direction),
        _reference_hvp(model, batch, execution, payload, direction),
        rtol=1e-9,
        atol=1e-11,
    )
    _parameter_hvp_check(model, batch, execution, payload, AE, 82)


def _mace() -> MACEPotential:
    return MACEPotential(
        SCALE,
        MACEArchitecture(
            species=(1, 8),
            cutoff=3.0,
            radial_basis_count=4,
            cutoff_power=5,
            channel_count=3,
            hidden_degree=1,
            edge_degree=2,
            interactions=("real-agnostic-density", "real-agnostic-residual"),
            correlations=(2, 2),
            radial_widths=(6,),
            readout_width=4,
            average_neighbor_count=2.0,
            energy_scaling="scale-shift",
        ),
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        energy_scale=np.asarray([1.3], dtype=np.float64),
        energy_shift=np.asarray([0.2], dtype=np.float64),
        key=jr.key(5),
    )


def test_mace_classical_hessian_and_almost_everywhere_third_order_training() -> None:
    model = _mace()
    batch = _molecule()
    execution = _execution(3)
    payload = _payload(batch, execution, cutoff=3.0)
    operator = prepare_atomistic_hessian(
        model, batch, execution, payload=payload
    ).operator
    assert operator.properties.self_adjoint
    assert operator.unresolved_conditions == ()
    direction = _vector(batch.positions.shape, 91)
    np.testing.assert_allclose(
        operator.mv(direction),
        _reference_hvp(model, batch, execution, payload, direction),
        rtol=1e-9,
        atol=1e-11,
    )
    with pytest.raises(ValueError, match="almost-everywhere-not-allowed"):
        prepare_atomistic_hessian(
            model, batch, execution, payload=payload, request=_request(MIXED)
        )
    _parameter_hvp_check(model, batch, execution, payload, AE, 92)
