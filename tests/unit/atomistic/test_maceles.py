"""Behavioral tests of native MACELES: MACE plus the latent Ewald energy.

The finite latent energy is checked against an independent dense O(N^2)
transcription of the source kernel ``k_e q_i q_j erf(r / (sqrt2 sigma)) / r``
(``k_e = N / (2 pi)``) with the source self policy; the periodic energy and the
Born charges use the independent host references of
``tests/_support/atomistic_latent_ewald_reference.py``.
"""

import copy
import functools
import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from phydrax._differentiation import RegularityPolicy
from phydrax._model._structure import model_recipe_array_inventory
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax._tree_math import (
    tree_add_scaled,
    tree_allfinite,
    tree_inner,
    tree_norm,
    tree_scale,
)
from phydrax.atomistic import (
    AtomicStructure,
    atomistic_energy_derivatives,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticGraphLayout,
    AtomisticGraphPayload,
    AtomisticScaleContract,
)
from phydrax.atomistic._conditioning import (
    AtomisticConditionLevel,
    AtomisticObservable,
    AtomisticObservableRequest,
    TOTAL_CHARGE_PORT,
)
from phydrax.atomistic._distillation import ATOMISTIC_HVP_TRAINING_REQUEST
from phydrax.atomistic._hessian import ATOMISTIC_HESSIAN_REQUEST, bind_atomistic_hessian
from phydrax.atomistic._latent_electrostatics import (
    LATENT_CHARGE_PORT,
    LATENT_EWALD_ENERGY_PORT,
    LatentEwaldDefinition,
    LatentEwaldEvaluation,
    LatentEwaldResources,
    SOURCE_NORM_FACTOR,
)
from phydrax.atomistic._model_artifact import (
    ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    AtomisticModelArtifactError,
    model_artifact_from_section,
    model_artifact_section,
)
from phydrax.atomistic._model_inputs import AtomisticGraphInputs
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace_les import MACELESPotential
from phydrax.nn.atomistic._mace_readout import (
    MACELinearReadout,
    MACENonlinearReadout,
)
from phydrax.units import ANGSTROM, BOHR, ELECTRONVOLT, HARTREE
from tests._support.atomistic_latent_ewald_reference import (
    born_charges as reference_born_charges,
    mode_energy,
)
from tests._support.atomistic_long_range_reference import central_difference
from tests._support.atomistic_model_families import deformed_batch, energy_oracle


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
CUTOFF = 3.0
EXECUTION = AtomisticGraphExecutionPlan(8, maximum_dense_atoms=6)
NUMBERS = np.asarray([8, 1, 1, 8, 1], dtype=np.int32)
POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.95, 0.1, 0.0],
        [-0.25, 0.85, 0.2],
        [0.3, -0.9, 0.6],
        [-0.7, -0.3, -0.8],
    ],
    dtype=np.float64,
)
CELL = np.asarray([[2.8, 0.0, 0.0], [0.3, 2.9, 0.0], [0.2, 0.1, 3.0]])
PERIODIC_POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [1.1, 0.3, 0.2], [0.4, 1.6, 1.3], [0.0, 0.0, 0.0]]
)
PERIODIC_MASK = np.asarray([True, True, True, False], dtype=np.bool_)
REQUEST = (
    AtomisticObservable.ENERGY,
    AtomisticObservable.ATOM_ENERGY,
    AtomisticObservable.GLOBAL_ENERGY_COMPONENT,
    AtomisticObservable.LATENT_CHARGES,
)


def _mace(**overrides: Any) -> MACEPotential:
    declaration: dict[str, Any] = {
        "species": (1, 8),
        "cutoff": CUTOFF,
        "radial_basis_count": 4,
        "cutoff_power": 5,
        "channel_count": 2,
        "hidden_degree": 1,
        "edge_degree": 2,
        "interactions": ("real-agnostic-density", "real-agnostic-density-residual"),
        "correlations": (2, 2),
        "radial_widths": (4,),
        "readout_width": 3,
        "average_neighbor_count": 3.0,
        "energy_scaling": "scale-shift",
    }
    declaration.update(overrides)
    architecture = MACEArchitecture(**declaration)
    scaled = architecture.energy_scaling == "scale-shift"
    return MACEPotential(
        SCALE,
        architecture,
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        energy_scale=np.asarray([1.3]) if scaled else None,
        energy_shift=np.asarray([0.2]) if scaled else None,
        key=jr.key(5),
    )


def _maceles(
    *,
    arguments: dict[str, Any] | None = None,
    capacity: int = 5,
    periodic: bool = False,
    mace: MACEPotential | None = None,
) -> MACELESPotential:
    definition = LatentEwaldDefinition.from_source_arguments(arguments or {})
    resources = LatentEwaldResources(
        mode_extents=definition.covering_extents(
            float(np.max(np.linalg.norm(CELL, axis=1)))
        )
        if periodic
        else None
    )
    return MACELESPotential.initialize(
        _mace() if mace is None else mace,
        definition,
        atom_capacity=capacity,
        resources=resources,
        key=jr.key(11),
    )


def _batch(capacity: int | None = None) -> AtomisticBatch:
    masses = np.where(NUMBERS == 8, 15.999, 1.008).astype(np.float64)
    return AtomisticBatch.from_structures(
        (AtomicStructure(NUMBERS, POSITIONS, masses, SCALE),), atom_capacity=capacity
    )


def _periodic_batch() -> AtomisticBatch:
    return AtomisticBatch(
        np.asarray([[1, 8, 1, 0]], dtype=np.int32),
        PERIODIC_POSITIONS[None],
        np.asarray([[1.008, 15.999, 1.008, 1.008]]),
        SCALE,
        atom_mask=PERIODIC_MASK[None],
        cells=CELL[None],
        periodic_axes=np.ones((1, 3), dtype=np.bool_),
    )


def _periodic_execution() -> AtomisticGraphExecutionPlan:
    return AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=512,
            maximum_degree=64,
            maximum_images=125,
        ),
    )


def _payload(batch: AtomisticBatch, execution: Any) -> AtomisticGraphPayload:
    return AtomisticGraphLayout.for_batch(batch, execution, cutoff=CUTOFF).prepare(batch)


def _inputs(
    model: Any,
    batch: AtomisticBatch,
    positions: Any = None,
    payload: AtomisticGraphPayload | None = None,
) -> AtomisticGraphInputs:
    bound = _payload(batch, EXECUTION) if payload is None else payload
    graph = bound.graph(cutoff=CUTOFF, positions=positions)
    return AtomisticGraphInputs(
        graph, batch.scale, species_kind=model.capabilities.species_kind
    )


def _evaluate(
    model: Any,
    batch: AtomisticBatch,
    positions: Any = None,
    *,
    payload: AtomisticGraphPayload | None = None,
    request: Any = REQUEST,
) -> Any:
    return model.evaluate_graph(
        _inputs(model, batch, positions, payload),
        None if request is None else AtomisticObservableRequest(request),
    )


def _dense_latent_energy(
    charges: np.ndarray, positions: np.ndarray, sigma: float, remove_self: bool
) -> float:
    """Source finite kernel over every pair, plus the self energy when kept."""
    coulomb = SOURCE_NORM_FACTOR / (2.0 * math.pi)
    total = 0.0
    count = positions.shape[0]
    for i in range(count):
        for j in range(i + 1, count):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            total += (
                coulomb
                * float(charges[i] * charges[j])
                * math.erf(distance / (math.sqrt(2.0) * sigma))
                / distance
            )
    if not remove_self:
        total += (
            SOURCE_NORM_FACTOR
            / (sigma * (2.0 * math.pi) ** 1.5)
            * float(np.sum(charges * charges))
        )
    return total


def _close(observed: Any, expected: Any, relative: float) -> None:
    observed_ = np.asarray(observed, dtype=np.float64)
    expected_ = np.asarray(expected, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(expected_))))
    np.testing.assert_allclose(observed_, expected_, rtol=0.0, atol=relative * scale)


@pytest.mark.parametrize("remove_self", [True, False])
def test_finite_energy_is_local_mace_plus_the_dense_latent_reference(
    remove_self: bool,
) -> None:
    model = _maceles(arguments={"remove_self_interaction": remove_self})
    model.validate()
    batch = _batch()
    inputs = _inputs(model, batch)
    outputs = model.evaluate_graph(inputs, AtomisticObservableRequest(REQUEST))
    local = model.mace.evaluate_graph(inputs)
    assert bool(outputs.successful[0])
    latent = outputs.observable(AtomisticObservable.LATENT_CHARGES)
    charges = np.asarray(latent.values)
    assert charges.shape == (5,) and np.all(np.isfinite(charges))
    assert float(np.max(np.abs(charges))) > 1e-6
    reference = _dense_latent_energy(charges, POSITIONS, 1.0, remove_self)

    np.testing.assert_allclose(
        outputs.energy[0], local.energy[0] + reference, rtol=1e-12, atol=1e-12
    )
    ledger = outputs.observable(AtomisticObservable.GLOBAL_ENERGY_COMPONENT)
    assert ledger.port.port_id == LATENT_EWALD_ENERGY_PORT.port_id
    assert ledger.level is AtomisticConditionLevel.CASE
    assert ledger.unit is not None
    assert ledger.unit.unit_id == SCALE.energy_unit.unit_id
    np.testing.assert_allclose(ledger.values[0], reference, rtol=1e-12, atol=1e-12)
    # Exactly as the source: atom energies are the scoped local decomposition,
    # the global latent term is only in the total.
    assert not outputs.atom_energy_complete and local.atom_energy_complete
    np.testing.assert_array_equal(
        np.asarray(outputs.atom_energy), np.asarray(local.atom_energy)
    )
    np.testing.assert_allclose(
        jnp.sum(outputs.atom_energy[0]), local.energy[0], rtol=1e-13, atol=1e-12
    )
    evaluation = outputs.evidence[0]
    assert isinstance(evaluation, LatentEwaldEvaluation)
    assert not bool(evaluation.periodic[0])
    np.testing.assert_allclose(
        model.energy(batch, EXECUTION), outputs.energy, rtol=1e-13, atol=1e-13
    )


def test_latent_charges_are_a_typed_latent_channel_not_physical_charges() -> None:
    model = _maceles()
    outputs = _evaluate(model, _batch())
    latent = outputs.observable(AtomisticObservable.LATENT_CHARGES)
    assert latent.port.port_id == LATENT_CHARGE_PORT.port_id
    assert latent.port.port_id != TOTAL_CHARGE_PORT.port_id
    assert latent.level is AtomisticConditionLevel.ATOM
    with pytest.raises(ValueError, match="unavailable"):
        outputs.observable(AtomisticObservable.CHARGES)
    with pytest.raises(ValueError, match="does not provide requested observables"):
        _evaluate(model, _batch(), request=(AtomisticObservable.CHARGES,))
    # The global ledger is always reported, also without a request.
    unrequested = _evaluate(model, _batch(), request=None)
    unrequested.observable(AtomisticObservable.GLOBAL_ENERGY_COMPONENT)
    embeddings = _evaluate(
        model,
        _batch(),
        request=(
            AtomisticObservable.NODE_EMBEDDINGS,
            AtomisticObservable.GRAPH_EMBEDDINGS,
        ),
    )
    embeddings.observable(AtomisticObservable.NODE_EMBEDDINGS)


def test_latent_readouts_copy_each_energy_readout_class_and_scale_the_channel() -> None:
    model = _maceles()
    assert isinstance(model.latent_readouts[0], MACELinearReadout)
    assert isinstance(model.latent_readouts[1], MACENonlinearReadout)

    def doubled(readout: Any) -> Any:
        match readout:
            case MACELinearReadout():
                return MACELinearReadout(2.0 * readout.weight)
            case MACENonlinearReadout():
                return MACENonlinearReadout(
                    readout.first,
                    2.0 * readout.second,
                    activation_scale=readout.activation_scale,
                )
            case _:
                return readout

    scaled = MACELESPotential(
        model.mace,
        tuple(doubled(readout) for readout in model.latent_readouts),
        model.latent_ewald,
    )
    batch = _batch()
    base = _evaluate(model, batch)
    twice = _evaluate(scaled, batch)
    np.testing.assert_allclose(
        twice.observable(AtomisticObservable.LATENT_CHARGES).values,
        2.0 * base.observable(AtomisticObservable.LATENT_CHARGES).values,
        rtol=1e-12,
        atol=1e-14,
    )
    # The latent energy is a quadratic form of the latent channel.
    np.testing.assert_allclose(
        twice.observable(AtomisticObservable.GLOBAL_ENERGY_COMPONENT).values,
        4.0 * base.observable(AtomisticObservable.GLOBAL_ENERGY_COMPONENT).values,
        rtol=1e-12,
        atol=1e-13,
    )


@pytest.mark.parametrize("remove_self", [True, False])
def test_periodic_energy_is_local_mace_plus_the_mode_enumeration(
    remove_self: bool,
) -> None:
    model = _maceles(
        arguments={"remove_self_interaction": remove_self}, capacity=4, periodic=True
    )
    batch = _periodic_batch()
    payload = _payload(batch, _periodic_execution())
    inputs = _inputs(model, batch, payload=payload)
    outputs = model.evaluate_graph(inputs, AtomisticObservableRequest(REQUEST))
    local = model.mace.evaluate_graph(inputs)
    assert bool(outputs.successful[0])
    charges = np.asarray(outputs.observable(AtomisticObservable.LATENT_CHARGES).values)
    assert charges[3] == 0.0
    reference = mode_energy(
        charges[:3], PERIODIC_POSITIONS[:3], CELL, 1.0, 2.0, remove_self=remove_self
    )
    np.testing.assert_allclose(
        outputs.energy[0], local.energy[0] + reference, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        outputs.observable(AtomisticObservable.GLOBAL_ENERGY_COMPONENT).values[0],
        reference,
        rtol=1e-11,
        atol=1e-11,
    )
    evaluation = outputs.evidence[0]
    assert isinstance(evaluation, LatentEwaldEvaluation)
    assert bool(evaluation.periodic[0]) and bool(evaluation.mode_coverage_successful[0])
    assert not outputs.atom_energy_complete
    np.testing.assert_array_equal(
        np.asarray(outputs.atom_energy), np.asarray(local.atom_energy)
    )


def test_finite_forces_are_the_negative_gradient_including_the_latent_term() -> None:
    model = _maceles()
    batch = _batch()

    def energy(value: jax.Array) -> jax.Array:
        return _evaluate(model, batch, value, request=None).energy[0]

    def local(value: jax.Array) -> jax.Array:
        return _evaluate(model.mace, batch, value, request=None).energy[0]

    forces = -np.asarray(jax.grad(energy)(batch.positions)[0])
    local_forces = -np.asarray(jax.grad(local)(batch.positions)[0])
    compiled = jax.jit(energy)
    numerical = central_difference(
        lambda point: float(compiled(jnp.asarray(point)[None])), POSITIONS, 1.0e-5
    )
    _close(forces, -numerical, 1e-7)
    # The latent energy moves the forces, through positions and latent charges.
    assert float(np.max(np.abs(forces - local_forces))) > 1e-6


def test_periodic_forces_and_stress_match_finite_differences() -> None:
    model = _maceles(capacity=4, periodic=True)
    batch = _periodic_batch()
    execution = _periodic_execution()
    payload = _payload(batch, execution)
    derivatives = atomistic_energy_derivatives(
        model, payload, batch.positions, compute_stress=True
    )
    assert bool(derivatives.successful[0])
    assert derivatives.forces is not None and derivatives.stress is not None
    energy = energy_oracle(model, batch, execution)

    zero = jnp.zeros((3, 3), dtype=jnp.float64)
    numerical = central_difference(
        lambda point: float(energy(jnp.asarray(point)[None], zero)),
        PERIODIC_POSITIONS,
        1.0e-5,
    )
    numerical[3] = 0.0
    _close(derivatives.forces[0], -numerical, 1e-7)
    step = 1.0e-5
    volume = abs(float(np.linalg.det(CELL)))
    stress = np.asarray(derivatives.stress[0])
    for a in range(3):
        for b in range(a, 3):
            direction = np.zeros((3, 3))
            direction[a, b] = direction[b, a] = 1.0
            difference = (
                float(energy(batch.positions, jnp.asarray(step * direction)))
                - float(energy(batch.positions, jnp.asarray(-step * direction)))
            ) / (2.0 * step)
            expected = volume * stress[a, b] * (1.0 if a == b else 2.0)
            np.testing.assert_allclose(
                difference, expected, rtol=1e-6, atol=1e-6 * max(1.0, abs(expected))
            )


def test_periodic_hvp_training_traces_fixed_definition_data_and_matches_differences() -> (
    None
):
    model = _maceles(capacity=4, periodic=True)
    batch = _periodic_batch()
    execution = _periodic_execution()
    payload = _payload(batch, execution)
    cells = batch.cells
    assert cells is not None
    probe = jnp.asarray(
        [[[0.2, -0.3, 0.1], [-0.1, 0.4, -0.2], [0.05, 0.15, -0.25], [0.0, 0.0, 0.0]]],
        dtype=jnp.float64,
    )
    request = AtomisticObservableRequest(
        (AtomisticObservable.HESSIAN_ACTION,),
        derivatives=(ATOMISTIC_HESSIAN_REQUEST, ATOMISTIC_HVP_TRAINING_REQUEST),
    )

    @eqx.filter_jit
    def objective(
        parameters: Any,
        model_state: Any,
        fixed: Any,
        graph: AtomisticGraphPayload,
        vectors: Array,
        direction: Array,
    ) -> tuple[Array, tuple[Array, Array]]:
        operator = bind_atomistic_hessian(
            combine_parameters(parameters, model_state, fixed),
            graph,
            graph.positions,
            cell_vectors=vectors,
            request=request,
            policy=RegularityPolicy(allow_almost_everywhere=True),
        )
        action = operator.mv(direction)
        return jnp.sum(action**2), (action, operator.action_valid)

    parameters, model_state, fixed = partition_parameters(model)
    (value, (action, valid)), gradient = eqx.filter_value_and_grad(
        objective, has_aux=True
    )(parameters, model_state, fixed, payload, cells, probe)
    assert bool(jnp.all(valid)) and bool(tree_allfinite(gradient))
    assert bool(jnp.isfinite(value))

    def forces(positions: Array) -> Array:
        displaced = deformed_batch(batch, positions, np.eye(3, dtype=np.float64))
        result = atomistic_energy_derivatives(
            model, _payload(displaced, execution), displaced.positions
        )
        assert bool(result.successful[0]) and result.forces is not None
        return result.forces

    step = 1e-5
    reference = -(
        forces(batch.positions + step * probe) - forces(batch.positions - step * probe)
    ) / (2.0 * step)
    assert float(jnp.max(jnp.abs(reference))) > 1e-6
    np.testing.assert_allclose(action, reference, rtol=2e-5, atol=2e-7)
    direction = jax.tree.map(jnp.ones_like, parameters)
    direction = tree_scale(1.0 / tree_norm(direction), direction)
    plus, (_, plus_valid) = objective(
        tree_add_scaled(parameters, direction, step),
        model_state,
        fixed,
        payload,
        cells,
        probe,
    )
    minus, (_, minus_valid) = objective(
        tree_add_scaled(parameters, direction, -step),
        model_state,
        fixed,
        payload,
        cells,
        probe,
    )
    assert bool(jnp.all(plus_valid)) and bool(jnp.all(minus_valid))
    numerical = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(
        tree_inner(gradient, direction), numerical, rtol=2e-5, atol=2e-7
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_born_charges_match_the_source_polarization_finite_difference(
    periodic: bool,
) -> None:
    arguments = {"remove_mean": True, "epsilon_factor": 2.5, "compute_bec": True}
    if periodic:
        model = _maceles(arguments=arguments, capacity=4, periodic=True)
        batch = _periodic_batch()
        execution = _periodic_execution()
        active = PERIODIC_MASK
        reference_positions = PERIODIC_POSITIONS
        cell: np.ndarray | None = CELL
    else:
        model = _maceles(arguments=arguments)
        batch = _batch()
        execution = EXECUTION
        active = np.ones((5,), dtype=np.bool_)
        reference_positions = POSITIONS
        cell = None
    layout = AtomisticGraphLayout.for_batch(batch, execution, cutoff=CUTOFF)
    payload = layout.prepare(batch)
    outputs = _evaluate(
        model,
        batch,
        payload=payload,
        request=(AtomisticObservable.ENERGY, AtomisticObservable.BORN_CHARGES),
    )
    born = outputs.observable(AtomisticObservable.BORN_CHARGES)
    assert born.port.port_id == model.definition.born.port().port_id
    assert bool(born.successful[0])
    values = np.asarray(born.values).reshape((-1, 3, 3))
    np.testing.assert_array_equal(values[~active], 0.0)

    @jax.jit
    def latent(prepared: AtomisticGraphPayload) -> tuple[jax.Array, jax.Array]:
        observed = _evaluate(
            model,
            batch,
            prepared.positions,
            payload=prepared,
            request=(AtomisticObservable.LATENT_CHARGES,),
        ).observable(AtomisticObservable.LATENT_CHARGES)
        return observed.values, observed.successful

    def charge_function(points: np.ndarray) -> np.ndarray:
        full = np.array(reference_positions, dtype=np.float64)
        full[active] = points
        probe = layout.prepare(
            deformed_batch(batch, full[None], np.eye(3, dtype=np.float64))
        )
        charges, successful = latent(probe)
        assert bool(successful[0])
        return np.asarray(charges)[active]

    expected = reference_born_charges(
        charge_function,
        reference_positions[active],
        cell,
        remove_mean=True,
        epsilon_factor=2.5,
        step=1.0e-5,
    )
    _close(values[active], expected, 1e-6)


def test_stale_capacity_binding_refuses() -> None:
    model = _maceles(capacity=5)
    with pytest.raises(ValueError, match="prepare it again"):
        _evaluate(model, _batch(capacity=6))


def test_maceles_composition_refusals() -> None:
    definition = LatentEwaldDefinition()
    resources = LatentEwaldResources()
    with pytest.raises(ValueError, match="no latent counterpart"):
        MACELESPotential.initialize(
            _mace(readout_variant="nonlinear-bias"),
            definition,
            atom_capacity=5,
            resources=resources,
        )
    with pytest.raises(ValueError, match="scale-shifted MACE"):
        MACELESPotential.initialize(
            _mace(energy_scaling="unscaled"),
            definition,
            atom_capacity=5,
            resources=resources,
        )
    with pytest.raises(ValueError, match="no invariant readout product"):
        MACELESPotential.initialize(
            _mace(
                interactions=("real-agnostic-residual",),
                correlations=(2,),
                readout_correlation=2,
            ),
            definition,
            atom_capacity=5,
            resources=resources,
        )
    with pytest.raises(ValueError, match="one latent channel"):
        MACELESPotential.initialize(
            _mace(),
            LatentEwaldDefinition(channels=2),
            atom_capacity=5,
            resources=resources,
        )
    model = _maceles()
    other = AtomisticScaleContract(BOHR, HARTREE)
    with pytest.raises(ValueError, match="prepared in the MACE scale contract"):
        MACELESPotential(
            model.mace,
            model.latent_readouts,
            definition.prepare_graph(other, 5, resources=resources),
        )
    with pytest.raises(ValueError, match="one latent readout slot"):
        MACELESPotential(model.mace, model.latent_readouts[:1], model.latent_ewald)
    with pytest.raises(ValueError, match="parameter copy"):
        MACELESPotential(model.mace, model.latent_readouts[::-1], model.latent_ewald)
    readout = model.latent_readouts[0]
    assert isinstance(readout, MACELinearReadout)
    poisoned = eqx.tree_at(
        lambda value: value.latent_readouts[0].weight,
        model,
        readout.weight.at[0, 0].set(jnp.nan),
    )
    with pytest.raises(ValueError, match="finite"):
        poisoned.validate()


def _latent_fields(section: dict[str, Any], *names: str) -> dict[str, Any]:
    """Recipe fields of the stored latent Ewald action, or a nested dataclass."""
    node = section["model_recipe"]["fields"]["latent_ewald"]
    for name in names:
        node = node["fields"][name]
    return node["fields"]


def _leaf(section: dict[str, Any], path: str) -> str:
    inventory = model_recipe_array_inventory(
        section["model_recipe"],
        prefix="model/leaves",
        limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    )
    (name,) = [entry.name for entry in inventory if entry.path == path]
    return name


@functools.cache
def _artifact(periodic: bool) -> tuple[MACELESPotential, dict[str, Any], Any, str]:
    """One finite or periodic MACELES with its artifact section, leaves, revision."""
    model = _maceles(capacity=4 if periodic else 5, periodic=periodic)
    section, arrays, manifest = model_artifact_section(model)
    return model, section, arrays, manifest.numeric_revision.revision_id


@pytest.mark.parametrize("periodic", [False, True], ids=["finite", "periodic"])
def test_artifact_restores_the_latent_binding_with_the_same_response(
    periodic: bool,
) -> None:
    model, section, arrays, revision = _artifact(periodic)
    restored = model_artifact_from_section(
        section, arrays, numeric_revision_id=revision
    ).model
    assert type(restored) is MACELESPotential
    latent, expected_latent = restored.latent_ewald, model.latent_ewald
    assert latent.prepared_id == expected_latent.prepared_id
    assert latent.resources.resources_id == expected_latent.resources.resources_id
    assert (latent.modes is None) == (not periodic)
    if latent.modes is not None and expected_latent.modes is not None:
        assert (
            latent.modes.resources.evidence_id
            == expected_latent.modes.resources.evidence_id
        )
    batch = _periodic_batch() if periodic else _batch()
    payload = _payload(batch, _periodic_execution() if periodic else EXECUTION)
    expected = atomistic_energy_derivatives(
        model, payload, batch.positions, compute_stress=periodic
    )
    observed = atomistic_energy_derivatives(
        restored, payload, batch.positions, compute_stress=periodic
    )
    assert bool(expected.successful[0])
    assert bool(observed.successful[0])
    _close(observed.energy, expected.energy, 1e-12)
    _close(observed.forces, expected.forces, 1e-12)
    if periodic:
        _close(observed.stress, expected.stress, 1e-12)


def _width(section: dict[str, Any], leaves: dict[str, np.ndarray]) -> None:
    name = _leaf(section, ".latent_ewald.definition.sigma")
    leaves[name] = np.asarray(1.5, dtype=leaves[name].dtype)


def _born(section: dict[str, Any], leaves: dict[str, np.ndarray]) -> None:
    _latent_fields(section, "definition", "born")["epsilon_factor"]["value"] = 2.0


def _unit_factor(section: dict[str, Any], leaves: dict[str, np.ndarray]) -> None:
    _latent_fields(section)["length_factor"]["value"] = 2.0


def _capacity(section: dict[str, Any], leaves: dict[str, np.ndarray]) -> None:
    _latent_fields(section)["atom_capacity"]["value"] += 1


def _budget(section: dict[str, Any], leaves: dict[str, np.ndarray]) -> None:
    _latent_fields(section, "resources", "budget")["maximum_modes"]["value"] -= 1


def _modes(section: dict[str, Any], leaves: dict[str, np.ndarray]) -> None:
    # Value-only: one stored wavevector component moves by one lattice index.
    name = _leaf(section, ".latent_ewald.modes.modes")
    shifted = leaves[name].copy()
    shifted[0, 0] += 1
    leaves[name] = shifted


_CORRUPTIONS: dict[str, Callable[[dict[str, Any], dict[str, np.ndarray]], None]] = {
    "width": _width,
    "born": _born,
    "unit-factor": _unit_factor,
    "capacity": _capacity,
    "budget": _budget,
    "modes": _modes,
}


@pytest.mark.parametrize(
    ("corruption", "periodic"),
    [
        pytest.param(name, periodic, id=f"{name}-{kind}")
        for name in _CORRUPTIONS
        for periodic, kind in ((False, "finite"), (True, "periodic"))
        if periodic or name != "modes"
    ],
)
def test_artifact_refuses_stale_latent_structure(corruption: str, periodic: bool) -> None:
    _, section, arrays, _ = _artifact(periodic)
    changed = copy.deepcopy(section)
    leaves = {name: np.array(value) for name, value in arrays.items()}
    _CORRUPTIONS[corruption](changed, leaves)
    with pytest.raises(AtomisticModelArtifactError):
        model_artifact_from_section(changed, leaves)


@pytest.mark.parametrize("periodic", [False, True])
def test_derivative_contract_and_capabilities_are_declared(periodic: bool) -> None:
    model = _maceles(capacity=4 if periodic else 5, periodic=periodic)
    assert model.derivative_contract is not None
    assert (model.latent_ewald.modes is not None) == periodic
    assert not model.capabilities.local_energy
    assert model.capabilities.cell_derivative
    assert AtomisticObservable.GLOBAL_ENERGY_COMPONENT in model.supported_observables
    assert AtomisticObservable.CHARGES not in model.supported_observables
    assert model.architecture_id != model.mace.architecture_id
    wider = _maceles(arguments={"sigma": 1.5})
    assert wider.architecture_id != _maceles().architecture_id
