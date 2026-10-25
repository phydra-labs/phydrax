"""Behavioral tests of the native MACE family expansion.

Unit references transcribe the pinned mace-torch 0.3.16 equations
(``mace/modules/{blocks,embeddings,gate}.py``) independently of the native
implementation; model tests check the conservative, symmetry, derivative,
observable, acceleration and prepared-route contracts of every new family.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._jax_config import default_device
from phydrax._model._ports import ValuePort
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    AtomicStructure,
    atomistic_energy_derivatives,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticGraphLayout,
    AtomisticGraphPayload,
    AtomisticScaleContract,
    AtomisticSpeciesKind,
)
from phydrax.atomistic._conditioning import (
    AtomisticConditionField,
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticConditionKind,
    AtomisticConditionLevel,
    AtomisticConditionStatus,
    AtomisticObservable,
    AtomisticObservableRequest,
    TOTAL_CHARGE_PORT,
)
from phydrax.atomistic._latent_electrostatics import (
    LatentEwaldDefinition,
    LatentEwaldResources,
)
from phydrax.atomistic._model_artifact import (
    read_atomistic_model_artifact,
    write_atomistic_model_artifact,
)
from phydrax.atomistic._model_inputs import AtomisticGraphInputs
from phydrax.discretization import ParticleImageCapacity
from phydrax.nn.atomistic._mace import (
    mace_gate,
    mace_node_feature_port,
    mace_radial_widths,
    mace_tensor_product_plan,
    MACEArchitecture,
    MACELayerContext,
    MACEPotential,
)
from phydrax.nn.atomistic._mace_embedding import (
    MACEEmbeddingFeature,
    MACEJointEmbedding,
)
from phydrax.nn.atomistic._mace_interaction import (
    gate_layout,
    mace_irrep_layout,
    MACEAttentionInteraction,
    MACEGate,
    MACENonlinearInteraction,
    MACEParityConvention,
    MACEPlainLinear,
)
from phydrax.nn.atomistic._mace_kernels import MACEAcceleratedCoupling, MACEKernelPlan
from phydrax.nn.atomistic._mace_les import MACELESPotential
from phydrax.nn.atomistic._mace_prepare import (
    prepare_mace_potential,
    StaleMACEPreparation,
    UnsupportedMACEPreparedRoute,
)
from phydrax.nn.atomistic._mace_readout import (
    covalent_radii_angstrom,
    MACENonlinearBiasReadout,
)
from phydrax.nn.atomistic._radial import (
    ChebyshevRadialBasis,
    GaussianRadialBasis,
    normalized_sigmoid_scale,
    normalized_silu_scale,
    SoftTransform,
)
from phydrax.nn.atomistic._radial_projection import RadialTableDeclaration
from phydrax.nn.operator.representations import O3IrrepBlock, O3IrrepLayout
from phydrax.sparse import StreamedRelationPlan
from phydrax.units import ANGSTROM, CHARGE, ELECTRONVOLT, ELEMENTARY_CHARGE
from tests._support.atomistic_long_range_reference import central_difference
from tests._support.atomistic_model_families import energy_oracle


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
CUTOFF = 3.0
EXECUTION = AtomisticGraphExecutionPlan(8, maximum_dense_atoms=6)
NUMBERS = np.asarray([8, 1, 1, 8, 1], dtype=np.int32)
# Five non-coplanar atoms with distinct pair distances: a chiral configuration,
# so a reflection is not a rotation of it.
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
NONLINEAR = ("real-agnostic-residual-nonlinear",) * 2
ATTENTION = ("real-agnostic-attention-residual",) * 2

FEATURES = (
    MACEEmbeddingFeature(
        "total_charge", "categorical", level="case", width=3, classes=5, offset=2
    ),
    MACEEmbeddingFeature(
        "total_spin", "categorical", level="case", width=2, classes=3, offset=-1
    ),
    MACEEmbeddingFeature("site_hint", "continuous", level="atom", width=2, input_width=2),
    MACEEmbeddingFeature("temperature", "continuous", level="case", width=2, bias=False),
)

BASE: dict[str, Any] = {
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

FAMILIES: dict[str, dict[str, Any]] = {
    "attention": {"interactions": ATTENTION},
    "nonlinear": {"interactions": NONLINEAR},
    "nonlinear-edge-irreps": {
        "interactions": NONLINEAR,
        "edge_channel_count": 3,
        "edge_feature_degree": 1,
        "edge_irreps_first": True,
    },
    "so3": {"parity": "so3", "interactions": ("real-agnostic", "real-agnostic-residual")},
    "edge-irreps": {"edge_channel_count": 3, "edge_feature_degree": 1},
    "edge-irreps-first": {
        "edge_channel_count": 3,
        "edge_feature_degree": 1,
        "edge_irreps_first": True,
    },
    "keep-last-layer-irreps": {"keep_last_layer_irreps": True},
    "nonlinear-bias-readout": {"readout_variant": "nonlinear-bias"},
    "gaussian-soft": {
        "radial_basis": "gaussian",
        "distance_transform": "soft",
        "soft_alpha": 4.0,
    },
    "chebyshev": {"radial_basis": "chebyshev"},
    "joint-embedding": {"embedding_features": FEATURES, "embedding_readout": True},
}
O3_FAMILIES = [name for name, value in FAMILIES.items() if value.get("parity") != "so3"]
ACCELERATED_FAMILIES = [
    "attention",
    "nonlinear",
    "nonlinear-edge-irreps",
    "so3",
    "edge-irreps-first",
    "joint-embedding",
]


def _architecture(family: str | None = None, **overrides: Any) -> MACEArchitecture:
    declaration = dict(BASE)
    if family is not None:
        declaration.update(FAMILIES[family])
    declaration.update(overrides)
    return MACEArchitecture(**declaration)


def _model(
    family: str | None = None,
    *,
    streaming: StreamedRelationPlan | None = None,
    key: int = 5,
    **overrides: Any,
) -> MACEPotential:
    architecture = _architecture(family, **overrides)
    heads = len(architecture.heads)
    scaled = architecture.energy_scaling == "scale-shift"
    model = MACEPotential(
        SCALE,
        architecture,
        atomic_energies=np.asarray([[-1.0, -2.0]] * heads, dtype=np.float64),
        energy_scale=np.asarray([1.3] * heads, dtype=np.float64) if scaled else None,
        energy_shift=np.asarray([0.2] * heads, dtype=np.float64) if scaled else None,
        streaming=streaming,
        key=jr.key(key),
    )
    nonlinear = [
        index
        for index, layer in enumerate(model.layers)
        if isinstance(layer.interaction, MACENonlinearInteraction)
    ]
    if nonlinear:
        # The source initializes beta = 0, which hides the density path; a
        # trained-like beta exercises the whole receiver normalization.
        model = eqx.tree_at(
            lambda value: [value.layers[i].interaction.density_beta for i in nonlinear],
            model,
            [jnp.asarray(0.25, dtype=jnp.float64)] * len(nonlinear),
        )
    return model


def _structure(numbers: Any = NUMBERS, positions: Any = POSITIONS) -> AtomicStructure:
    values = np.asarray(numbers, dtype=np.int32)
    masses = np.where(values == 8, 15.999, 1.008).astype(np.float64)
    return AtomicStructure(values, np.asarray(positions, dtype=np.float64), masses, SCALE)


def _batch(
    numbers: Any = NUMBERS, positions: Any = POSITIONS, *, capacity: int | None = None
) -> AtomisticBatch:
    return AtomisticBatch.from_structures(
        (_structure(numbers, positions),), atom_capacity=capacity
    )


def _periodic_batch() -> AtomisticBatch:
    # Three atoms and one masked slot in a compact triclinic cell.
    return AtomisticBatch(
        np.asarray([[1, 8, 1, 0]], dtype=np.int32),
        np.asarray(
            [[[0.0, 0.0, 0.0], [1.1, 0.3, 0.2], [0.4, 1.6, 1.3], [0.0, 0.0, 0.0]]]
        ),
        np.asarray([[1.008, 15.999, 1.008, 1.008]]),
        SCALE,
        atom_mask=np.asarray([[True, True, True, False]]),
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


def _condition_values(
    batch: AtomisticBatch, *, charge: Any = 0, spin: Any = 2
) -> dict[str, np.ndarray]:
    cases = batch.case_count
    numbers = np.asarray(batch.atomic_numbers, dtype=np.float64).reshape((-1,))
    return {
        "total_charge": np.array(
            np.broadcast_to(np.asarray(charge, dtype=np.int64), (cases,))
        ),
        "total_spin": np.array(
            np.broadcast_to(np.asarray(spin, dtype=np.int64), (cases,))
        ),
        "site_hint": np.stack([0.1 * numbers, 0.3 - 0.05 * numbers], axis=-1),
        "temperature": np.full((cases,), 0.3, dtype=np.float64),
    }


def _conditioning(
    model: Any, batch: AtomisticBatch, *, charge: Any = 0, spin: Any = 2
) -> AtomisticConditioning | None:
    if not model.configuration.embedding_features:
        return None
    return AtomisticConditioning(
        model.conditioning,
        _condition_values(batch, charge=charge, spin=spin),
        case_count=batch.case_count,
        atom_capacity=batch.atom_capacity,
    )


def _payload(batch: AtomisticBatch, execution: Any) -> AtomisticGraphPayload:
    return AtomisticGraphLayout.for_batch(batch, execution, cutoff=CUTOFF).prepare(batch)


def _inputs(
    model: Any,
    batch: AtomisticBatch,
    positions: Any = None,
    *,
    payload: AtomisticGraphPayload | None = None,
    conditioning: AtomisticConditioning | None = None,
) -> AtomisticGraphInputs:
    bound = _payload(batch, EXECUTION) if payload is None else payload
    graph = bound.graph(cutoff=CUTOFF, positions=positions)
    return AtomisticGraphInputs(
        graph,
        batch.scale,
        species_kind=model.capabilities.species_kind,
        conditioning=_conditioning(model, batch)
        if conditioning is None
        else conditioning,
    )


def _evaluate(
    model: Any,
    batch: AtomisticBatch,
    positions: Any = None,
    *,
    request: Any = None,
    payload: AtomisticGraphPayload | None = None,
    conditioning: AtomisticConditioning | None = None,
) -> Any:
    return model.evaluate_graph(
        _inputs(model, batch, positions, payload=payload, conditioning=conditioning),
        None if request is None else AtomisticObservableRequest(request),
    )


def _energy_forces(
    model: Any, numbers: Any = NUMBERS, positions: Any = POSITIONS
) -> tuple[float, np.ndarray]:
    batch = _batch(numbers, positions)

    def energy(value: jax.Array) -> jax.Array:
        return _evaluate(model, batch, value).energy[0]

    value, gradient = jax.value_and_grad(energy)(batch.positions)
    return float(value), -np.asarray(gradient[0])


def _close(observed: Any, expected: Any, relative: float) -> None:
    """Elementwise agreement within ``relative`` times the reference magnitude."""
    observed_ = np.asarray(observed, dtype=np.float64)
    expected_ = np.asarray(expected, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(expected_))))
    np.testing.assert_allclose(observed_, expected_, rtol=0.0, atol=relative * scale)


def _rotation(seed: int) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --- source equations of the new building blocks ------------------------------


def test_plain_linear_matches_e3nn_linear_with_zero_unmatched_outputs() -> None:
    inputs = O3IrrepLayout(
        (
            O3IrrepBlock("a", 0, 1, multiplicity=2),
            O3IrrepBlock("v", 1, -1, multiplicity=2),
            O3IrrepBlock("b", 0, 1, multiplicity=3),
        )
    )
    outputs = O3IrrepLayout(
        (
            O3IrrepBlock("s", 0, 1, multiplicity=2),
            O3IrrepBlock("w", 1, -1, multiplicity=4),
            O3IrrepBlock("t", 2, 1, multiplicity=2),
            O3IrrepBlock("p", 1, 1, multiplicity=1),
        )
    )
    rng = np.random.default_rng(3)
    scalar_weight = rng.normal(size=(2, 5))
    vector_weight = rng.normal(size=(4, 2))
    plain = MACEPlainLinear(
        inputs, outputs, weights=(scalar_weight, vector_weight), dtype=jnp.float64
    )
    assert plain.matched == (0, 1)
    x = rng.normal(size=(4, inputs.packed_size))
    a = x[:, :2]
    v = x[:, 2:8].reshape((4, 2, 3))
    b = x[:, 8:11]
    # e3nn Linear: each output sums its matching input paths over the whole
    # fan-in; outputs without any matching input (2e, and 1e against 1o) are 0.
    s = (a @ scalar_weight[:, :2].T + b @ scalar_weight[:, 2:].T) / np.sqrt(5.0)
    w = np.einsum("of,nfd->nod", vector_weight, v) / np.sqrt(2.0)
    expected = np.concatenate(
        [s, w.reshape((4, 12)), np.zeros((4, 10)), np.zeros((4, 3))], axis=-1
    )
    observed = np.asarray(plain(jnp.asarray(x)))

    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-14)
    blocks = outputs.split(observed)
    np.testing.assert_array_equal(np.asarray(blocks[2]), 0.0)
    np.testing.assert_array_equal(np.asarray(blocks[3]), 0.0)
    with pytest.raises(ValueError, match="at least one matching block"):
        MACEPlainLinear(
            inputs,
            O3IrrepLayout((O3IrrepBlock("q", 3, -1, multiplicity=2),)),
            dtype=jnp.float64,
            key=jr.key(0),
        )
    with pytest.raises(ValueError, match="weights or a key"):
        MACEPlainLinear(inputs, outputs, dtype=jnp.float64)


@pytest.mark.parametrize("parity", ["o3", "so3"])
def test_gate_layout_orders_merged_scalars_and_gates_then_gated_blocks(
    parity: MACEParityConvention,
) -> None:
    target = mace_irrep_layout(2, 2, parity=parity)
    layout = gate_layout(target)
    odd = -1 if parity == "o3" else 1

    # Source: sort(scalars + gates + gated) with the 0e groups merged.
    assert [(b.degree, b.parity, b.multiplicity) for b in layout.blocks] == [
        (0, 1, 6),
        (1, odd, 2),
        (2, 1, 2),
    ]
    with pytest.raises(ValueError, match="even scalar block"):
        gate_layout(O3IrrepLayout((O3IrrepBlock("v", 1, -1, multiplicity=2),)))


@pytest.mark.parametrize("parity", ["o3", "so3"])
def test_gate_matches_source_gated_equivariant_block(
    parity: MACEParityConvention,
) -> None:
    target = mace_irrep_layout(2, 2, parity=parity)
    gate = MACEGate(target, scalar_scale=1.3, gate_scale=1.7)
    assert gate.input_layout.layout_id == gate_layout(target).layout_id
    x = np.random.default_rng(4).normal(size=(3, gate.input_layout.packed_size))
    # GatedEquivariantBlock.forward in mul_ir layout.
    scalars = 1.3 * _silu(x[:, :2])
    gates = 1.7 * _sigmoid(x[:, 2:6])
    vectors = x[:, 6:12].reshape((3, 2, 3)) * gates[:, 0:2, None]
    tensors = x[:, 12:22].reshape((3, 2, 5)) * gates[:, 2:4, None]
    expected = np.concatenate(
        [scalars, vectors.reshape((3, 6)), tensors.reshape((3, 10))], axis=-1
    )

    np.testing.assert_allclose(gate(jnp.asarray(x)), expected, rtol=1e-13, atol=1e-14)
    with pytest.raises(ValueError):
        MACEGate(target, scalar_scale=0.0, gate_scale=1.0)


def _source_nonlinear_bias(
    weights: tuple[np.ndarray, ...], scale: float, heads: int, x: np.ndarray
) -> np.ndarray:
    """NonLinearBiasReadoutBlock with e3nn fan-in normalization and mask_head."""
    first, middle, middle_bias, second, bias = weights
    hidden = second.shape[0]
    width = hidden // heads
    columns = []
    for head in range(heads):
        mask = np.ones((hidden,))
        if heads > 1:
            mask = np.zeros((hidden,))
            mask[head * width : (head + 1) * width] = 1.0
        value = scale * _silu(x @ first / np.sqrt(first.shape[0])) * mask
        value = scale * _silu(value @ middle / np.sqrt(hidden) + middle_bias) * mask
        columns.append((value @ second / np.sqrt(hidden) + bias)[:, head])
    return np.stack(columns, axis=-1)


@pytest.mark.parametrize("heads", [1, 3])
def test_nonlinear_bias_readout_matches_source_block_under_head_mask(heads: int) -> None:
    rng = np.random.default_rng(heads)
    channels, width = 4, 3
    hidden = heads * width
    weights = (
        rng.normal(size=(channels, hidden)),
        rng.normal(size=(hidden, hidden)),
        rng.normal(size=(hidden,)),
        rng.normal(size=(hidden, heads)),
        rng.normal(size=(heads,)),
    )
    first, middle, middle_bias, second, bias = (jnp.asarray(value) for value in weights)
    readout = MACENonlinearBiasReadout(
        first, middle, middle_bias, second, bias, activation_scale=1.6
    )
    x = rng.normal(size=(5, channels))
    expected = _source_nonlinear_bias(weights, 1.6, heads, x)

    for head in range(heads):
        np.testing.assert_allclose(
            readout(jnp.asarray(x), head), expected[:, head], rtol=1e-12, atol=1e-13
        )
    with pytest.raises(ValueError, match="heads \\* width per layer"):
        MACENonlinearBiasReadout(
            first,
            middle,
            jnp.zeros((hidden + 1,)),
            second,
            bias,
            activation_scale=1.6,
        )


def test_joint_embedding_matches_source_generic_joint_embedding() -> None:
    embedding = MACEJointEmbedding.initialize(
        FEATURES, 3, key=jr.key(2), dtype=jnp.float64
    )
    cases, capacity = 2, 3
    mask = np.asarray([True, True, True, True, True, False])
    atom_cases = np.repeat(np.arange(cases), capacity)
    rng = np.random.default_rng(8)
    values = {
        "total_charge": np.asarray([-1, 2], dtype=np.int64),
        "total_spin": np.asarray([1, 3], dtype=np.int64),
        "site_hint": rng.normal(size=(cases * capacity, 2)),
        "temperature": np.asarray([0.3, -0.7]),
    }
    conditioning = AtomisticConditioning(
        embedding.conditioning, values, case_count=cases, atom_capacity=capacity
    )
    parameters = [[np.asarray(leaf) for leaf in group] for group in embedding.parameters]
    # nn.Embedding rows at code + offset, case features broadcast by feat[batch].
    charge = parameters[0][0][values["total_charge"][atom_cases] + 2]
    spin = parameters[1][0][values["total_spin"][atom_cases] - 1]
    first, first_bias, second, second_bias = parameters[2]
    site = _silu(values["site_hint"] @ first + first_bias) @ second + second_bias
    temperature = values["temperature"][atom_cases][:, None]
    plain_first, plain_second = parameters[3]
    heat = _silu(temperature @ plain_first) @ plain_second
    joint = _silu(
        np.concatenate([charge, spin, site, heat], axis=-1)
        @ np.asarray(embedding.projection)
    )
    expected = np.where(mask[:, None], joint, 0.0)

    observed = embedding(conditioning, jnp.asarray(atom_cases), jnp.asarray(mask))
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-13)
    assert [field.name for field in embedding.conditioning.fields] == [
        feature.name for feature in FEATURES
    ]
    charge_field = embedding.conditioning.fields[0]
    assert charge_field.port.port_id == TOTAL_CHARGE_PORT.port_id
    assert charge_field.categories == (-2, -1, 0, 1, 2)
    assert embedding.conditioning.fields[1].categories == (1, 2, 3)


def test_joint_embedding_feature_declarations_refuse_inadmissible_specs() -> None:
    with pytest.raises(ValueError, match="physical case quantity"):
        MACEEmbeddingFeature("total_charge", "continuous", level="case", width=2)
    with pytest.raises(ValueError, match="physical case quantity"):
        MACEEmbeddingFeature(
            "total_spin", "categorical", level="atom", width=2, classes=3
        )
    with pytest.raises(ValueError, match="declares its classes"):
        MACEEmbeddingFeature("label", "categorical", level="case", width=2)
    with pytest.raises(ValueError, match="broadcasts only scalar"):
        MACEEmbeddingFeature("field", "continuous", level="case", width=2, input_width=2)
    with pytest.raises(ValueError, match="no classes or offset"):
        MACEEmbeddingFeature("field", "continuous", level="atom", width=2, classes=2)
    with pytest.raises(ValueError, match="distinct"):
        MACEJointEmbedding.initialize(
            (FEATURES[0], FEATURES[0]), 3, key=jr.key(0), dtype=jnp.float64
        )


def _spec_with(replacement: AtomisticConditionField | None, drop: str) -> Any:
    fields = [field for field in _model_spec().fields if field.name != drop]
    if replacement is not None:
        fields = [
            replacement if field.name == replacement.name else field for field in fields
        ]
    return AtomisticConditioningSpec(tuple(fields))


def _model_spec() -> AtomisticConditioningSpec:
    return _architecture("joint-embedding").conditioning


def test_conditioning_admission_refuses_missing_or_differently_declared_fields() -> None:
    model = _model("joint-embedding")
    batch = _batch()
    graph = _payload(batch, EXECUTION).graph(cutoff=CUTOFF)
    values = _condition_values(batch)

    def evaluate(spec: Any, fields: dict[str, Any]) -> Any:
        conditioning = AtomisticConditioning(
            spec, fields, case_count=1, atom_capacity=batch.atom_capacity
        )
        return model.evaluate_graph(
            AtomisticGraphInputs(
                graph,
                SCALE,
                species_kind=AtomisticSpeciesKind.ATOMIC_NUMBER,
                conditioning=conditioning,
            )
        )

    with pytest.raises(ValueError, match="needs conditioning"):
        model.evaluate_graph(
            AtomisticGraphInputs(
                graph, SCALE, species_kind=AtomisticSpeciesKind.ATOMIC_NUMBER
            )
        )
    without_spin = {name: value for name, value in values.items() if name != "total_spin"}
    with pytest.raises(ValueError, match="lacks the embedding feature 'total_spin'"):
        evaluate(_spec_with(None, "total_spin"), without_spin)
    other_port = ValuePort(
        "tests.another-total-charge",
        event_shape=(),
        component_ids=("charge",),
        representation="case-total-charge",
        dimensions=(CHARGE,),
        normalization_id="identity",
    )
    wrong_port = AtomisticConditionField(
        "total_charge",
        other_port,
        unit=ELEMENTARY_CHARGE,
        level=AtomisticConditionLevel.CASE,
        kind=AtomisticConditionKind.CATEGORICAL,
        dtype=jnp.int64,
        categories=tuple(range(-2, 3)),
    )
    with pytest.raises(ValueError, match="does not bind the declared"):
        evaluate(_spec_with(wrong_port, ""), values)
    wrong_kind = AtomisticConditionField(
        "total_charge",
        TOTAL_CHARGE_PORT,
        unit=ELEMENTARY_CHARGE,
        level=AtomisticConditionLevel.CASE,
        kind=AtomisticConditionKind.CONTINUOUS,
        dtype=jnp.float64,
    )
    continuous = dict(values, total_charge=np.asarray([0.0]))
    with pytest.raises(ValueError, match="does not bind the declared"):
        evaluate(_spec_with(wrong_kind, ""), continuous)
    wrong_categories = AtomisticConditionField(
        "total_charge",
        TOTAL_CHARGE_PORT,
        unit=ELEMENTARY_CHARGE,
        level=AtomisticConditionLevel.CASE,
        kind=AtomisticConditionKind.CATEGORICAL,
        dtype=jnp.int64,
        categories=tuple(range(-3, 3)),
    )
    with pytest.raises(ValueError, match="does not bind the declared"):
        evaluate(_spec_with(wrong_categories, ""), values)


@pytest.mark.parametrize("code", [3, -3])
def test_out_of_range_category_makes_only_its_case_unsuccessful(code: int) -> None:
    model = _model("joint-embedding")
    batch = AtomisticBatch.from_structures(
        (_structure(), _structure(positions=POSITIONS * 1.02))
    )
    conditioning = _conditioning(model, batch, charge=[0, code])
    assert conditioning is not None
    status = np.asarray(conditioning.case_status(batch.atom_mask.reshape((-1,))))
    np.testing.assert_array_equal(
        status, [0, int(AtomisticConditionStatus.INVALID_CATEGORY)]
    )
    outputs = _evaluate(model, batch, conditioning=conditioning)
    # The categorical row is never silently clamped into a plausible result:
    # the violating case is unsuccessful while its neighbor is unaffected.
    np.testing.assert_array_equal(np.asarray(outputs.successful), [True, False])
    single = _evaluate(model, _batch())
    np.testing.assert_allclose(outputs.energy[0], single.energy[0], rtol=1e-12)
    with pytest.raises(eqx.EquinoxRuntimeError, match="evaluation failed"):
        model.energy(batch, EXECUTION, conditioning=conditioning)


# --- architecture declaration -------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param(
            {"edge_channel_count": 2, "edge_feature_degree": 2},
            "above the hidden degree",
            id="edge-degree-above-hidden",
        ),
        pytest.param({"edge_channel_count": 2}, "declare both", id="edge-irreps-half"),
        pytest.param(
            {"edge_irreps_first": True}, "needs declared edge irreps", id="first-no-edge"
        ),
        pytest.param(
            {
                "interactions": ("real-agnostic-residual",),
                "correlations": (2,),
                "readout_width": None,
                "keep_last_layer_irreps": True,
            },
            "multi-interaction models",
            id="keep-last-single-interaction",
        ),
        pytest.param(
            {"gate_scales": (1.0, 1.0)},
            "belong to residual-nonlinear",
            id="gate-scales-without-nonlinear",
        ),
        pytest.param(
            {"interactions": NONLINEAR, "gate_scales": (1.0,)},
            "silu, sigmoid",
            id="gate-scales-arity",
        ),
        pytest.param(
            {"distance_transform": "soft"}, "steepness alpha", id="soft-without-alpha"
        ),
        pytest.param({"soft_alpha": 4.0}, "soft_alpha requires", id="alpha-without-soft"),
        pytest.param(
            {
                "interactions": ("real-agnostic-residual",),
                "correlations": (2,),
                "readout_width": None,
                "readout_variant": "nonlinear-bias",
            },
            "needs a nonlinear readout",
            id="bias-readout-without-nonlinear",
        ),
        pytest.param(
            {"embedding_readout": True},
            "needs a declared joint embedding",
            id="embedding-readout-without-features",
        ),
        pytest.param(
            {"embedding_features": (FEATURES[0], FEATURES[0])},
            "distinct",
            id="duplicate-embedding-features",
        ),
        pytest.param(
            {"radial_basis": "gaussian", "radial_basis_count": 1},
            "at least two functions",
            id="single-gaussian",
        ),
        pytest.param(
            {
                "species": (0, 2),
                "species_kind": AtomisticSpeciesKind.ATOM_TYPE_ID,
                "distance_transform": "soft",
                "soft_alpha": 4.0,
            },
            "covalent radii of atomic numbers",
            id="soft-on-type-ids",
        ),
    ],
)
def test_architecture_refuses_inadmissible_declarations(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _architecture(**overrides)


def test_each_new_field_enters_the_architecture_identity() -> None:
    variants: dict[str, dict[str, Any]] = {
        "base": {},
        "gaussian": {"radial_basis": "gaussian"},
        "chebyshev": {"radial_basis": "chebyshev"},
        "soft": {"distance_transform": "soft", "soft_alpha": 4.0},
        "soft-steeper": {"distance_transform": "soft", "soft_alpha": 5.0},
        "so3": {"parity": "so3"},
        "edge": {"edge_channel_count": 3, "edge_feature_degree": 1},
        "edge-wider": {"edge_channel_count": 4, "edge_feature_degree": 1},
        "edge-scalar": {"edge_channel_count": 3, "edge_feature_degree": 0},
        "edge-first": {
            "edge_channel_count": 3,
            "edge_feature_degree": 1,
            "edge_irreps_first": True,
        },
        "bias-readout": {"readout_variant": "nonlinear-bias"},
        "keep-last": {"keep_last_layer_irreps": True},
        "attention": {"interactions": ATTENTION},
        "nonlinear": {"interactions": NONLINEAR},
        "nonlinear-gates": {"interactions": NONLINEAR, "gate_scales": (1.5, 1.9)},
        "embedding": {"embedding_features": FEATURES},
        "embedding-subset": {"embedding_features": FEATURES[:2]},
        "embedding-readout": {"embedding_features": FEATURES, "embedding_readout": True},
    }
    identities = {}
    for name, overrides in variants.items():
        architecture = _architecture(**overrides)
        architecture.validate()
        identities[name] = architecture.architecture_id
    assert len(set(identities.values())) == len(identities)
    explicit = _architecture(
        radial_basis="bessel",
        distance_transform="none",
        parity="o3",
        edge_irreps_first=False,
        readout_variant="nonlinear",
        keep_last_layer_irreps=False,
        embedding_features=(),
        embedding_readout=False,
    )
    assert explicit.architecture_id == identities["base"]
    defaulted = _architecture(interactions=NONLINEAR)
    assert defaulted.gate_scales == (normalized_silu_scale(), normalized_sigmoid_scale())
    assert (
        _architecture(
            interactions=NONLINEAR, gate_scales=defaulted.gate_scales
        ).architecture_id
        == identities["nonlinear"]
    )


def test_declared_layouts_follow_parity_edge_irreps_and_last_layer() -> None:
    so3 = _architecture("so3")
    assert all(block.parity == 1 for block in so3.layout(2, 2).blocks)
    assert all(block.parity == 1 for block in so3.harmonics_layout().blocks)
    o3 = _architecture()
    assert [block.parity for block in o3.harmonics_layout().blocks] == [1, -1, 1]

    edge = _architecture("edge-irreps")
    assert edge.edge_layout(0).layout_id == edge.input_layout(0).layout_id
    assert edge.edge_layout(1).layout_id == edge.layout(3, 1).layout_id
    first = _architecture("edge-irreps-first")
    assert first.edge_layout(0).layout_id == first.layout(3, 0).layout_id
    left = mace_tensor_product_plan(first, 0).left_representation
    assert isinstance(left, O3IrrepLayout)
    assert left.layout_id == first.layout(3, 0).layout_id

    assert _architecture().output_degree(1) == 0
    keep = _architecture("keep-last-layer-irreps")
    assert keep.output_degree(1) == keep.hidden_degree
    assert keep.output_layout(1).layout_id == keep.layout(2, 1).layout_id

    nonlinear = _architecture("nonlinear")
    gate = mace_gate(nonlinear)
    assert (gate.scalar_scale, gate.gate_scale) == nonlinear.gate_scales
    assert gate.target_layout.layout_id == nonlinear.target_layout().layout_id
    for index in range(2):
        parameters = mace_tensor_product_plan(nonlinear, index).parameter_count
        assert mace_radial_widths(nonlinear, index) == (4 + 2 * 2, 4, parameters)
        attention = _architecture("attention")
        assert mace_radial_widths(attention, index) == (
            4 + 128,
            256,
            256,
            256,
            mace_tensor_product_plan(attention, index).parameter_count,
        )


# --- native models of every new family ----------------------------------------


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_constructs_validates_and_decomposes_its_energy(family: str) -> None:
    model = _model(family)
    model.validate()
    assert model.derivative_contract is not None
    outputs = _evaluate(model, _batch())
    assert bool(outputs.successful[0])
    assert outputs.atom_energy_complete
    assert np.isfinite(float(outputs.energy[0]))
    np.testing.assert_allclose(
        outputs.energy[0], jnp.sum(outputs.atom_energy[0]), rtol=1e-13, atol=1e-12
    )
    match family:
        case "attention":
            assert all(
                isinstance(layer.interaction, MACEAttentionInteraction)
                for layer in model.layers
            )
        case "nonlinear" | "nonlinear-edge-irreps":
            assert all(
                isinstance(layer.interaction, MACENonlinearInteraction)
                for layer in model.layers
            )
        case "nonlinear-bias-readout":
            assert isinstance(model.layers[-1].readout, MACENonlinearBiasReadout)
        case "gaussian-soft":
            radial = model.geometry.radial
            assert isinstance(radial.basis, GaussianRadialBasis)
            assert isinstance(radial.transform, SoftTransform)
            assert radial.transform.alpha == 4.0
            np.testing.assert_array_equal(
                np.asarray(radial.transform.covalent_radii),
                covalent_radii_angstrom((1, 8)),
            )
        case "chebyshev":
            assert isinstance(model.geometry.radial.basis, ChebyshevRadialBasis)
        case "joint-embedding":
            assert model.joint_embedding is not None
            assert model.embedding_readout is not None
            assert model.conditioning.names == tuple(f.name for f in FEATURES)
        case _:
            pass


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_is_rotation_translation_and_permutation_invariant(family: str) -> None:
    model = _model(family)
    energy, forces = _energy_forces(model)
    rotation = _rotation(7)
    moved = POSITIONS @ rotation.T + np.asarray([0.4, -0.3, 1.1])
    observed_energy, observed_forces = _energy_forces(model, positions=moved)
    _close(observed_energy, energy, 1e-11)
    _close(observed_forces, forces @ rotation.T, 1e-10)
    order = np.asarray([2, 0, 4, 1, 3], dtype=np.intp)
    permuted_energy, permuted_forces = _energy_forces(
        model, NUMBERS[order], POSITIONS[order]
    )
    _close(permuted_energy, energy, 1e-12)
    _close(permuted_forces, forces[order], 1e-11)


@pytest.mark.parametrize("family", O3_FAMILIES)
def test_o3_family_is_reflection_invariant_on_a_chiral_configuration(
    family: str,
) -> None:
    model = _model(family)
    energy, forces = _energy_forces(model)
    mirror = np.diag([1.0, -1.0, 1.0])
    reflected_energy, reflected_forces = _energy_forces(
        model, positions=POSITIONS @ mirror.T
    )
    _close(reflected_energy, energy, 1e-11)
    _close(reflected_forces, forces @ mirror.T, 1e-10)


def test_so3_convention_is_rotation_but_not_reflection_invariant() -> None:
    # The source use_so3 option labels every block even, admitting couplings
    # such as 1 x 1 -> 1 that build pseudoscalars: the energy is SO(3) but not
    # O(3) invariant, so a chiral configuration and its mirror image differ.
    model = _model("so3")
    energy, _ = _energy_forces(model)
    rotated, _ = _energy_forces(model, positions=POSITIONS @ _rotation(3).T)
    _close(rotated, energy, 1e-11)
    mirror = np.diag([1.0, -1.0, 1.0])
    reflected, _ = _energy_forces(model, positions=POSITIONS @ mirror.T)
    assert abs(reflected - energy) > 1e-7 * max(1.0, abs(energy))
    # The same model declared under O(3) is reflection invariant.
    o3 = _model("so3", parity="o3")
    o3_energy, _ = _energy_forces(o3)
    o3_reflected, _ = _energy_forces(o3, positions=POSITIONS @ mirror.T)
    _close(o3_reflected, o3_energy, 1e-11)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_forces_are_the_negative_energy_gradient(family: str) -> None:
    model = _model(family)
    batch = _batch()
    _, forces = _energy_forces(model)
    energy = jax.jit(lambda value: _evaluate(model, batch, value).energy[0])
    numerical = central_difference(
        lambda point: float(energy(jnp.asarray(point)[None])), POSITIONS, 1.0e-5
    )
    _close(forces, -numerical, 1e-7)
    assert float(np.max(np.abs(forces))) > 1e-6


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_periodic_stress_matches_the_strain_derivative(family: str) -> None:
    model = _model(family)
    batch = _periodic_batch()
    execution = _periodic_execution()
    payload = _payload(batch, execution)
    conditioning = _conditioning(model, batch)
    derivatives = atomistic_energy_derivatives(
        model,
        payload,
        batch.positions,
        compute_stress=True,
        conditioning=conditioning,
    )
    assert bool(derivatives.successful[0]) and derivatives.stress is not None
    volume = abs(float(np.linalg.det(CELL)))

    energy = energy_oracle(model, batch, execution, conditioning)

    step = 1.0e-5
    stress = np.asarray(derivatives.stress[0])
    expected = np.zeros((3, 3))
    observed = np.zeros((3, 3))
    for a in range(3):
        for b in range(a, 3):
            direction = np.zeros((3, 3))
            direction[a, b] = direction[b, a] = 1.0
            difference = (
                energy(batch.positions, step * direction)
                - energy(batch.positions, -step * direction)
            ) / (2.0 * step)
            observed[a, b] = difference
            expected[a, b] = volume * stress[a, b] * (1.0 if a == b else 2.0)
    _close(observed, expected, 1e-6)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_parameter_jvp_and_vjp_are_dual(family: str) -> None:
    model = _model(family)
    batch = _batch()
    parameters, state, fixed = partition_parameters(model)

    def energy(lane: Any) -> jax.Array:
        return _evaluate(combine_parameters(lane, state, fixed), batch).energy[0]

    rng = np.random.default_rng(21)
    tangent = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(0.01 * rng.normal(size=leaf.shape), dtype=leaf.dtype),
        parameters,
    )
    _, forward = jax.jvp(energy, (parameters,), (tangent,))
    gradient = jax.grad(energy)(parameters)
    leaves = jax.tree_util.tree_leaves(gradient)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    reverse = sum(
        float(jnp.vdot(left, right))
        for left, right in zip(leaves, jax.tree_util.tree_leaves(tangent), strict=True)
    )
    np.testing.assert_allclose(float(forward), reverse, rtol=1e-10, atol=1e-12)
    step = 1.0e-4

    def shifted(sign: float) -> float:
        lane = jax.tree_util.tree_map(
            lambda value, direction: value + sign * step * direction, parameters, tangent
        )
        return float(energy(lane))

    numerical = (shifted(1.0) - shifted(-1.0)) / (2.0 * step)
    np.testing.assert_allclose(numerical, float(forward), rtol=1e-6, atol=1e-8)


@pytest.fixture
def interpreter_on_cpu() -> Iterator[None]:
    # The "cpu_interpret" target executes only on CPU devices.
    with default_device(jax.devices("cpu")[0]):
        yield


def _coupling() -> MACEAcceleratedCoupling:
    return MACEAcceleratedCoupling(
        MACEKernelPlan(
            target="cpu_interpret",
            precision="float64",
            receiver_tile=2,
            edge_tile=4,
            channel_tile=128,
            reduction_programs=2,
            fragment_budget_bytes=1 << 26,
        )
    )


@pytest.mark.usefixtures("interpreter_on_cpu")
@pytest.mark.parametrize("family", ACCELERATED_FAMILIES)
def test_accelerated_coupling_matches_the_exact_route(family: str) -> None:
    streaming = StreamedRelationPlan(
        receiver_tile=2, edge_tile=4, channel_capacity=2048, accumulation="deterministic"
    )
    exact = _model(family, streaming=streaming)
    accelerated = exact.with_acceleration(_coupling())
    assert accelerated.architecture_id == exact.architecture_id
    assert accelerated.derivative_contract is None
    energy, forces = _energy_forces(exact)
    observed_energy, observed_forces = _energy_forces(accelerated)
    _close(observed_energy, energy, 1e-12)
    _close(observed_forces, forces, 1e-10)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_node_and_graph_embeddings_bind_their_ports_and_layouts(family: str) -> None:
    model = _model(family)
    architecture = model.configuration
    batch = _batch(capacity=6)
    outputs = _evaluate(
        model,
        batch,
        request=(
            AtomisticObservable.ENERGY,
            AtomisticObservable.NODE_EMBEDDINGS,
            AtomisticObservable.GRAPH_EMBEDDINGS,
        ),
    )
    width = sum(
        architecture.output_layout(index).packed_size
        for index in range(architecture.interaction_count)
    )
    nodes = outputs.observable(AtomisticObservable.NODE_EMBEDDINGS)
    graph = outputs.observable(AtomisticObservable.GRAPH_EMBEDDINGS)
    assert (
        nodes.port.port_id
        == mace_node_feature_port(architecture, AtomisticConditionLevel.ATOM).port_id
    )
    assert (
        graph.port.port_id
        == mace_node_feature_port(architecture, AtomisticConditionLevel.CASE).port_id
    )
    assert nodes.level is AtomisticConditionLevel.ATOM
    assert graph.level is AtomisticConditionLevel.CASE
    assert nodes.values.shape == (6, width) and graph.values.shape == (1, width)
    np.testing.assert_allclose(
        graph.values[0], jnp.sum(nodes.values[:5], axis=0), rtol=1e-12, atol=1e-12
    )
    # Padding is not a physical atom: the padded energy is the unpadded one.
    unpadded = _evaluate(model, _batch())
    np.testing.assert_allclose(outputs.energy, unpadded.energy, rtol=1e-12)
    with pytest.raises(ValueError, match="does not provide requested observables"):
        _evaluate(
            model,
            batch,
            request=(AtomisticObservable.ENERGY, AtomisticObservable.CHARGES),
        )
    with pytest.raises(ValueError, match="does not provide requested observables"):
        _evaluate(model, batch, request=(AtomisticObservable.LATENT_CHARGES,))


def test_init_features_refuses_a_missing_or_undeclared_joint_addend() -> None:
    species = jnp.zeros((3,), dtype=jnp.int32)
    mask = jnp.ones((3,), dtype=jnp.bool_)
    joint = _model("joint-embedding")
    with pytest.raises(ValueError, match="joint-embedding addend"):
        joint.init_features(species, mask, MACELayerContext(None, None))
    with pytest.raises(ValueError, match="needs conditioning"):
        joint.joint_features(None, jnp.zeros((3,), dtype=jnp.int32), mask)
    with pytest.raises(ValueError, match="embedding-readout energy is required"):
        joint.atom_energies(
            species,
            mask,
            jnp.zeros((2, 3), dtype=jnp.float64),
            jnp.zeros((3, 2), dtype=jnp.float64),
            MACELayerContext(jnp.zeros((3, 2), dtype=jnp.float64), None),
        )
    plain = _model("attention")
    with pytest.raises(ValueError, match="joint-embedding addend"):
        plain.init_features(
            species, mask, MACELayerContext(jnp.zeros((3, 2), dtype=jnp.float64), None)
        )
    assert plain.joint_features(None, jnp.zeros((3,), dtype=jnp.int32), mask) is None
    with pytest.raises(ValueError, match="supplied with a reconstruction"):
        MACEPotential(
            SCALE,
            _architecture("joint-embedding"),
            atomic_energies=np.asarray([[-1.0, -2.0]]),
            energy_scale=np.asarray([1.3]),
            energy_shift=np.asarray([0.2]),
            joint_embedding=joint.joint_embedding,
            key=jr.key(1),
        )


def test_embedding_readout_is_an_unscaled_atom_energy_term() -> None:
    model = _model("joint-embedding")
    batch = _batch()
    inputs = _inputs(model, batch)
    outputs = model.evaluate_graph(inputs)
    readout = model.embedding_readout
    assert readout is not None
    zeroed = eqx.tree_at(
        lambda value: value.embedding_readout.weight,
        model,
        jnp.zeros_like(readout.weight),
    )
    base = zeroed.evaluate_graph(_inputs(zeroed, batch))
    mask = inputs.atom_mask
    species = model.species_indices(inputs.species_ids, mask)
    joint = model.joint_features(inputs.conditioning, inputs.atom_cases, mask)
    features = np.asarray(
        model.init_features(species, mask, MACELayerContext(joint, None))
    )
    head = model.energy_reference.head
    weight = np.asarray(readout.weight)
    # Source LinearReadoutBlock: x @ W[:, head] / sqrt(C), added to E0 unscaled.
    term = features @ weight[:, head] / np.sqrt(weight.shape[0])
    np.testing.assert_allclose(
        np.asarray(outputs.atom_energy[0] - base.atom_energy[0]),
        term,
        rtol=1e-11,
        atol=1e-12,
    )
    assert outputs.atom_energy_complete
    np.testing.assert_allclose(
        outputs.energy[0], jnp.sum(outputs.atom_energy[0]), rtol=1e-13
    )


def test_joint_embedding_conditions_the_energy() -> None:
    model = _model("joint-embedding")
    batch = _batch()
    neutral = _evaluate(model, batch, conditioning=_conditioning(model, batch))
    charged = _evaluate(model, batch, conditioning=_conditioning(model, batch, charge=1))
    triplet = _evaluate(model, batch, conditioning=_conditioning(model, batch, spin=3))
    assert bool(neutral.successful[0] & charged.successful[0] & triplet.successful[0])
    assert abs(float(charged.energy[0] - neutral.energy[0])) > 1e-8
    assert abs(float(triplet.energy[0] - neutral.energy[0])) > 1e-8


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_persists_and_restores_its_function(family: str, tmp_path: Path) -> None:
    model = _model(family)
    path = tmp_path / f"{family}.phydrax"
    write_atomistic_model_artifact(path, model)
    restored = read_atomistic_model_artifact(path).model
    restored.validate()
    assert restored.architecture_id == model.architecture_id
    batch = _batch()
    np.testing.assert_allclose(
        _evaluate(restored, batch).energy, _evaluate(model, batch).energy, rtol=1e-13
    )


# --- prepared routes ----------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "overrides"),
    [
        pytest.param("attention", {}, id="attention"),
        pytest.param(
            "nonlinear", {"readout_variant": "nonlinear-bias"}, id="nonlinear-bias"
        ),
        pytest.param("joint-embedding", {}, id="joint-embedding-readout"),
    ],
)
def test_prepared_exact_route_matches_the_raw_model(
    family: str, overrides: dict[str, Any]
) -> None:
    model = _model(family, **overrides)
    prepared = prepare_mace_potential(model)
    prepared.validate()
    batch = _batch(capacity=6)
    request = (
        AtomisticObservable.ENERGY,
        AtomisticObservable.ATOM_ENERGY,
        AtomisticObservable.NODE_EMBEDDINGS,
        AtomisticObservable.GRAPH_EMBEDDINGS,
    )
    raw = _evaluate(model, batch, request=request)
    folded = _evaluate(prepared, batch, request=request)
    assert bool(folded.successful[0]) and folded.atom_energy_complete
    _close(folded.energy, raw.energy, 1e-12)
    _close(folded.atom_energy, raw.atom_energy, 1e-12)
    for observable in (
        AtomisticObservable.NODE_EMBEDDINGS,
        AtomisticObservable.GRAPH_EMBEDDINGS,
    ):
        left = folded.observable(observable)
        right = raw.observable(observable)
        assert left.port.port_id == right.port.port_id
        _close(left.values, right.values, 1e-12)

    def forces(potential: Any) -> jax.Array:
        return -jax.grad(lambda value: _evaluate(potential, batch, value).energy[0])(
            batch.positions
        )

    _close(forces(prepared), forces(model), 1e-10)


@pytest.mark.parametrize(
    ("family", "reason"),
    [
        ("attention", "tabulated-attention-edge-network"),
        ("nonlinear", "tabulated-nonlinear-edge-network"),
    ],
)
def test_tabulated_preparation_refuses_context_dependent_edge_networks(
    family: str, reason: str
) -> None:
    with pytest.raises(UnsupportedMACEPreparedRoute) as refusal:
        prepare_mace_potential(
            _model(family),
            radial="tabulated",
            tables=RadialTableDeclaration(0.3, 256, layout="projected-width"),
        )
    assert refusal.value.reason == reason


def test_prepare_refuses_the_latent_ewald_composition() -> None:
    maceles = MACELESPotential.initialize(
        _model("attention"),
        LatentEwaldDefinition(),
        atom_capacity=5,
        resources=LatentEwaldResources(),
        key=jr.key(1),
    )
    with pytest.raises(UnsupportedMACEPreparedRoute) as refusal:
        prepare_mace_potential(maceles)  # ty: ignore[invalid-argument-type]
    assert refusal.value.reason == "latent-ewald-composition"


def test_prepared_context_energy_refuses_a_conditioned_model() -> None:
    prepared = prepare_mace_potential(_model("joint-embedding"))
    # Refused before the context is read: a system context carries no typed
    # conditioning for the joint embedding.
    with pytest.raises(UnsupportedMACEPreparedRoute) as refusal:
        prepared.energy(None)
    assert refusal.value.reason == "context-energy-conditioning"


@pytest.mark.parametrize(
    ("family", "field"),
    [
        ("attention", "source_context_rows"),
        ("attention", "receiver_context_rows"),
        ("nonlinear", "receiver_context_rows"),
    ],
)
def test_tampered_prepared_context_rows_are_stale(family: str, field: str) -> None:
    prepared = prepare_mace_potential(_model(family))
    rows = getattr(prepared.layers[0], field)
    assert rows is not None
    tampered = eqx.tree_at(
        lambda value: getattr(value.layers[0], field), prepared, rows + 0.1
    )
    with pytest.raises(StaleMACEPreparation):
        tampered.validate()
