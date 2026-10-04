#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Accelerated MACE fragment coupling against the native O(3) tensor product.

The accelerated route streams the shared prepared relation's fragments
through `PreparedStreamedRelation.evaluate_fragments`, with each fragment's
seeded receiver aggregation executed by the Pallas kernels and its local
radial weights and harmonics generated inside the fragment. The oracle is the
native ``O3TensorProduct`` evaluated per edge over the whole graph, masked and
summed with ``jax.ops.segment_sum``: ordinary JAX with its own derivatives.
The kernels run through the CPU Mosaic GPU interpreter, which verifies kernel
semantics, ownership and transformation rules; it is not GPU qualification.
"""

import math
from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.backends.atomistic import pallas_atomistic_availability
from phydrax.nn.atomistic._mace_kernel_derivatives import MACECouplingDerivativeError
from phydrax.nn.atomistic._mace_kernels import (
    fragment_extent,
    MACEAcceleratedCoupling,
    MACEEdgeCouplingSpec,
    MACEKernelPlan,
    require_fragment_routing,
)
from phydrax.nn.operator.layers import (
    O3TensorProduct,
    O3TensorProductPath,
    O3TensorProductPlan,
)
from phydrax.nn.operator.representations import O3IrrepBlock, O3IrrepLayout
from phydrax.sparse import (
    EdgeRelation,
    KeyGroupAccumulation,
    PreparedStreamedRelation,
    RelationAccumulation,
    StreamedFragment,
    StreamedPayloadSpec,
    StreamedRelationPlan,
)
from phydrax.special import RealCartesianHarmonics


pytestmark = pytest.mark.strict_jax

CHANNELS = 2
NODES = 7
SOURCE = O3IrrepLayout(
    (
        O3IrrepBlock("s0", 0, 1, multiplicity=CHANNELS),
        O3IrrepBlock("s1", 1, -1, multiplicity=CHANNELS),
    )
)
HARMONICS = O3IrrepLayout(
    (O3IrrepBlock("y0", 0, 1), O3IrrepBlock("y1", 1, -1), O3IrrepBlock("y2", 2, 1))
)
# Two paths share block m2; m4 is degree three; m5 carries a negative path scale.
MESSAGE = O3IrrepLayout(
    (
        O3IrrepBlock("m0", 0, 1, multiplicity=CHANNELS),
        O3IrrepBlock("m1", 1, -1, multiplicity=CHANNELS),
        O3IrrepBlock("m2", 0, 1, multiplicity=CHANNELS),
        O3IrrepBlock("m3", 1, 1, multiplicity=CHANNELS),
        O3IrrepBlock("m4", 3, -1, multiplicity=CHANNELS),
        O3IrrepBlock("m5", 2, 1, multiplicity=CHANNELS),
    )
)


def _path(left: str, right: str, output: str, scale: float = 1.0) -> O3TensorProductPath:
    return O3TensorProductPath(
        left, right, output, connection_mode="uvu", weighted=True, path_scale=scale
    )


PLAN = O3TensorProductPlan(
    SOURCE,
    HARMONICS,
    MESSAGE,
    paths=(
        _path("s0", "y0", "m0"),
        _path("s0", "y1", "m1"),
        _path("s1", "y1", "m2", 0.7),
        _path("s0", "y0", "m2"),
        _path("s1", "y1", "m3"),
        _path("s1", "y2", "m4"),
        _path("s0", "y2", "m5", -1.3),
    ),
)

# Receiver 0 has degree 12 (split across three four-lane fragments); receivers
# 5 and 6 are empty; routes 3 and 4 duplicate one endpoint pair; route 5 is
# structurally invalid with an out-of-range endpoint; route 9 is inactive at
# runtime.
SENDERS = np.array(
    [1, 2, 3, 4, 4, 9, 6, 0, 1, 2, 3, 5, 0, 2, 1, 3, 6, 0, 4, 5, 2, 3, 6], dtype=np.int32
)
RECEIVERS = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 4, 4, 1, 3, 4, 2], dtype=np.int32
)
VALID = np.ones(SENDERS.shape, dtype=np.bool_)
VALID[5] = False
ACTIVE = np.ones(SENDERS.shape, dtype=np.bool_)
ACTIVE[9] = False
EDGES = SENDERS.shape[0]


def _coupling(
    precision: str = "float64",
    accumulation: RelationAccumulation = "deterministic",
    **overrides: Any,
) -> MACEAcceleratedCoupling:
    values: dict[str, Any] = {
        "target": "cpu_interpret",
        "precision": precision,
        "receiver_tile": 2,
        "edge_tile": 4,
        "channel_tile": 128,
        "reduction_programs": 3,
        "fragment_budget_bytes": 1 << 24,
        "accumulation": accumulation,
    }
    values.update(overrides)
    return MACEAcceleratedCoupling(MACEKernelPlan(**values))


def _relation(
    accumulation: RelationAccumulation = "deterministic",
    *,
    senders: np.ndarray = SENDERS,
    receivers: np.ndarray = RECEIVERS,
    valid: np.ndarray = VALID,
    sources: int = NODES,
    targets: int = NODES,
    receiver_tile: int = 2,
    edge_tile: int = 4,
) -> PreparedStreamedRelation:
    return StreamedRelationPlan(
        receiver_tile=receiver_tile, edge_tile=edge_tile, accumulation=accumulation
    ).prepare(
        EdgeRelation(
            senders, receivers, source_size=sources, target_size=targets, valid=valid
        ),
        owner_id="mace-kernel-test-graph",
    )


type _Lanes = Callable[[Any, Any], tuple[Array, Array]]
"""``(lane parameters, one fragment's edge-row PyTree) -> (harmonics, radial)``."""


def _stream(
    coupling: MACEAcceleratedCoupling,
    relation: PreparedStreamedRelation,
    lanes: _Lanes,
    epilogue: Callable[[Any, Array], Array],
    output: jax.ShapeDtypeStruct,
    parameters: tuple[O3TensorProduct, Any],
    source: Array,
    edges: Any,
    active: np.ndarray | None = ACTIVE,
) -> Array:
    """Receiver outputs of the fragment-streamed accelerated coupling.

    ``lanes`` turns one fragment's lane rows into its local ``(harmonics,
    radial)``; ``epilogue`` runs once per receiver on its complete aggregate.
    """
    dtype = coupling.plan.dtype

    def aggregate(
        theta: tuple[O3TensorProduct, Any],
        sources: Array,
        receivers: tuple[()],
        rows: Any,
        fragment: StreamedFragment,
        seed: KeyGroupAccumulation,
    ) -> KeyGroupAccumulation:
        del receivers
        harmonics, radial = lanes(theta[1], rows)
        return coupling.fragment(
            theta[0], fragment, sources[fragment.source_ids], harmonics, radial, seed
        )

    def finish(
        theta: tuple[O3TensorProduct, Any], receiver: tuple[()], value: Array
    ) -> Array:
        del receiver
        return epilogue(theta[1], value)

    result = relation.evaluate_fragments(
        StreamedPayloadSpec(jax.ShapeDtypeStruct((MESSAGE.packed_size,), dtype), output),
        aggregate,
        finish,
        parameters,
        source,
        (),
        edges,
        edge_active=None if active is None else jnp.asarray(active),
    )
    return result.receiver_outputs


def _given_lanes(theta: Any, rows: tuple[Array, Array]) -> tuple[Array, Array]:
    del theta
    return rows


def _identity(theta: Any, value: Array) -> Array:
    del theta
    return value


def _accelerated_aggregate(
    coupling: MACEAcceleratedCoupling,
    relation: PreparedStreamedRelation,
    tensor_product: O3TensorProduct,
    source: Array,
    harmonics: Array,
    radial: Array,
    active: np.ndarray | None = ACTIVE,
) -> Array:
    return _stream(
        coupling,
        relation,
        _given_lanes,
        _identity,
        jax.ShapeDtypeStruct((MESSAGE.packed_size,), coupling.plan.dtype),
        (tensor_product, ()),
        source,
        (harmonics, radial),
        active,
    )


def _reference(
    tensor_product: O3TensorProduct,
    source: Array,
    harmonics: Array,
    radial: Array,
    *,
    senders: np.ndarray = SENDERS,
    receivers: np.ndarray = RECEIVERS,
    used: np.ndarray = VALID & ACTIVE,
    valid: np.ndarray = VALID,
    targets: int = NODES,
) -> Array:
    message = tensor_product(source[jnp.where(valid, senders, 0)], harmonics, radial)
    message = jnp.where(jnp.asarray(used)[:, None], message, 0.0)
    return jax.ops.segment_sum(
        message, jnp.where(valid, receivers, 0), num_segments=targets
    )


def _operands(
    dtype: type[np.floating], sources: int = NODES, edges: int = EDGES
) -> tuple[O3TensorProduct, Array, Array, Array]:
    rng = np.random.default_rng(11)
    tensor_product = O3TensorProduct(PLAN, internal_weights=False, dtype=dtype)
    source = jnp.asarray(rng.normal(size=(sources, SOURCE.packed_size)), dtype=dtype)
    harmonics = jnp.asarray(rng.normal(size=(edges, HARMONICS.packed_size)), dtype=dtype)
    radial = jnp.asarray(rng.normal(size=(edges, PLAN.parameter_count)), dtype=dtype)
    return tensor_product, source, harmonics, radial


@pytest.mark.parametrize(
    ("precision", "accumulation", "tolerance"),
    [
        pytest.param("float64", "deterministic", 1e-12, id="float64-deterministic"),
        pytest.param("float64", "compensated", 1e-12, id="float64-compensated"),
        pytest.param("float32", "deterministic", 2e-5, id="float32-deterministic"),
        pytest.param("float32", "compensated", 2e-5, id="float32-compensated"),
    ],
)
def test_fragment_aggregate_matches_native_tensor_product_sum(
    precision: str, accumulation: RelationAccumulation, tolerance: float
) -> None:
    dtype = np.float64 if precision == "float64" else np.float32
    tensor_product, source, harmonics, radial = _operands(dtype)
    relation = _relation(accumulation)
    aggregate = _accelerated_aggregate(
        _coupling(precision, accumulation),
        relation,
        tensor_product,
        source,
        harmonics,
        radial,
    )
    expected = _reference(tensor_product, source, harmonics, radial)
    assert aggregate.dtype == dtype
    np.testing.assert_allclose(aggregate, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(aggregate[5:], 0.0)
    # The degree-12 receiver really was split across fragments with seeds.
    assert int(relation.schedule.fragmented_receivers) >= 1


def test_runtime_lane_mask_changes_values_without_new_signature() -> None:
    tensor_product, source, harmonics, radial = _operands(np.float64)
    coupling = _coupling()
    relation = _relation()
    spec = MACEEdgeCouplingSpec(tensor_product)
    run = partial(_accelerated_aggregate, coupling, relation, tensor_product, source)
    all_active = run(harmonics, radial, None)
    masked = run(harmonics, radial, ACTIVE)
    assert not np.allclose(all_active[0], masked[0])
    np.testing.assert_allclose(all_active[1:], masked[1:], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        run(harmonics, 2.0 * radial, ACTIVE), 2.0 * masked, rtol=1e-13
    )
    extent = fragment_extent(relation.fragments())
    signature = coupling.executable_signature(spec, extent, "graph").signature_id
    assert signature == coupling.executable_signature(spec, extent, "graph").signature_id
    retiled = _coupling(edge_tile=8)
    assert retiled.executable_signature(spec, extent, "graph").signature_id != signature
    wider = fragment_extent(_relation(edge_tile=8).fragments())
    assert coupling.executable_signature(spec, wider, "graph").signature_id != signature


def _scan_fragments(
    coupling: MACEAcceleratedCoupling,
    tensor_product: O3TensorProduct,
    fragments: StreamedFragment,
    radial: np.ndarray,
) -> KeyGroupAccumulation:
    """Chain one receiver's single-lane fragments, carrying the seed between them."""
    seed = KeyGroupAccumulation(
        jnp.zeros((1, 1), np.float64), jnp.zeros((1, 1), np.float64)
    )
    for tile in range(fragments.lane_slots.shape[0]):
        fragment = jax.tree.map(lambda value, tile=tile: value[tile], fragments)
        seed = coupling.fragment(
            tensor_product,
            fragment,
            jnp.ones((1, 1), np.float64),
            jnp.ones((1, 1), np.float64),
            jnp.asarray(radial[tile : tile + 1], np.float64),
            seed,
        )
    return seed


@pytest.mark.parametrize("accumulation", ["deterministic", "compensated"])
def test_seeded_cancellation_is_carried_across_fragments(
    accumulation: RelationAccumulation,
) -> None:
    scalar = O3TensorProductPlan(
        O3IrrepLayout((O3IrrepBlock("s0", 0, 1, multiplicity=1),)),
        O3IrrepLayout((O3IrrepBlock("y0", 0, 1),)),
        O3IrrepLayout((O3IrrepBlock("m0", 0, 1, multiplicity=1),)),
        paths=(_path("s0", "y0", "m0"),),
    )
    tensor_product = O3TensorProduct(scalar, internal_weights=False, dtype=np.float64)
    spec = MACEEdgeCouplingSpec(tensor_product)
    coefficient = float(spec.coefficients(tensor_product)[0])
    # One receiver, three lanes, one lane per fragment: the middle event is
    # below the large events' rounding unit and cancels in plain summation.
    radial = np.array([[1e17], [3.0], [-1e17]])
    relation = _relation(
        accumulation,
        senders=np.zeros(3, np.int32),
        receivers=np.zeros(3, np.int32),
        valid=np.ones(3, bool),
        sources=1,
        targets=1,
        receiver_tile=1,
        edge_tile=1,
    )
    fragments = relation.fragments()
    assert fragments.lane_slots.shape == (3, 1)
    result = _scan_fragments(
        _coupling(accumulation=accumulation), tensor_product, fragments, radial
    )
    events = [float(value) * coefficient for value in radial[:, 0]]
    high, correction = 0.0, 0.0
    for event in events:
        if accumulation == "compensated":
            total = high + event
            virtual = total - high
            correction += (high - (total - virtual)) + (event - virtual)
            high = total
        else:
            high += event
    np.testing.assert_array_equal(result.high, [[high]])
    np.testing.assert_array_equal(result.correction, [[correction]])
    value = float(result.value[0, 0])
    if accumulation == "compensated":
        assert value == math.fsum(events)
    else:
        assert value != math.fsum(events)


def test_rectangular_relation_with_independent_extents() -> None:
    senders = np.array([4, 0, 3, 3, 1, 2, 4, 0, 1], dtype=np.int32)
    receivers = np.array([2, 0, 0, 0, 1, 2, 2, 1, 0], dtype=np.int32)
    valid = np.ones(senders.shape, dtype=np.bool_)
    used = valid.copy()
    used[6] = False
    tensor_product, source, harmonics, radial = _operands(np.float64, 5, senders.shape[0])
    relation = _relation(
        senders=senders,
        receivers=receivers,
        valid=valid,
        sources=5,
        targets=3,
        receiver_tile=2,
        edge_tile=3,
    )
    extent = fragment_extent(relation.fragments())
    assert (extent.receivers, extent.edges) == (2, 3)
    aggregate = _accelerated_aggregate(
        _coupling(), relation, tensor_product, source, harmonics, radial, used
    )
    expected = _reference(
        tensor_product,
        source,
        harmonics,
        radial,
        senders=senders,
        receivers=receivers,
        used=used,
        valid=valid,
        targets=3,
    )
    assert aggregate.shape == (3, MESSAGE.packed_size)
    np.testing.assert_allclose(aggregate, expected, rtol=1e-12, atol=1e-12)


def test_operand_cotangents_match_native_reverse_mode() -> None:
    tensor_product, source, harmonics, radial = _operands(np.float64)
    coupling = _coupling(accumulation="compensated")
    relation = _relation("compensated")
    weights = jnp.asarray(
        np.random.default_rng(3).normal(size=(NODES, MESSAGE.packed_size))
    )

    def accelerated(values: tuple[O3TensorProduct, Array, Array, Array]) -> Array:
        aggregate = _accelerated_aggregate(coupling, relation, *values)
        return jnp.sum(weights * jnp.tanh(aggregate))

    def reference(values: tuple[O3TensorProduct, Array, Array, Array]) -> Array:
        return jnp.sum(weights * jnp.tanh(_reference(*values)))

    arguments = (tensor_product, source, harmonics, radial)
    got = eqx.filter_grad(accelerated)(arguments)
    want = eqx.filter_grad(reference)(arguments)
    for observed, expected in zip(got[1:], want[1:], strict=True):
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    # Coefficient cotangents exist on the executed coefficient support; zero
    # table entries are structure, not kernel operands.
    for observed, expected, table in zip(
        got[0].prepared.coefficients,
        want[0].prepared.coefficients,
        tensor_product.prepared.coefficients,
        strict=True,
    ):
        structural = np.asarray(table) != 0.0
        np.testing.assert_allclose(
            np.asarray(observed)[structural],
            np.asarray(expected)[structural],
            rtol=1e-12,
            atol=1e-12,
        )


def _imported_with_residual() -> O3TensorProduct:
    """Native tables plus one admitted residual where the native coupling is zero."""
    native = O3TensorProduct(PLAN, internal_weights=False)
    tables = [np.asarray(table).copy() for table in native.prepared.coefficients]
    assert tables[2][0, 0, 1] == 0.0
    tables[2][0, 0, 1] = 1e-7
    return O3TensorProduct(
        PLAN, internal_weights=False, coefficients=tables, coefficient_tolerance=1e-6
    )


def test_imported_coefficient_residuals_execute_like_native() -> None:
    _, source, harmonics, radial = _operands(np.float64)
    imported = _imported_with_residual()
    native_spec = MACEEdgeCouplingSpec(O3TensorProduct(PLAN, internal_weights=False))
    spec = MACEEdgeCouplingSpec(imported)
    assert spec.spec_id != native_spec.spec_id
    coupling, relation = _coupling(), _relation()
    aggregate = _accelerated_aggregate(
        coupling, relation, imported, source, harmonics, radial
    )
    expected = _reference(imported, source, harmonics, radial)
    np.testing.assert_allclose(aggregate, expected, rtol=1e-12, atol=1e-12)
    # Only the residual drives this lane input; native gives it exactly.
    residual_only = (
        jnp.zeros_like(source).at[:, 2].set(1.0),
        jnp.zeros_like(harmonics).at[:, 2].set(1.0),
        jnp.zeros_like(radial).at[:, 4].set(1.0),
    )
    np.testing.assert_allclose(
        _accelerated_aggregate(coupling, relation, imported, *residual_only),
        _reference(imported, *residual_only),
        rtol=1e-12,
        atol=0.0,
    )
    # A table changed after preparation to leave the recorded support is
    # refused, never truncated to the stale structure.
    tables = list(imported.prepared.coefficients)
    tables[2] = tables[2].at[0, 1, 0].set(1e-3)
    stale = eqx.tree_at(
        lambda value: value.prepared.coefficients, imported, tuple(tables)
    )
    with pytest.raises((eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError)):
        jax.block_until_ready(
            _accelerated_aggregate(coupling, relation, stale, source, harmonics, radial)
        )


def test_mixed_slot_tangents_share_one_compensated_accumulator() -> None:
    """Every active slot's lane events stream into one compensated accumulator
    in the receiver, source and coefficient roles; slot subtotals never collapse.

    Four scalar lanes carry per-lane mixed-slot tangent events
    ``[1e17, -1e17, 3, 0]``: the first slot's tangent on lane 1 and the radial
    tangent on lanes 0 and 2.
    """
    tensor_product = O3TensorProduct(PLAN, internal_weights=False)
    coupling = _coupling(accumulation="compensated")
    lanes = np.ones(4, dtype=np.bool_)
    exact = math.fsum([1e17, -1e17, 3.0, 0.0])
    harmonics = jnp.zeros((4, HARMONICS.packed_size)).at[:, 0].set(1.0)
    harmonic_tangent = jnp.zeros_like(harmonics).at[1, 0].set(1.0)
    radial = jnp.zeros((4, PLAN.parameter_count)).at[1, 0].set(-1e17)
    radial_tangent = jnp.zeros_like(radial).at[0, 0].set(1e17).at[2, 0].set(3.0)
    # Receiver role: four sources into one receiver; source and radial slots.
    fan_in = _relation(
        "compensated",
        senders=np.arange(4, dtype=np.int32),
        receivers=np.zeros(4, np.int32),
        valid=lanes,
        sources=4,
        targets=1,
    )
    source = jnp.zeros((4, SOURCE.packed_size)).at[:, 0].set(1.0)
    source_tangent = jnp.zeros_like(source).at[1, 0].set(1.0)
    _, tangent = jax.jvp(
        lambda values, weights: _accelerated_aggregate(
            coupling, fan_in, tensor_product, values, harmonics, weights, lanes
        ),
        (source, radial),
        (source_tangent, radial_tangent),
    )
    assert float(tangent[0, 0]) == exact
    # Source and coefficient roles: one source feeding four lanes; harmonic
    # and radial slots of their cotangent kernels.
    fan_out = _relation(
        "compensated",
        senders=np.zeros(4, np.int32),
        receivers=np.zeros(4, np.int32),
        valid=lanes,
        sources=1,
        targets=1,
    )
    single = source[:1]

    def aggregate(product: O3TensorProduct, values: Array, y: Array, w: Array) -> Array:
        return _accelerated_aggregate(coupling, fan_out, product, values, y, w, lanes)[
            0, 0
        ]

    def source_cotangent(y: Array, w: Array) -> Array:
        return jax.grad(aggregate, argnums=1)(tensor_product, single, y, w)

    def coefficient_cotangent(y: Array, w: Array) -> Array:
        product = eqx.filter_grad(aggregate)(tensor_product, single, y, w)
        return product.prepared.coefficients[0]

    for cotangent in (source_cotangent, coefficient_cotangent):
        _, tangent = jax.jvp(
            cotangent, (harmonics, radial), (harmonic_tangent, radial_tangent)
        )
        assert float(jnp.ravel(tangent)[0]) == exact


def test_reverse_over_mixed_forward_tangent_matches_native() -> None:
    """Seeded tangent chains transpose: the seed takes the pair cotangent."""
    tensor_product, source, harmonics, radial = _operands(np.float64)
    coupling = _coupling(accumulation="compensated")
    relation = _relation("compensated")
    rng = np.random.default_rng(12)
    directions = (
        jnp.asarray(rng.normal(size=source.shape)),
        jnp.asarray(rng.normal(size=radial.shape)),
    )

    def accelerated(values: Array, weights: Array) -> Array:
        return _accelerated_aggregate(
            coupling, relation, tensor_product, values, harmonics, weights
        )

    def reference(values: Array, weights: Array) -> Array:
        return _reference(tensor_product, values, harmonics, weights)

    results = []
    for function in (accelerated, reference):

        def loss(values: Array, weights: Array, function: Any = function) -> Array:
            tangent = jax.jvp(function, (values, weights), directions)[1]
            return jnp.sum(jnp.tanh(tangent))

        results.append(jax.grad(loss, argnums=(0, 1))(source, radial))
    for observed, expected in zip(*results, strict=True):
        np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-11)


# Smooth cluster energy: per-lane radial weights and harmonics are generated
# from lane displacements inside each fragment (accelerated) or over the whole
# graph (reference); the receiver readout runs once per complete aggregate.


def _displacements(
    coordinates: Array,
    strain: Array,
    senders: np.ndarray,
    receivers: np.ndarray,
    valid: np.ndarray,
) -> Array:
    deformed = coordinates @ (jnp.eye(3) + strain).T
    safe = jnp.asarray(valid)
    vector = (
        deformed[jnp.where(safe, receivers, 0)] - deformed[jnp.where(safe, senders, 0)]
    )
    # Invalid routes keep an admitted direction before harmonic evaluation.
    return jnp.where(safe[:, None], vector, jnp.ones_like(vector))


def _lane_geometry(projection: Array, vector: Array) -> tuple[Array, Array]:
    distance = jnp.sqrt(jnp.sum(vector * vector, axis=-1))
    envelope = jnp.where(distance < 6.0, (1.0 - distance / 6.0) ** 3, 0.0)
    basis = jnp.sin(jnp.arange(1.0, 5.0)[None, :] * distance[:, None]) / distance[:, None]
    radial = (basis * envelope[:, None]) @ projection
    harmonics = RealCartesianHarmonics(2, normalization="fully_normalized")(vector)
    return harmonics, radial


def _readout(readout: Array, aggregate: Array) -> Array:
    return jnp.tanh(aggregate) @ readout


def _source_features(coordinates: Array) -> Array:
    return jnp.tile(
        jnp.cos(coordinates @ jnp.ones((3,)))[:, None], (1, SOURCE.packed_size)
    )


type _Energy = Callable[[Array, Array, tuple[Array, Array]], Array]


def _accelerated_energy(
    coupling: MACEAcceleratedCoupling,
    relation: PreparedStreamedRelation,
    senders: np.ndarray = SENDERS,
    receivers: np.ndarray = RECEIVERS,
    valid: np.ndarray = VALID,
    active: np.ndarray = ACTIVE,
) -> _Energy:
    tensor_product = O3TensorProduct(PLAN, internal_weights=False, dtype=np.float64)

    def energy(
        coordinates: Array, strain: Array, parameters: tuple[Array, Array]
    ) -> Array:
        projection, readout = parameters
        outputs = _stream(
            coupling,
            relation,
            lambda theta, rows: _lane_geometry(theta[0], rows),
            lambda theta, value: _readout(theta[1], value),
            jax.ShapeDtypeStruct((), np.float64),
            (tensor_product, (projection, readout)),
            _source_features(coordinates),
            _displacements(coordinates, strain, senders, receivers, valid),
            active,
        )
        return jnp.sum(outputs)

    return energy


def _reference_energy(
    senders: np.ndarray = SENDERS,
    receivers: np.ndarray = RECEIVERS,
    valid: np.ndarray = VALID,
    active: np.ndarray = ACTIVE,
) -> _Energy:
    tensor_product = O3TensorProduct(PLAN, internal_weights=False, dtype=np.float64)

    def energy(
        coordinates: Array, strain: Array, parameters: tuple[Array, Array]
    ) -> Array:
        projection, readout = parameters
        vector = _displacements(coordinates, strain, senders, receivers, valid)
        harmonics, radial = _lane_geometry(projection, vector)
        aggregate = _reference(
            tensor_product,
            _source_features(coordinates),
            harmonics,
            radial,
            senders=senders,
            receivers=receivers,
            used=valid & active,
            valid=valid,
            targets=int(coordinates.shape[0]),
        )
        return jnp.sum(_readout(readout, aggregate))

    return energy


def _positions(nodes: int = NODES) -> Array:
    return jnp.asarray(np.random.default_rng(5).normal(size=(nodes, 3)) * 1.7)


def _parameters() -> tuple[Array, Array]:
    projection = np.random.default_rng(6).normal(size=(4, PLAN.parameter_count)) * 0.5
    readout = np.random.default_rng(7).normal(size=(MESSAGE.packed_size,))
    return jnp.asarray(projection), jnp.asarray(readout)


def _force_stress_loss(
    energy: _Energy, positions: Array
) -> Callable[[tuple[Array, Array]], Array]:
    strain = jnp.zeros((3, 3))
    target_forces = jnp.ones_like(positions)

    def loss(parameters: tuple[Array, Array]) -> Array:
        forces = -jax.grad(energy, argnums=0)(positions, strain, parameters)
        stress = jax.grad(energy, argnums=1)(positions, strain, parameters)
        return jnp.sum((forces - target_forces) ** 2) + jnp.sum(stress**2)

    return loss


def _energies() -> tuple[_Energy, _Energy]:
    return _accelerated_energy(_coupling(), _relation()), _reference_energy()


def test_force_and_stress_loss_parameter_gradients_match_native() -> None:
    positions = _positions()
    results = [
        jax.grad(_force_stress_loss(energy, positions))(_parameters())
        for energy in _energies()
    ]
    for observed, expected in zip(results[0], results[1], strict=True):
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)


def test_coordinate_hessian_vector_product_matches_native() -> None:
    positions, strain, parameters = _positions(), jnp.zeros((3, 3)), _parameters()
    direction = jnp.asarray(np.random.default_rng(8).normal(size=(NODES, 3)))
    results = []
    for energy in _energies():

        def gradient(coordinates: Array, energy: _Energy = energy) -> Array:
            return jax.grad(energy, argnums=0)(coordinates, strain, parameters)

        results.append(jax.jvp(gradient, (positions,), (direction,))[1])
    np.testing.assert_allclose(results[0], results[1], rtol=1e-10, atol=1e-10)


def test_batched_forward_jacobian_matches_native() -> None:
    tensor_product, source, harmonics, radial = _operands(np.float64)
    coupling = _coupling()
    relation = _relation()
    got = jax.jacfwd(
        lambda values: _accelerated_aggregate(
            coupling, relation, tensor_product, values, harmonics, radial
        )[0]
    )(source)
    want = jax.jacfwd(
        lambda values: _reference(tensor_product, values, harmonics, radial)[0]
    )(source)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_third_derivative_order_is_refused() -> None:
    tensor_product, source, harmonics, radial = _operands(np.float64)
    coupling = _coupling()
    relation = _relation()

    def energy(values: Array) -> Array:
        return jnp.sum(
            jnp.tanh(
                _accelerated_aggregate(
                    coupling, relation, tensor_product, values, harmonics, radial
                )
            )
        )

    def hvp(values: Array) -> Array:
        return jax.jvp(jax.grad(energy), (values,), (values,))[1]

    hvp(source)
    with pytest.raises(MACECouplingDerivativeError, match="through order 2"):
        jax.jvp(hvp, (source,), (source,))


def test_unadmitted_structures_operands_and_budgets_are_refused() -> None:
    uvw = O3TensorProductPlan(
        SOURCE,
        HARMONICS,
        MESSAGE,
        paths=(O3TensorProductPath("s0", "y0", "m0", connection_mode="uvw"),),
    )
    with pytest.raises(ValueError, match="'uvu'"):
        MACEEdgeCouplingSpec(O3TensorProduct(uvw, internal_weights=False))
    mixed = O3IrrepLayout(
        (
            O3IrrepBlock("s0", 0, 1, multiplicity=2),
            O3IrrepBlock("s1", 1, -1, multiplicity=3),
        )
    )
    with pytest.raises(ValueError, match="multiplicity"):
        MACEEdgeCouplingSpec(
            O3TensorProduct(
                O3TensorProductPlan(
                    mixed,
                    HARMONICS,
                    O3IrrepLayout((O3IrrepBlock("m0", 0, 1, multiplicity=2),)),
                    paths=(_path("s0", "y0", "m0"),),
                ),
                internal_weights=False,
            )
        )
    tensor_product, source, harmonics, radial = _operands(np.float64)
    relation = _relation()
    with pytest.raises(TypeError, match="precision float32"):
        _accelerated_aggregate(
            _coupling(precision="float32"),
            relation,
            tensor_product,
            source,
            harmonics,
            radial,
        )
    with pytest.raises(ValueError, match="one stream has one order"):
        _accelerated_aggregate(
            _coupling(accumulation="compensated"),
            relation,
            tensor_product,
            source,
            harmonics,
            radial,
        )
    with pytest.raises(ValueError, match="fragment budget"):
        _accelerated_aggregate(
            _coupling(fragment_budget_bytes=4096),
            relation,
            tensor_product,
            source,
            harmonics,
            radial,
        )


def test_malformed_fragment_routing_is_refused() -> None:
    fragments = _relation().fragments(jnp.asarray(ACTIVE))
    checked = require_fragment_routing(fragments)
    np.testing.assert_array_equal(checked.lane_slots, fragments.lane_slots)
    overlong = eqx.tree_at(
        lambda value: value.receiver_lane_counts,
        fragments,
        fragments.receiver_lane_counts.at[0, 0].add(1),
    )
    padded_lane = eqx.tree_at(
        lambda value: value.lane_use, fragments, fragments.lane_use.at[-1, -1].set(True)
    )
    for corrupted in (overlong, padded_lane):
        with pytest.raises((eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError)):
            jax.block_until_ready(require_fragment_routing(corrupted).lane_slots)


def test_cuda_target_refuses_without_an_admitted_device() -> None:
    if pallas_atomistic_availability("cuda").available:
        pytest.skip("An admitted CUDA device is present; refusal is not observable.")
    with pytest.raises(RuntimeError, match="pallas-atomistic"):
        _coupling(precision="float32", target="cuda")
