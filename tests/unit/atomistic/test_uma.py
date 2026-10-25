"""Behavioral tests of the native UMA/eSCN-MD potential (fairchem ``bb0a062``)."""

import copy
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._model._structure import pack_model_array_tree
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    AtomicStructure,
    atomistic_energy_derivatives,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticGraphInputs,
    AtomisticGraphLayout,
    AtomisticScaleContract,
)
from phydrax.atomistic._conditioning import (
    AtomisticConditionStatus,
    AtomisticObservable,
    AtomisticObservableRequest,
)
from phydrax.atomistic._model_artifact import (
    ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    AtomisticModelArtifactError,
    model_artifact_from_section,
    model_artifact_section,
)
from phydrax.atomistic._neighborhood_bundle import AtomisticNeighborhoodRole
from phydrax.atomistic._potential import atomistic_model_revision
from phydrax.discretization import ParticleImageCapacity
from phydrax.discretization.particle._image_neighborhood import (
    ImageNeighborSelectionPolicy,
)
from phydrax.nn.atomistic._uma import (
    PreparedUMAMergedModel,
    StaleUMAMergeError,
    UMAModel,
    UMANetwork,
    UMAPotential,
    UMAPropertyModel,
)
from phydrax.nn.atomistic._uma_conditioning import (
    balance_channels,
    UMAChargeSpinEmbedding,
    UMAConditionSchema,
    UMAMoLEBinding,
    UMAMoLERouter,
)
from phydrax.nn.atomistic._uma_layers import (
    gate_activation,
    UMALinear,
    UMAMixtureLinear,
    UMANormalization,
    UMASO2Convolution,
)
from phydrax.nn.atomistic._uma_so3 import (
    uma_edge_rotation,
    uma_keyed_edge_rolls,
    uma_source_basis_residual,
    uma_source_wigner,
    UMACoefficientLayout,
    UMAEdgeRollGauge,
    UMASphericalGrid,
)
from phydrax.nn.atomistic._uma_source import (
    parse_uma_source_config,
    uma_initial_source_tensors,
    uma_model_from_source,
)
from phydrax.sparse import StreamedRelationPlan
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tests._support.uma_source import uma_config


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
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


def _rotation(seed: int) -> np.ndarray:
    matrix, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return matrix * np.sign(np.linalg.det(matrix))


def _e3nn_degree_two(vector: np.ndarray) -> np.ndarray:
    """e3nn degree-2 harmonics (component normalization) in source (x, y, z)."""
    x, y, z = vector
    root15 = np.sqrt(15.0)
    return np.asarray(
        [
            root15 * x * z,
            root15 * x * y,
            np.sqrt(5.0) * (y * y - 0.5 * (x * x + z * z)),
            root15 * y * z,
            0.5 * root15 * (z * z - x * x),
        ]
    )


def _model(
    seed: int = 3,
    roll_gauge: UMAEdgeRollGauge = "two-chart-nlerp-zero-roll",
    **config: Any,
) -> UMAModel:
    model_config, tasks = uma_config(**config)
    declaration = parse_uma_source_config(model_config, tasks, roll_gauge=roll_gauge)
    tensors = uma_initial_source_tensors(declaration, key=jr.key(seed))
    potential, residual = uma_model_from_source(
        SCALE,
        declaration,
        tensors,
        streaming=StreamedRelationPlan(
            receiver_tile=2, edge_tile=4, channel_capacity=512
        ),
        node_block=2,
    )
    assert residual < 1e-6
    return potential


def _first_edge_mixture(network: UMANetwork) -> UMAMixtureLinear:
    """The first block's ``m = 0`` edgewise MoLE operator: an expert mixture."""
    operator = network.blocks[0].edgewise.first.m0
    assert isinstance(operator, UMAMixtureLinear)
    return operator


def _batch() -> AtomisticBatch:
    masses = np.where(NUMBERS == 8, 15.999, 1.008)
    return AtomisticBatch.from_structure(
        AtomicStructure(NUMBERS, POSITIONS, masses, SCALE)
    )


def _graph(
    model: Any, batch: AtomisticBatch, positions: Any, *, selected: bool = True
) -> Any:
    """The model's graph; the declared selection is bound by the graph owner."""
    payload = AtomisticGraphLayout.for_batch(
        batch, EXECUTION, cutoff=model.configuration.cutoff
    ).prepare(batch)
    return payload.graph(
        cutoff=model.configuration.cutoff,
        positions=positions,
        selection=_selection(model) if selected else None,
    )


def _selection(model: Any) -> Any:
    return model.requirements.neighborhood(
        AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
    ).selection


def _outputs(
    model: Any,
    positions: Any,
    conditioning: Any,
    request: AtomisticObservableRequest | None = None,
    batch: AtomisticBatch | None = None,
) -> Any:
    batch = _batch() if batch is None else batch
    inputs = AtomisticGraphInputs(
        _graph(model, batch, positions),
        batch.scale,
        species_kind=model.capabilities.species_kind,
        conditioning=conditioning,
    )
    return model.evaluate_graph(inputs, request)


def _condition(
    model: UMAModel, task: str = "omol", charge: int = 0, spin: int = 1
) -> Any:
    return model.schema.conditioning(
        [task], atom_capacity=5, charges=[charge], spins=[spin]
    )


# --- basis and rotations -------------------------------------------------


def test_source_wigner_is_a_representation_in_the_e3nn_basis() -> None:
    first, second = _rotation(1), _rotation(2)
    d_first = np.asarray(uma_source_wigner(jnp.asarray(first), 3))
    d_second = np.asarray(uma_source_wigner(jnp.asarray(second), 3))
    d_product = np.asarray(uma_source_wigner(jnp.asarray(first @ second), 3))
    # Degree one is the Cartesian rotation in source (x, y, z) order.
    np.testing.assert_allclose(d_first[1:4, 1:4], first, atol=1e-12)
    np.testing.assert_allclose(d_product, d_first @ d_second, atol=1e-11)
    np.testing.assert_allclose(d_first @ d_first.T, np.eye(16), atol=1e-11)
    vector = np.asarray([0.3, -0.8, 0.5])
    np.testing.assert_allclose(
        d_first[4:9, 4:9] @ _e3nn_degree_two(vector),
        _e3nn_degree_two(first @ vector),
        atol=1e-11,
    )


def test_stored_e3nn_j_evidence_identifies_the_relabelled_basis() -> None:
    rotation = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    samples = np.random.default_rng(4).normal(size=(40, 3))
    harmonics = np.stack([_e3nn_degree_two(v) for v in samples])
    rotated = np.stack([_e3nn_degree_two(rotation @ v) for v in samples])
    degree_two, *_ = np.linalg.lstsq(harmonics, rotated, rcond=None)
    blocks = [np.ones((1, 1)), rotation, degree_two.T]
    assert uma_source_basis_residual(blocks) < 1e-10
    # The same rotation written in native (y, z, x) component order is a
    # different source rotation; degree two then contradicts the evidence.
    native_order = rotation[[1, 2, 0]][:, [1, 2, 0]]
    assert uma_source_basis_residual([np.ones((1, 1)), native_order, degree_two.T]) > 0.1


def test_edge_rotation_maps_the_edge_to_the_polar_axis_smoothly() -> None:
    vectors = np.asarray(
        [[0.3, 0.2, -0.9], [0.0, -2.0, 1e-9], [1e-9, 3.0, 0.0], [0.4, -0.9, 0.1]]
    )
    rotations = np.asarray(uma_edge_rotation(jnp.asarray(vectors)))
    for vector, rotation in zip(vectors, rotations, strict=True):
        np.testing.assert_allclose(
            rotation @ (vector / np.linalg.norm(vector)), [0.0, 1.0, 0.0], atol=1e-9
        )
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) > 0.0
    jacobian = jax.jacobian(uma_edge_rotation)(jnp.asarray([0.2, 0.05, 0.4]))
    assert bool(jnp.all(jnp.isfinite(jacobian)))


def test_grid_reproduces_the_e3nn_integral_grid_and_inverts_band_limited_signals() -> (
    None
):
    grid = UMASphericalGrid(2, 2, resolution=None, dtype=jnp.float64)
    assert (grid.latitude, grid.longitude) == (6, 7)
    beta = (np.arange(6) + 0.5) / 6 * np.pi
    alpha = np.arange(7) / 7 * 2 * np.pi
    b, a = np.meshgrid(beta, alpha, indexing="ij")
    point = np.stack([np.sin(b) * np.sin(a), np.cos(b), np.sin(b) * np.cos(a)], axis=-1)
    np.testing.assert_allclose(
        np.asarray(grid.to_grid)[..., 1:4],
        np.sqrt(3.0 / (4.0 * np.pi)) * point,
        atol=1e-12,
    )
    coefficients = jnp.asarray(np.random.default_rng(5).normal(size=(9, 4)))
    np.testing.assert_allclose(
        grid.integrate(grid.project(coefficients)), coefficients, atol=1e-10
    )


# --- layer formulas -------------------------------------------------------


def _roll(theta: float) -> np.ndarray:
    return np.asarray(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )


def test_so2_convolution_and_gate_commute_with_edge_rolls() -> None:
    layout = UMACoefficientLayout(2, 1)
    rng = np.random.default_rng(6)
    m0 = UMALinear(rng.normal(size=(3 * 3 + 2 * 3, 3 * 2)), rng.normal(size=(3 * 3 + 6,)))
    m1 = UMALinear(rng.normal(size=(2 * 2 * 3, 2 * 2)))
    conv = UMASO2Convolution(layout, m0, (m1,), channels_in=2, channels_out=3, extra=6)
    rows = np.asarray(layout.edge_rows)
    full = np.asarray(uma_source_wigner(jnp.asarray(_roll(0.7)), 2))
    others = np.setdiff1d(np.arange(9), rows)
    np.testing.assert_allclose(full[rows][:, others], 0.0, atol=1e-12)
    roll = jnp.asarray(full[rows][:, rows])
    values = jnp.asarray(rng.normal(size=(layout.reduced_size, 2)))
    case = jnp.asarray(0)
    output, gating = conv(values, None, case)
    rolled, rolled_gating = conv(roll @ values, None, case)
    np.testing.assert_allclose(rolled, roll @ output, atol=1e-11)
    np.testing.assert_allclose(rolled_gating, gating, atol=1e-11)
    gated = gate_activation(gating, output, layout.m_major_degrees, 2)
    np.testing.assert_allclose(
        gate_activation(rolled_gating, rolled, layout.m_major_degrees, 2),
        roll @ gated,
        atol=1e-11,
    )


def test_rms_norm_sh_matches_the_source_formula_and_is_rotation_invariant() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(4, 9, 5))
    weight = rng.normal(size=(3, 5))
    bias = rng.normal(size=(5,))
    norm = UMANormalization("rms_norm_sh", 2, weight=weight, bias=bias)
    degrees = np.repeat(np.arange(3), 2 * np.arange(3) + 1)
    centered = values.copy()
    centered[:, 0] -= centered[:, 0].mean(axis=-1, keepdims=True)
    balance = (1.0 / (2 * degrees + 1)) / 3.0
    scale = (np.einsum("nic,i->nc", centered**2, balance).mean(axis=-1) + 1e-5) ** -0.5
    expected = centered * scale[:, None, None] * weight[degrees]
    expected[:, 0] += bias
    observed = np.asarray(norm(jnp.asarray(values)))
    np.testing.assert_allclose(observed, expected, atol=1e-12)
    rotation = np.asarray(uma_source_wigner(jnp.asarray(_rotation(8)), 2))
    np.testing.assert_allclose(
        np.asarray(norm(jnp.asarray(np.einsum("ij,njc->nic", rotation, values)))),
        np.einsum("ij,njc->nic", rotation, observed),
        atol=1e-11,
    )


def test_mixture_linear_mixes_operators_before_nonlinear_layers() -> None:
    rng = np.random.default_rng(9)
    mixture = UMAMixtureLinear(rng.normal(size=(3, 4, 5)), rng.normal(size=(4,)))
    bias = mixture.bias
    assert bias is not None
    coefficients = jnp.asarray(rng.random(size=(2, 3)))
    resolved = mixture.resolve(coefficients)
    values = jnp.asarray(rng.normal(size=(5,)))
    for case in (0, 1):
        operator = np.einsum("eoi,e->oi", mixture.weights, coefficients[case])
        np.testing.assert_allclose(
            resolved(values, jnp.asarray(case)),
            operator @ values + bias,
            atol=1e-12,
        )
    # A mixed operator followed by a nonlinearity differs from mixing the
    # nonlinear predictions of each expert (not an ensemble).
    mixed = jax.nn.silu(resolved(values, jnp.asarray(0)))
    ensemble = sum(
        coefficients[0, e] * jax.nn.silu(mixture.weights[e] @ values + bias)
        for e in range(3)
    )
    assert float(jnp.max(jnp.abs(mixed - ensemble))) > 1e-3


# --- conditioning ----------------------------------------------------------


def test_spin_index_zero_is_the_null_token_not_multiplicity_zero() -> None:
    weight = np.random.default_rng(10).normal(size=(4,))
    spin = UMAChargeSpinEmbedding("pos_emb", "spin", weight)
    charge = UMAChargeSpinEmbedding("pos_emb", "charge", weight)
    embedded = np.asarray(spin(jnp.asarray([0, 1, 3])))
    np.testing.assert_array_equal(embedded[0], 0.0)
    assert np.any(embedded[1] != 0.0)
    assert np.any(np.asarray(charge(jnp.asarray([0])))[0] != 0.0)
    linear = UMAChargeSpinEmbedding(
        "lin_emb", "spin", np.ones((8, 1)) * 0.5, bias=np.arange(8.0)
    )
    np.testing.assert_allclose(
        linear(jnp.asarray([0]))[0], 0.5 * -100.0 + np.arange(8.0), atol=1e-12
    )


def test_condition_schema_records_source_defaults_and_refuses_bad_categories() -> None:
    schema = UMAConditionSchema(("omol", "omat"))
    omol = schema.conditioning(["omol"], atom_capacity=2)
    omat = schema.conditioning(["omat"], atom_capacity=2)
    assert int(omol.value("uma_spin_index")[0]) == 1
    assert int(omat.value("uma_spin_index")[0]) == 0
    assert int(omol.value("total_charge")[0]) == 0
    assert bool(jnp.all(omol.defaults_used))
    with pytest.raises(ValueError, match="Mixed-task"):
        schema.conditioning(["omol", "omat"], atom_capacity=2)
    mask = jnp.ones((4,), dtype=jnp.bool_)
    bad = schema.conditioning(
        ["omol", "omat"], atom_capacity=2, charges=[101, 0], spins=[1, 0]
    )
    status = np.asarray(bad.case_status(mask))
    assert status[0] & int(AtomisticConditionStatus.OUT_OF_RANGE)
    assert status[1] == 0


def test_channel_balance_sets_each_case_sum_to_its_target() -> None:
    rng = np.random.default_rng(11)
    values = jnp.asarray(rng.normal(size=(6, 4, 3)))
    mask = jnp.asarray([True, True, False, True, True, True])
    balanced = balance_channels(values, jnp.asarray([2, -1]), mask, 2, 3, (0, 2), 0.0)
    sums = np.asarray(
        jnp.sum(
            jnp.where(mask[:, None], balanced[:, 0, 0:2], 0.0).reshape(2, 3, 2), axis=1
        )
    )
    np.testing.assert_allclose(sums, [[2.0, 2.0], [-1.0, -1.0]], atol=1e-12)
    np.testing.assert_array_equal(balanced[2], values[2])
    np.testing.assert_array_equal(balanced[:, :, 2], values[:, :, 2])


def test_composition_policies_and_binding_are_exact_distinct_definitions() -> None:
    rng = np.random.default_rng(12)
    linears = [
        UMALinear(rng.normal(size=(4, 6)), rng.normal(size=(4,))),
        UMALinear(rng.normal(size=(4, 4)), rng.normal(size=(4,))),
        UMALinear(rng.normal(size=(2, 4)), rng.normal(size=(2,))),
    ]
    table = rng.normal(size=(10, 3))
    water = jnp.zeros((10,), dtype=jnp.int32).at[1].set(2).at[8].set(1)[None]
    double = 2 * water
    mean = UMAMoLERouter(
        linears,
        composition=table,
        composition_policy="mean",
        normalization="softmax-plus-0.005",
    )
    seeded = UMAMoLERouter(
        linears,
        composition=table,
        composition_policy="sum-over-count-plus-one",
        normalization="softmax-plus-0.005",
    )
    np.testing.assert_allclose(
        mean.composition_embedding(water), mean.composition_embedding(double), atol=1e-12
    )
    np.testing.assert_allclose(
        seeded.composition_embedding(water),
        (2 * table[1] + table[8])[None] / 4.0,
        atol=1e-12,
    )
    assert not np.allclose(
        seeded.composition_embedding(water), seeded.composition_embedding(double)
    )
    coefficients = mean.coefficients(
        jnp.asarray(rng.normal(size=(1, 3))), water, jnp.asarray([True])
    )
    np.testing.assert_allclose(jnp.sum(coefficients), 1.0 + 2 * 0.005, atol=1e-12)
    binding = UMAMoLEBinding(
        {1: 4, 8: 2}, total_charge=0, spin_index=1, task=1, composition_policy="mean"
    )
    assert binding.composition == ((1, 2), (8, 1))
    charge, spin, task = jnp.asarray([0, 0]), jnp.asarray([1, 1]), jnp.asarray([1, 1])
    counts = jnp.concatenate([water, double])
    assert bool(jnp.all(binding.matches(counts, charge, spin, task)))
    exact = UMAMoLEBinding(
        {1: 2, 8: 1},
        total_charge=0,
        spin_index=1,
        task=1,
        composition_policy="sum-over-count-plus-one",
    )
    np.testing.assert_array_equal(
        exact.matches(counts, charge, spin, task), [True, False]
    )


# --- potential --------------------------------------------------------------


def test_requirements_declare_the_source_nearest_neighbour_rule() -> None:
    model = _model(max_neighbors=7)
    expected = ImageNeighborSelectionPolicy(
        2.5,
        7,
        degeneracy_tolerance=0.01,
        measure="squared-distance",
        strict=False,
        radius_boundary="closed",
        coincidence_threshold=0.0,
    )
    assert _selection(model).policy_id == expected.policy_id


def test_evaluation_refuses_graphs_without_the_bound_selection() -> None:
    model = _model()
    positions = jnp.asarray(POSITIONS)[None]
    inputs = AtomisticGraphInputs(
        _graph(model, _batch(), positions, selected=False),
        SCALE,
        species_kind=model.capabilities.species_kind,
        conditioning=_condition(model),
    )
    with pytest.raises(ValueError, match="nearest-neighbour"):
        model.evaluate_graph(inputs)


def test_bound_selection_keeps_the_source_nearest_neighbours_and_ties() -> None:
    model = _model(max_neighbors=1)
    positions = jnp.asarray(POSITIONS)[None]
    graph = _graph(model, _batch(), positions)
    receivers = np.asarray(graph.graph.receivers)
    active = np.asarray(graph.graph.edge_mask)
    # Independent rule: keep d^2 <= (second smallest d^2) + 0.01 (non-strict k = 1).
    squared = np.sum((POSITIONS[None, :] - POSITIONS[:, None]) ** 2, axis=-1)
    for receiver in range(len(NUMBERS)):
        eligible = np.sort(
            [
                squared[receiver, j]
                for j in range(len(NUMBERS))
                if j != receiver and squared[receiver, j] <= 2.5**2
            ]
        )
        expected = (
            len(eligible)
            if len(eligible) <= 1
            else int(np.sum(eligible <= eligible[1] + 0.01))
        )
        assert int(np.sum(active & (receivers == receiver))) == expected
    assert bool(_outputs(model, positions, _condition(model)).successful[0])


def test_energy_is_invariant_and_forces_match_finite_differences() -> None:
    model = _model(ff_type="spectral")
    condition = _condition(model)
    positions = jnp.asarray(POSITIONS)[None]

    def energy(x: Any) -> Any:
        return _outputs(model, x, condition).energy[0]

    value = energy(positions)
    assert np.isfinite(float(value))
    rotation = jnp.asarray(_rotation(13))
    np.testing.assert_allclose(
        energy(positions @ rotation.T + 0.3), value, rtol=1e-10, atol=1e-10
    )
    gradient = jax.grad(energy)(positions)
    step = 1e-5
    for atom, axis in ((0, 0), (2, 1), (4, 2)):
        shift = jnp.zeros_like(positions).at[0, atom, axis].set(step)
        estimate = (energy(positions + shift) - energy(positions - shift)) / (2 * step)
        np.testing.assert_allclose(
            gradient[0, atom, axis], estimate, rtol=1e-6, atol=1e-8
        )
    tangent = jnp.asarray(np.random.default_rng(14).normal(size=positions.shape))
    np.testing.assert_allclose(
        jax.jvp(energy, (positions,), (tangent,))[1],
        jnp.sum(gradient * tangent),
        rtol=1e-10,
    )


def test_force_loss_parameter_gradient_matches_finite_differences() -> None:
    """Mixed parameter-coordinate derivative of a force objective (training route)."""
    model = _model(ff_type="spectral")
    condition = _condition(model)
    positions = jnp.asarray(POSITIONS)[None]
    lane, state, fixed = partition_parameters(model)

    def loss(parameters: Any) -> Any:
        rebuilt = combine_parameters(parameters, state, fixed)
        forces = -jax.grad(lambda x: _outputs(rebuilt, x, condition).energy[0])(positions)
        return jnp.sum(forces * forces)

    gradient = jax.grad(loss)(lane)
    leaves, treedef = jax.tree_util.tree_flatten(lane)
    direction = [
        jnp.asarray(
            np.random.default_rng(17 + i).normal(size=leaf.shape), dtype=leaf.dtype
        )
        for i, leaf in enumerate(leaves)
    ]
    tangent = jax.tree_util.tree_unflatten(treedef, direction)
    analytic = sum(
        jnp.sum(g * d)
        for g, d in zip(jax.tree_util.tree_leaves(gradient), direction, strict=True)
    )
    step = 1e-6
    shifted = [
        jax.tree_util.tree_map(lambda p, d, s=s: p + s * step * d, lane, tangent)
        for s in (1.0, -1.0)
    ]
    estimate = (loss(shifted[0]) - loss(shifted[1])) / (2 * step)
    np.testing.assert_allclose(analytic, estimate, rtol=1e-5, atol=1e-8)


def test_mixed_task_batch_matches_separate_cases() -> None:
    model = _model(ff_type="spectral")
    numbers = np.stack([NUMBERS, NUMBERS])
    positions = np.stack(
        [POSITIONS, POSITIONS + 0.05 * np.random.default_rng(15).normal(size=(5, 3))]
    )
    masses = np.where(numbers == 8, 15.999, 1.008)
    batch = AtomisticBatch(numbers, positions, masses, SCALE)
    condition = model.schema.conditioning(
        ["omat", "omol"], atom_capacity=5, charges=[0, -1], spins=[0, 2]
    )
    joint = _outputs(model, jnp.asarray(positions), condition, batch=batch)
    assert bool(jnp.all(joint.successful))
    for case, (task, charge, spin) in enumerate((("omat", 0, 0), ("omol", -1, 2))):
        single = _outputs(
            model,
            jnp.asarray(positions[case])[None],
            _condition(model, task, charge, spin),
        )
        np.testing.assert_allclose(
            joint.energy[case], single.energy[0], rtol=1e-11, atol=1e-11
        )


def test_conservative_stress_matches_strain_finite_difference() -> None:
    model = _model(ff_type="spectral")
    cell = np.asarray([[3.3, 0.0, 0.0], [0.4, 3.5, 0.0], [0.2, 0.3, 3.7]])
    execution = AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=8,
            maximum_edges=1024,
            maximum_degree=64,
            maximum_images=125,
        ),
    )

    def batch(strain: np.ndarray) -> AtomisticBatch:
        deform = np.eye(3) + strain
        return AtomisticBatch(
            np.asarray([[1, 8, 1]], dtype=np.int32),
            np.asarray([[[0.0, 0.0, 0.0], [1.1, 0.3, 0.2], [0.4, 1.6, 1.3]]]) @ deform.T,
            np.asarray([[1.008, 15.999, 1.008]]),
            SCALE,
            cells=(cell @ deform.T)[None],
            periodic_axes=np.ones((1, 3), dtype=np.bool_),
        )

    condition = model.schema.conditioning(["omat"], atom_capacity=3)

    def derivatives(strain: np.ndarray, stress: bool) -> Any:
        current = batch(strain)
        payload = AtomisticGraphLayout.for_batch(current, execution, cutoff=2.5).prepare(
            current
        )
        return atomistic_energy_derivatives(
            model,
            payload,
            current.positions,
            compute_stress=stress,
            conditioning=condition,
        )

    reference = derivatives(np.zeros((3, 3)), True)
    stress = np.asarray(reference.stress[0])
    np.testing.assert_allclose(stress, stress.T, atol=1e-10)
    volume = abs(np.linalg.det(cell))
    step = 1e-5
    direction = np.zeros((3, 3))
    direction[0, 1] = direction[1, 0] = 0.5
    plus = float(derivatives(step * direction, False).energy[0])
    minus = float(derivatives(-step * direction, False).energy[0])
    np.testing.assert_allclose(
        (plus - minus) / (2 * step) / volume, stress[0, 1], rtol=1e-5, atol=1e-8
    )


def _binding(model: UMAModel, *, charge: int = 0) -> UMAMoLEBinding:
    return UMAMoLEBinding(
        {8: 2, 1: 3},
        total_charge=charge,
        spin_index=1,
        task=model.schema.code("omol"),
        composition_policy=model.configuration.composition_policy,
    )


def _property_model() -> UMAPropertyModel:
    config, tasks = uma_config(direct=True, ff_type="spectral")
    for name in ("omat_energy", "omol_energy"):
        config["heads"][name]["reduce"] = "mean"
    declaration = parse_uma_source_config(config, tasks)
    model, _ = uma_model_from_source(
        SCALE, declaration, uma_initial_source_tensors(declaration, key=jr.key(7))
    )
    assert isinstance(model, UMAPropertyModel)
    return model


def _first_inexact_leaf(tree: Any) -> int:
    return next(
        index
        for index, leaf in enumerate(jax.tree_util.tree_leaves(tree))
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
    )


def test_merged_mole_matches_routing_and_refuses_stale_or_unbound_use() -> None:
    model = _model()
    binding = _binding(model)
    merged = model.merged(binding)
    merged.validate()
    positions = jnp.asarray(POSITIONS)[None]
    routed = _outputs(model, positions, _condition(model))
    bound = _outputs(merged, positions, _condition(model))
    assert bool(bound.successful[0])
    np.testing.assert_allclose(bound.energy, routed.energy, rtol=1e-11, atol=1e-11)
    other = _outputs(merged, positions, _condition(model, charge=1))
    assert not bool(other.successful[0])
    assert bool(_outputs(model, positions, _condition(model, charge=1)).successful[0])
    weights = _first_edge_mixture(model.network).weights
    changed = eqx.tree_at(
        lambda m: m.network.blocks[0].edgewise.first.m0.weights, model, weights * 1.01
    )
    merged.require_current(model)
    with pytest.raises(StaleUMAMergeError):
        merged.require_current(changed)


def test_merged_mole_prepared_identity_is_separate_from_the_scientific_model() -> None:
    model = _model()
    merged = model.merged(_binding(model))
    revision = atomistic_model_revision(model)
    # Scientific identities belong to the exact raw model; the realization
    # has its own prepared identity bound to the raw revision and binding.
    assert merged.model is model
    assert merged.source_revision_id == revision.revision_id
    assert merged.revision().revision_id == revision.revision_id
    assert merged.prepared_id not in (model.architecture_id, revision.revision_id)
    assert model.merged(_binding(model)).prepared_id == merged.prepared_id
    assert model.merged(_binding(model, charge=1)).prepared_id != merged.prepared_id
    # Parameter updates change only dynamic identity words, never the structure.
    weights = _first_edge_mixture(model.network).weights
    changed = eqx.tree_at(
        lambda m: m.network.blocks[0].edgewise.first.m0.weights, model, weights * 1.01
    )
    remerged = changed.merged(_binding(changed))
    assert remerged.prepared_id != merged.prepared_id
    assert jax.tree_util.tree_structure(remerged) == jax.tree_util.tree_structure(merged)


def test_merged_mole_validate_refuses_tampered_operators_identities_and_model() -> None:
    model = _model()
    merged = model.merged(_binding(model))
    index = _first_inexact_leaf(merged.merged_blocks)
    corrupted = eqx.tree_at(
        lambda m: jax.tree_util.tree_leaves(m.merged_blocks)[index],
        merged,
        jax.tree_util.tree_leaves(merged.merged_blocks)[index] + 1.0,
    )
    with pytest.raises(StaleUMAMergeError, match="Merged MoLE operators"):
        corrupted.validate()
    weights = _first_edge_mixture(model.network).weights
    stale = eqx.tree_at(
        lambda m: m.frozen_model.value.network.blocks[0].edgewise.first.m0.weights,
        merged,
        weights * 1.01,
    )
    with pytest.raises(StaleUMAMergeError):
        stale.validate()
    with pytest.raises(StaleUMAMergeError):
        stale.revision()
    other = model.merged(_binding(model, charge=1))
    relabelled = eqx.tree_at(
        lambda m: m.prepared_identity, merged, other.prepared_identity
    )
    with pytest.raises(StaleUMAMergeError, match="merged UMA identity"):
        relabelled.validate()
    rebound = eqx.tree_at(
        lambda m: m.source_revision_identity,
        merged,
        other.prepared_identity,
    )
    with pytest.raises(StaleUMAMergeError, match="source revision"):
        rebound.validate()
    # Merged operators of another binding refuse even with that binding's identity.
    swapped = eqx.tree_at(lambda m: m.merged_blocks, merged, other.merged_blocks)
    with pytest.raises(StaleUMAMergeError, match="Merged MoLE operators"):
        swapped.validate()


def test_merged_mole_refuses_inadmissible_bindings() -> None:
    model = _model()
    with pytest.raises(ValueError, match="composition policy"):
        model.merged(
            UMAMoLEBinding(
                {8: 2, 1: 3},
                total_charge=0,
                spin_index=1,
                task=0,
                composition_policy="sum-over-count-plus-one"
                if model.configuration.composition_policy == "mean"
                else "mean",
            )
        )
    with pytest.raises(ValueError, match="task code"):
        model.merged(
            UMAMoLEBinding(
                {8: 2, 1: 3},
                total_charge=0,
                spin_index=1,
                task=len(model.configuration.tasks),
                composition_policy=model.configuration.composition_policy,
            )
        )
    with pytest.raises(ValueError, match="element table"):
        model.merged(
            UMAMoLEBinding(
                {model.configuration.max_elements: 1},
                total_charge=0,
                spin_index=1,
                task=0,
                composition_policy=model.configuration.composition_policy,
            )
        )


def _binding_recipe(section: dict[str, Any]) -> dict[str, Any]:
    """The restored binding's field recipes inside one merged model section."""
    return section["model_recipe"]["fields"]["binding"]["fields"]


@pytest.mark.parametrize("kind", ["potential", "property"])
def test_merged_mole_artifact_round_trip_preserves_outputs_and_refuses_tampering(
    kind: str,
) -> None:
    model = (
        _model(direct=True, ff_type="spectral")
        if kind == "potential"
        else _property_model()
    )
    binding = _binding(model)
    merged = model.merged(binding)
    revision = atomistic_model_revision(model).revision_id
    section, arrays, _ = model_artifact_section(merged, licenses=("test-fixture",))
    restored = model_artifact_from_section(section, arrays, numeric_revision_id=revision)
    prepared = restored.model
    assert isinstance(prepared, PreparedUMAMergedModel)
    assert restored.manifest.architecture_id == model.architecture_id
    assert restored.manifest.method_id == model.method_id
    assert restored.manifest.numeric_revision.revision_id == revision
    assert prepared.prepared_id == merged.prepared_id
    assert prepared.source_revision_id == revision
    assert prepared.binding.binding_id == binding.binding_id
    assert prepared.model_kind == kind
    positions = jnp.asarray(POSITIONS)[None]
    written = _outputs(merged, positions, _condition(model))
    read = _outputs(prepared, positions, _condition(model))
    assert written.energy_available == read.energy_available == (kind == "potential")
    if kind == "potential":
        np.testing.assert_array_equal(read.energy, written.energy)
    else:
        np.testing.assert_array_equal(
            read.observable(AtomisticObservable.REPORTED_ENERGY).values,
            written.observable(AtomisticObservable.REPORTED_ENERGY).values,
        )
    np.testing.assert_array_equal(
        read.observable(AtomisticObservable.DIRECT_FORCES).values,
        written.observable(AtomisticObservable.DIRECT_FORCES).values,
    )
    # The raw model round trips separately and re-prepares the same realization.
    raw_section, raw_arrays, _ = model_artifact_section(model)
    raw = model_artifact_from_section(
        raw_section, raw_arrays, numeric_revision_id=revision
    )
    assert raw.model.merged(binding).prepared_id == merged.prepared_id

    # A binding whose conditions changed under its recorded id.
    recharged = copy.deepcopy(section)
    _binding_recipe(recharged)["total_charge"]["value"] = 1
    # A consistently relabelled binding: its merged operators no longer match.
    relabelled = copy.deepcopy(recharged)
    _binding_recipe(relabelled)["binding_id"]["value"] = _binding(
        model, charge=1
    ).binding_id
    # A non-canonical (unsorted) composition the constructor never produces.
    unsorted = copy.deepcopy(section)
    _binding_recipe(unsorted)["composition"]["items"].reverse()
    for changed_section in (recharged, relabelled, unsorted):
        with pytest.raises(AtomisticModelArtifactError):
            model_artifact_from_section(changed_section, arrays)
    # Corrupted merged buffers and a changed raw model.
    index = _first_inexact_leaf(prepared.merged_blocks)
    corrupted = eqx.tree_at(
        lambda m: jax.tree_util.tree_leaves(m.merged_blocks)[index],
        prepared,
        jax.tree_util.tree_leaves(prepared.merged_blocks)[index] + 1.0,
    )
    weights = _first_edge_mixture(prepared.model.network).weights
    stale = eqx.tree_at(
        lambda m: m.frozen_model.value.network.blocks[0].edgewise.first.m0.weights,
        prepared,
        weights * 1.01,
    )
    for changed in (corrupted, stale):
        changed_arrays = pack_model_array_tree(
            changed,
            section["model_recipe"],
            prefix="model/leaves",
            limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS,
        )
        with pytest.raises(AtomisticModelArtifactError) as refusal:
            model_artifact_from_section(section, changed_arrays)
        assert isinstance(refusal.value.__cause__, StaleUMAMergeError)
        with pytest.raises(StaleUMAMergeError):
            model_artifact_section(changed)


def test_direct_force_head_is_a_covariant_typed_output() -> None:
    model = _model(direct=True, ff_type="spectral")
    request = AtomisticObservableRequest(
        (AtomisticObservable.ENERGY, AtomisticObservable.DIRECT_FORCES)
    )
    condition = _condition(model)
    positions = jnp.asarray(POSITIONS)[None]
    result = _outputs(model, positions, condition, request)
    direct = result.observable(AtomisticObservable.DIRECT_FORCES)
    assert direct.values.shape == (5, 3)
    assert bool(direct.available[0])
    rotation = jnp.asarray(_rotation(16))
    rotated = _outputs(model, positions @ rotation.T, condition, request)
    np.testing.assert_allclose(
        rotated.observable(AtomisticObservable.DIRECT_FORCES).values,
        direct.values @ rotation.T,
        atol=1e-9,
    )
    conservative = _model(ff_type="spectral")
    with pytest.raises(ValueError, match="does not provide"):
        _outputs(conservative, positions, _condition(conservative), request)


def test_moe_single_ties_gradients_to_one_shared_mixture() -> None:
    model = _model(moe_single=True, ff_type="spectral")
    assert len(model.network.shared_mixtures) == 2
    condition = _condition(model)
    positions = jnp.asarray(POSITIONS)[None]
    lane, state, fixed = partition_parameters(model)

    def energy(parameters: Any) -> Any:
        rebuilt = combine_parameters(parameters, state, fixed)
        return _outputs(rebuilt, positions, condition).energy[0]

    gradient = jax.grad(energy)(lane)
    shared = gradient.network.shared_mixtures[0].weights
    assert float(jnp.max(jnp.abs(shared))) > 0.0
    # References carry no parameters; the shared leaf receives every use.
    assert gradient.network.blocks[0].edgewise.first.orders[0].__class__.__name__ == (
        "UMASharedMixture"
    )


def test_mean_reduced_head_potential_is_the_differentiated_sum() -> None:
    model_config, tasks = uma_config(ff_type="spectral")
    reduced = {**model_config, "heads": {"efs": {**model_config["heads"]["efs"]}}}
    reduced["heads"]["efs"]["head_kwargs"] = {"reduce": "mean"}
    summed = parse_uma_source_config(model_config, tasks)
    meaned = parse_uma_source_config(reduced, tasks)
    assert meaned.architecture.energy_reduction == "mean"
    tensors = uma_initial_source_tensors(summed, key=jr.key(5))
    total, _ = uma_model_from_source(SCALE, summed, tensors)
    potential, _ = uma_model_from_source(SCALE, meaned, tensors)
    assert isinstance(total, UMAPotential) and isinstance(potential, UMAPotential)
    positions = jnp.asarray(POSITIONS)[None]
    references = 2 * -2041.8 + 3 * -13.6
    request = AtomisticObservableRequest(
        (AtomisticObservable.ENERGY, AtomisticObservable.REPORTED_ENERGY)
    )
    summed_energy = _outputs(total, positions, _condition(total)).energy[0]
    result = _outputs(potential, positions, _condition(potential), request)
    # The force-generating potential is rmsd * sum_i e_i (no mean/references).
    np.testing.assert_allclose(summed_energy - result.energy[0], references, rtol=1e-12)
    # The source task energy report keeps its per-atom mean and references.
    reported = result.observable(AtomisticObservable.REPORTED_ENERGY)
    np.testing.assert_allclose(
        reported.values[0], result.energy[0] / 5.0 + references, rtol=1e-12
    )
    assert potential.method_id.startswith(
        "negative-position-gradient-of-uma-unreduced-head-sum-potential"
    )
    assert total.method_id == "negative-position-gradient-of-total-uma-energy"


def test_mean_heads_with_direct_forces_form_a_property_model_without_energy() -> None:
    direct_config, direct_tasks = uma_config(direct=True, ff_type="spectral")
    for name in ("omat_energy", "omol_energy"):
        direct_config["heads"][name]["reduce"] = "mean"
    declaration = parse_uma_source_config(direct_config, direct_tasks)
    assert declaration.architecture.model_kind == "property"
    tensors = uma_initial_source_tensors(declaration, key=jr.key(7))
    model, _ = uma_model_from_source(SCALE, declaration, tensors)
    assert isinstance(model, UMAPropertyModel)
    assert AtomisticObservable.ENERGY not in model.supported_observables
    positions = jnp.asarray(POSITIONS)[None]
    with pytest.raises(ValueError, match="does not provide"):
        _outputs(
            model,
            positions,
            _condition(model),
            AtomisticObservableRequest((AtomisticObservable.ENERGY,)),
        )
    result = _outputs(model, positions, _condition(model))
    assert not result.energy_available
    reported = result.observable(AtomisticObservable.REPORTED_ENERGY)
    direct = result.observable(AtomisticObservable.DIRECT_FORCES)
    assert bool(reported.available[0]) and direct.values.shape == (5, 3)
    # The same trunk weights evaluated as a summed-head potential differ from the
    # report only through the source's per-atom mean and its references.
    summed_config, summed_tasks = uma_config(direct=True, ff_type="spectral")
    summed_declaration = parse_uma_source_config(summed_config, summed_tasks)
    potential, _ = uma_model_from_source(SCALE, summed_declaration, tensors)
    references = 2 * -2041.8 + 3 * -13.6
    energy = _outputs(potential, positions, _condition(potential)).energy[0]
    np.testing.assert_allclose(
        reported.values[0], (energy - references) / 5.0 + references, rtol=1e-12
    )
    assert model.method_id == (
        "uma-source-reported-energy-and-direct-outputs-without-potential"
    )
    binding = UMAMoLEBinding(
        {8: 2, 1: 3},
        total_charge=0,
        spin_index=1,
        task=model.schema.code("omol"),
        composition_policy=model.configuration.composition_policy,
    )
    merged = model.merged(binding)
    assert merged.model_kind == "property"
    assert not merged.capabilities.conservative_energy
    assert not hasattr(merged, "energy")
    bound = _outputs(merged, positions, _condition(model))
    assert not bound.energy_available
    np.testing.assert_allclose(
        bound.observable(AtomisticObservable.REPORTED_ENERGY).values,
        reported.values,
        rtol=1e-11,
        atol=1e-11,
    )
    with pytest.raises(ValueError, match="does not provide"):
        _outputs(
            merged,
            positions,
            _condition(model),
            AtomisticObservableRequest((AtomisticObservable.ENERGY,)),
        )


def test_s2_fixed_roll_gauge_is_a_distinct_named_realization() -> None:
    gate = _model()
    s2 = _model(act_type="s2")
    assert gate.configuration.roll_realization == "roll-invariant"
    assert gate.method_id == "negative-position-gradient-of-total-uma-energy"
    assert s2.configuration.roll_realization == "fixed-gauge-s2-quadrature-dependent"
    assert s2.method_id.endswith("edge-roll-gauge=two-chart-nlerp-zero-roll")
    vector = jnp.asarray([0.3, -0.4, 0.8])
    rolled = uma_edge_rotation(vector, roll=jnp.asarray(0.9))
    np.testing.assert_allclose(
        rolled, jnp.asarray(_roll(0.9)) @ uma_edge_rotation(vector), atol=1e-12
    )
    keyed = _model(act_type="s2", roll_gauge="keyed-route-uniform-roll")
    assert keyed.configuration.roll_realization == "keyed-route-uniform-roll-s2"
    assert keyed.method_id != s2.method_id
    assert keyed.configuration.architecture_id != s2.configuration.architecture_id
    with pytest.raises(ValueError, match="roll invariant"):
        _model(roll_gauge="keyed-route-uniform-roll")


def test_keyed_rolls_are_route_addressed_full_width_and_explicit() -> None:
    key = jr.key(21)
    senders = jnp.asarray([3, 1, 2], dtype=jnp.int64)
    receivers = jnp.asarray([0, 0, 1], dtype=jnp.int64)
    rolls = uma_keyed_edge_rolls(key, (senders, receivers), dtype=jnp.float64)
    assert bool(jnp.all((rolls >= 0.0) & (rolls < 2 * np.pi)))
    order = jnp.asarray([2, 0, 1])
    permuted = uma_keyed_edge_rolls(
        key, (senders[order], receivers[order]), dtype=jnp.float64
    )
    np.testing.assert_array_equal(permuted, rolls[order])
    high = senders + (1 << 40)
    shifted = uma_keyed_edge_rolls(key, (high, receivers), dtype=jnp.float64)
    assert not bool(jnp.any(shifted == rolls))
    other = uma_keyed_edge_rolls(jr.key(22), (senders, receivers), dtype=jnp.float64)
    assert not bool(jnp.any(other == rolls))
