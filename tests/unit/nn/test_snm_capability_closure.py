# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax._classification import (
    ordinal_log_prob_from_cumulative_logits,
    pointwise_classification_loss,
)
from phydrax._training import DelayedTargetPolicy, TargetParameterState
from phydrax.discretization.spectral._modal_discovery import (
    discover_modal_support,
    ModalSupportDiscoveryPlan,
)
from phydrax.nn.models import (
    CoercivePolyconvexEnvelope,
    ConstrainedPolyconvexPotential,
    PolyconvexMaterialConstraints,
    ReferenceConfiguration,
)
from phydrax.nn.operator.architectures import (
    ConvolutionSupportPlan,
    WaveletDecodePolicy,
)
from phydrax.nn.operator.training import load_pretrained_operator
from phydrax.nn.parameters import (
    LowRankUpdate,
    OrderedOrdinalCutpoints,
    ParameterSubspace,
)


def test_ordered_cutpoints_and_soft_ordinal_one_hot_parity():
    cutpoints = OrderedOrdinalCutpoints(4, initial=(-1.0, 0.0, 2.0))
    values = cutpoints()
    assert jnp.all(jnp.diff(values) > 0.0)
    logits = values - jnp.asarray(0.25)
    hard = -ordinal_log_prob_from_cumulative_logits(logits, jnp.asarray(2))
    soft = pointwise_classification_loss(
        logits,
        jax.nn.one_hot(2, 4),
        kind="ordinal",
        objective="soft_cross_entropy",
    )
    assert jnp.allclose(hard, soft)


def test_complex_low_rank_update_preserves_native_dtype():
    base = jnp.eye(3, dtype=jnp.complex64)
    update = LowRankUpdate(base, rank=1, key=jax.random.key(0))
    assert update.materialize().dtype == jnp.complex64
    assert jnp.allclose(update.materialize(), base)


def test_delayed_target_exact_accepted_update_lag():
    state = TargetParameterState.initialize(
        {"x": jnp.asarray(0.0)},
        DelayedTargetPolicy(1),
    )
    state = state.update({"x": jnp.asarray(1.0)})
    assert state.target["x"] == 0.0
    state = state.update({"x": jnp.asarray(2.0)})
    assert state.target["x"] == 1.0


def test_constrained_polyconvex_reference_is_fixed_during_parameter_updates():
    deformation = jnp.asarray(((1.2, 0.1), (0.0, 0.9)))
    reference = ReferenceConfiguration(deformation)
    model = ConstrainedPolyconvexPotential(
        reference,
        CoercivePolyconvexEnvelope(PolyconvexMaterialConstraints()),
    )
    trainable, fixed = eqx.partition(model, eqx.is_inexact_array)
    reference_leaves = jax.tree_util.tree_leaves(model.reference)
    envelope_leaves = jax.tree_util.tree_leaves(model.envelope)
    trainable_leaves = jax.tree_util.tree_leaves(trainable)

    assert reference_leaves == []
    assert len(trainable_leaves) == len(envelope_leaves) == 4
    assert model(deformation) == 0.0
    assert jnp.linalg.norm(model.first_piola_stress(deformation)) < 1.0e-4

    initial_shift = model.reference_energy_shift
    updated_trainable = jax.tree.map(
        lambda value: value + jnp.asarray(0.1, dtype=value.dtype),
        trainable,
    )
    updated = eqx.combine(updated_trainable, fixed)
    updated_gradient = updated.reference.deformation_gradient
    updated_report = updated.constraint_report()

    assert not jnp.array_equal(updated.reference_energy_shift, initial_shift)
    assert jnp.array_equal(updated_gradient, deformation)
    assert jnp.allclose(
        updated.reference.inverse @ updated_gradient,
        jnp.eye(2),
    )
    assert jnp.allclose(
        updated.reference.determinant,
        jnp.linalg.det(updated_gradient),
    )
    assert updated_report.reference_determinant > 0.0
    assert jnp.allclose(updated_report.reference_energy, 0.0)
    assert updated_report.reference_stress_norm < 1.0e-4


def test_static_wavelet_and_nonperiodic_support_policies():
    assert WaveletDecodePolicy().out_of_support == "error"
    support = ConvolutionSupportPlan(("dirichlet_sine", "neumann_cosine"))
    assert not support.periodic


class _Layout:
    coefficient_shape = (4,)
    layout_id = "test-layout"


def test_modal_support_has_fixed_capacity_and_stable_top_energy():
    plan = ModalSupportDiscoveryPlan(_Layout(), 2)
    support = discover_modal_support(plan, jnp.asarray((1.0, 4.0, 2.0, 0.0)))
    assert support.multi_indices.shape == (2, 1)
    assert tuple(support.multi_indices[:, 0]) == (1, 2)


def test_bundled_pretrained_weights_load_and_execute_without_io_in_predict():
    fno = load_pretrained_operator("fno-diffusion-1d")
    result = jax.jit(lambda values: fno.predict(values, 24))(jnp.ones((16, 1)))
    assert result.shape == (24, 1)
    deeponet = load_pretrained_operator("deeponet-antiderivative-1d")
    independent = jax.jit(deeponet.predict)(
        jnp.ones((32, 1)),
        jnp.linspace(0.0, 1.0, 7)[:, None],
    )
    assert independent.shape == (7,)


def test_bundled_pretrained_fno_preserves_retained_modes_across_resolutions():
    fno = load_pretrained_operator("fno-diffusion-1d")
    source_size = 16
    points = jnp.arange(source_size, dtype=jnp.float32)
    values = (1.25 + 0.4 * jnp.cos(2.0 * jnp.pi * 2.0 * points / source_size))[:, None]

    same_resolution = fno.predict(values)
    assert jnp.array_equal(same_resolution, fno.predict(values, source_size))
    retained_modes = jnp.asarray((0, 2))
    reference = jnp.fft.rfft(same_resolution[:, 0], norm="forward")[retained_modes]
    assert jnp.all(jnp.abs(reference) > 1e-6)

    for target_size in (17, 24, 31):
        transferred = fno.predict(values, target_size)
        transferred_modes = jnp.fft.rfft(transferred[:, 0], norm="forward")[
            retained_modes
        ]
        assert jnp.allclose(transferred_modes, reference, rtol=1e-5, atol=5e-10)


def test_hermitian_support_selection_is_pair_atomic():
    plan = ModalSupportDiscoveryPlan(
        _Layout(),
        2,
        conjugate_indices=jnp.asarray((0, 3, 2, 1)),
        conjugate_signs=jnp.ones((4,)),
    )
    support = discover_modal_support(
        plan,
        jnp.asarray((0.0, 3.0 + 2.0j, 0.0, 3.0 - 2.0j)),
    )
    assert set(map(int, support.multi_indices[:, 0])) == {1, 3}
    assert bool(jnp.all(support.active))


def test_hermitian_support_marks_padding_inactive_with_energetic_zero_mode():
    plan = ModalSupportDiscoveryPlan(
        _Layout(),
        3,
        conjugate_indices=jnp.asarray((1, 0, 3, 2)),
        conjugate_signs=jnp.ones((4,)),
    )
    support = discover_modal_support(
        plan,
        jnp.asarray((5.0, 5.0, 1.0, 1.0)),
    )

    indices = tuple(map(int, support.multi_indices[:, 0]))
    active = tuple(map(bool, support.active))
    assert indices == (0, 1, 0)
    assert active == (True, True, False)
    assert support.coefficients[-1, 0] == 0.0
    assert support.energies[-1] == 0.0


def test_hermitian_support_activates_conjugate_orbits_atomically():
    plan = ModalSupportDiscoveryPlan(
        _Layout(),
        2,
        conjugate_indices=jnp.asarray((0, 3, 2, 1)),
        conjugate_signs=jnp.ones((4,)),
    )
    support = discover_modal_support(
        plan,
        jnp.asarray((0.0, 3.0 + 2.0j, 0.0, 0.0)),
    )

    indices = tuple(map(int, support.multi_indices[:, 0]))
    active_indices = {
        index
        for index, is_active in zip(indices, support.active, strict=True)
        if is_active
    }
    assert active_indices == {1, 3}


def test_hermitian_omp_never_selects_a_partial_or_over_capacity_orbit():
    plan = ModalSupportDiscoveryPlan(
        _Layout(),
        3,
        method="omp",
        conjugate_indices=jnp.asarray((1, 0, 3, 2)),
        conjugate_signs=jnp.ones((4,)),
        omp_iterations=4,
        omp_step_size=0.25,
    )
    support = discover_modal_support(
        plan,
        jnp.zeros((4,)),
        measurement=jnp.eye(4),
        observations=jnp.asarray((5.0, 5.0, 1.0, 1.0))[:, None],
    )

    indices = tuple(map(int, support.multi_indices[:, 0]))
    assert indices == (0, 1, 0)
    assert tuple(map(bool, support.active)) == (True, True, False)


class _AliasedTree(eqx.Module):
    canonical: jax.Array
    alias: jax.Array


def test_parameter_subspace_reconstructs_one_canonical_alias_leaf():
    tree = _AliasedTree(jnp.asarray(1.0), jnp.asarray(1.0))
    subspace = ParameterSubspace.from_leaf_paths(
        tree,
        (".canonical", ".alias"),
        alias_groups=((".canonical", ".alias"),),
    )
    selected = eqx.tree_at(
        lambda value: value.canonical,
        subspace.initial,
        jnp.asarray(4.0),
    )
    reconstructed = subspace.reconstruct(selected)
    assert reconstructed.canonical == 4.0
    assert reconstructed.alias == 4.0
    assert subspace.total_dimension == 1
