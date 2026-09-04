from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.closure_data import (
    ClosureField,
    FilterSpec,
    les_energy_transfer_target,
    les_reynolds_stress_target,
    les_scalar_flux_target,
    les_stress_divergence_target,
    LESFilterPair,
    prepare_periodic_les_analysis,
    sgs_energy_target,
)
from phydrax.discretization._axis_domain import AxisDomain
from phydrax.discretization.spectral._basis import FourierBasisPlan
from phydrax.discretization.spectral._operators import spectral_derivative_operator
from phydrax.discretization.spectral._space import TensorSpectralPlan
from phydrax.discretization.spectral._transfer import prepare_spectral_modal_transfer
from phydrax.equations._les_closures import ResolvedLESFilter


_AXES = ("x", "y", "z")


def _space(shape):
    return TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for count in shape),
        axis_names=_AXES,
        field_name="velocity",
    ).prepare(tuple(AxisDomain.periodic(0.0, 2.0 * np.pi) for _ in shape))


def _filter(name="resolved"):
    return ResolvedLESFilter(
        name,
        family="sharp-fourier-projection",
        axis_names=_AXES,
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )


def _context(source_shape=(12, 12, 12), resolved_shape=(4, 4, 4)):
    source = _space(source_shape)
    resolved = _space(resolved_shape)
    context = prepare_periodic_les_analysis(
        source,
        resolved,
        _filter(),
        reference_manifest_id="reference-manifest",
    )
    return source, resolved, context


def _coordinates(space):
    return jnp.meshgrid(*(axis.nodes for axis in space.axes), indexing="ij")


def _velocity_field(source):
    x, y, z = _coordinates(source)
    values = jnp.stack(
        (
            jnp.sin(x) + 0.2 * jnp.sin(3.0 * y),
            jnp.cos(y) + 0.15 * jnp.cos(4.0 * z),
            jnp.sin(z) + 0.1 * jnp.cos(3.0 * x),
        ),
        axis=-1,
    )
    return ClosureField(
        values,
        name="velocity",
        units="m/s",
        schema_id="flow-schema",
        lineage_ids=("source-snapshot",),
    )


def test_periodic_filter_matches_modal_transfer_retained_modes_nyquist_and_constants():
    source, resolved, context = _context((8, 8, 8), (4, 4, 4))
    coarse_coefficients = jnp.zeros(resolved.modal_shape, dtype=jnp.complex128)
    coarse_coefficients = coarse_coefficients.at[1, 0, 0].set(1.25 - 0.5j)
    coarse_coefficients = coarse_coefficients.at[2, 0, 0].set(0.75)
    embedded = prepare_spectral_modal_transfer(resolved, source)(coarse_coefficients)
    np.testing.assert_allclose(
        context.filter_modal(embedded), coarse_coefficients, atol=1e-12
    )

    source_coefficients = source.project(jnp.ones(source.physical_shape))
    runtime_convention = prepare_spectral_modal_transfer(source, resolved)(
        source_coefficients
    )
    np.testing.assert_allclose(
        context.filter_modal(source_coefficients), runtime_convention, atol=1e-12
    )
    np.testing.assert_allclose(
        context.filter_field(jnp.ones(source.physical_shape)),
        jnp.ones(resolved.physical_shape),
        atol=1e-12,
    )


def test_periodic_product_filter_projects_the_source_product_not_separate_factors():
    source, resolved, context = _context()
    x, y, _ = _coordinates(source)
    left = jnp.sin(x) + 0.4 * jnp.cos(5.0 * y)
    right = jnp.cos(y) - 0.3 * jnp.sin(4.0 * x)
    expected = resolved.reconstruct(context.modal_transfer(source.project(left * right)))
    np.testing.assert_allclose(context.filter_product(left, right), expected, atol=2e-12)


def test_reynolds_stress_conventions_divergence_and_positive_forward_sign():
    source, resolved, context = _context()
    velocity = _velocity_field(source)
    derivatives = tuple(spectral_derivative_operator(resolved, axis) for axis in range(3))

    full = les_reynolds_stress_target(velocity, context, convention="full")
    deviatoric = les_reynolds_stress_target(velocity, context, convention="deviatoric")
    np.testing.assert_allclose(full.values, jnp.swapaxes(full.values, -1, -2), atol=2e-12)
    np.testing.assert_allclose(
        jnp.trace(deviatoric.values, axis1=-2, axis2=-1), 0.0, atol=2e-12
    )
    isotropic = (jnp.trace(full.values, axis1=-2, axis2=-1) / 3.0)[
        ..., None, None
    ] * jnp.eye(3)
    np.testing.assert_allclose(full.values - deviatoric.values, isotropic, atol=2e-12)

    divergence = les_stress_divergence_target(full, context, derivatives)
    expected_divergence = jnp.zeros(resolved.physical_shape + (3,))
    for axis in range(3):
        expected_divergence = expected_divergence + resolved.partial_derivative(
            full.values[..., :, axis], axis=axis
        )
    np.testing.assert_allclose(divergence.values, expected_divergence, atol=2e-11)

    transfer = les_energy_transfer_target(full, velocity, context, derivatives)
    resolved_velocity = context.filter_field(velocity.values)
    gradient = jnp.stack(
        tuple(
            resolved.partial_derivative(resolved_velocity, axis=axis) for axis in range(3)
        ),
        axis=-1,
    )
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    expected_transfer = -jnp.sum(full.values * strain, axis=(-2, -1))
    np.testing.assert_allclose(transfer.values, expected_transfer, atol=2e-11)
    assert dict(transfer.node.parameters)["transfer_sign"] == (
        "positive-forward:-tau_ij*S_ij"
    )

    dag = context.analysis_dag((full, divergence, transfer))
    reference = context.bind_target(transfer, dag)
    assert reference.reference_manifest_id == "reference-manifest"
    assert reference.source_discretization_id == source.prepared_id
    assert reference.resolved_discretization_id == resolved.prepared_id
    assert reference.filter_id == context.resolved_filter.filter_id
    assert reference.target_id == transfer.target_id
    assert reference.analysis_dag_id == dag.dag_id


def test_named_generic_scalar_flux_uses_the_same_exact_projection():
    source, _, context = _context()
    velocity = _velocity_field(source)
    x, y, z = _coordinates(source)
    scalar = ClosureField(
        jnp.cos(x - y) + 0.3 * jnp.sin(4.0 * z),
        name="mixture_fraction",
        units="1",
        schema_id=velocity.schema_id,
        lineage_ids=("source-snapshot",),
    )
    target = les_scalar_flux_target(
        velocity,
        scalar,
        context,
        name="mixture_fraction",
    )
    expected = context.filter_field(velocity.values * scalar.values[..., None]) - (
        context.filter_field(velocity.values)
        * context.filter_field(scalar.values)[..., None]
    )
    np.testing.assert_allclose(target.values, expected, atol=2e-12)
    assert target.node.output_name == "mixture_fraction_sgs_flux"
    assert target.target_kind == "scalar_flux"


def test_filter_identity_transfer_derivative_and_source_resolution_mismatches_refuse():
    source = _space((8, 8, 8))
    resolved = _space((4, 4, 4))
    wrong_axes_filter = ResolvedLESFilter(
        "wrong-axes",
        family="sharp-fourier-projection",
        axis_names=("a", "b", "c"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    with pytest.raises(ValueError, match="axis names"):
        prepare_periodic_les_analysis(
            source,
            resolved,
            wrong_axes_filter,
            reference_manifest_id="reference",
        )

    context = prepare_periodic_les_analysis(
        source,
        resolved,
        _filter(),
        reference_manifest_id="reference",
    )
    velocity = _velocity_field(source)
    stress = les_reynolds_stress_target(velocity, context)
    wrong_derivatives = tuple(
        spectral_derivative_operator(source, axis) for axis in range(3)
    )
    with pytest.raises(ValueError, match="matching exact first derivative"):
        les_stress_divergence_target(stress, context, wrong_derivatives)

    too_coarse = _space((4, 4, 4))
    too_fine = _space((6, 6, 6))
    with pytest.raises(ValueError, match="source resolution"):
        prepare_periodic_les_analysis(
            too_coarse,
            too_fine,
            _filter(),
            reference_manifest_id="reference",
        )


def test_les_additions_do_not_change_existing_filter_energy_or_pair_semantics():
    prepared = FilterSpec.identity().prepare((4, 4, 4))
    values = jnp.ones((4, 4, 4, 3))
    np.testing.assert_array_equal(prepared(values), values)

    source, _, context = _context((8, 8, 8), (4, 4, 4))
    stress = les_reynolds_stress_target(_velocity_field(source), context)
    energy = sgs_energy_target(stress)
    np.testing.assert_allclose(
        energy.values,
        0.5 * jnp.trace(stress.values, axis1=-2, axis2=-1),
        atol=0.0,
    )

    test_filter = _filter("test")
    pair = LESFilterPair(context.resolved_filter, test_filter)
    assert pair.primary_filter.filter_id == context.resolved_filter.filter_id
    assert pair.test_filter.filter_id == test_filter.filter_id
    assert pair.test_filter_input == "primary-resolved"
    with pytest.raises(ValueError, match="distinct identities"):
        LESFilterPair(context.resolved_filter, context.resolved_filter)
