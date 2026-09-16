#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.applications.supersymmetric_lattice import (
    assess_twisted_fermion_algebra,
    RegulatedTwistedDiracOperator,
    TwistedKahlerDiracOperator,
    TwistedN2SYMPlan,
    TwistedSYMCoordinateLayout,
)
from phydrax.operators.path_integral import dirac_normal_operator


def _fixture():
    theory = TwistedN2SYMPlan(
        (1, 1),
        matrix_rank=1,
        coupling=1.0,
        fermion_mass=0.5,
        coordinate_bound=1.0,
        temporal_axis=0,
        fermion_boundary_phase=-1.0,
    )
    bosonic = theory.prepare_bosonic()
    layout = TwistedSYMCoordinateLayout(bosonic)
    coordinates = jnp.zeros(layout.coordinate_shape, dtype=jnp.float64)
    coordinates = coordinates.at[..., 0, 0].set(0.2)
    coordinates = coordinates.at[..., 1, 0].set(-0.1)
    return theory, layout, coordinates


def test_real_coordinate_layout_preserves_independent_complex_links():
    _, layout, coordinates = _fixture()
    configuration = layout.unpack(coordinates)
    recovered = layout.pack(configuration)
    np.testing.assert_array_equal(recovered, coordinates)
    assert not np.shares_memory(
        np.asarray(configuration.links.values),
        np.asarray(configuration.reverse_links.values),
    )
    evidence = layout.evidence(coordinates)
    np.testing.assert_allclose(evidence.roundtrip_residual, 0.0)
    assert bool(evidence.finite)


def test_twisted_kahler_dirac_is_antisymmetric_and_has_exact_adjoint():
    theory, layout, coordinates = _fixture()
    operator = TwistedKahlerDiracOperator(theory, layout, coordinates)
    evidence = assess_twisted_fermion_algebra(
        operator,
        regulator_mass=theory.fermion_mass,
        structural_lower_bound=theory.normal_spectral_lower,
        structural_upper_bound=theory.normal_spectral_upper,
        maximum_dense_elements=operator.source.size**2,
        tolerance=1e-10,
    )
    assert bool(evidence.accepted)
    assert float(evidence.antisymmetry_residual) < 1e-12
    assert float(evidence.adjoint_residual) < 1e-12
    assert bool(evidence.interval_contains_spectrum)


def test_regulated_operator_normal_is_kahler_normal_plus_mass_shift():
    theory, layout, coordinates = _fixture()
    kahler = TwistedKahlerDiracOperator(theory, layout, coordinates)
    regulated = RegulatedTwistedDiracOperator(kahler, theory.fermion_mass)
    vector = (
        jnp.arange(kahler.source.size, dtype=jnp.float64).reshape(kahler.source.shape)
        + 1.0j
    )
    normal = dirac_normal_operator(regulated)
    expected = kahler.adjoint_mv(kahler.mv(vector)) + theory.fermion_mass**2 * vector
    np.testing.assert_allclose(normal.mv(vector), expected, rtol=1e-12, atol=1e-12)
    changed = regulated.with_links(0.5 * coordinates)
    assert changed.operator_id == regulated.operator_id
    assert not np.allclose(changed.mv(vector), regulated.mv(vector))
