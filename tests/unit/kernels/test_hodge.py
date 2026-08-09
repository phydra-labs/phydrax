#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _annulus_complex():
    outer = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    vertices = np.concatenate((outer, 0.4 * outer), axis=0)
    faces = np.asarray(
        [(index, (index + 1) % 4, 4 + (index + 1) % 4) for index in range(4)]
        + [(index, 4 + (index + 1) % 4, 4 + index) for index in range(4)],
        dtype=np.int32,
    )
    return phx.graph.triangle_mesh_to_cochain_complex(vertices, faces)


def _packed_columns(complex_ir, degree, columns):
    packed = jnp.zeros((columns.shape[1], complex_ir.num_cells), dtype=columns.dtype)
    start = complex_ir.cell_offsets[degree]
    return packed.at[:, start : start + columns.shape[0]].set(columns.T)


def test_hodge_sectors_span_cochains_and_satisfy_differential_invariants():
    complex_ir = _annulus_complex()
    spectra = phx.graph.cochain_hodge_sector_spectra(complex_ir, 1)

    assert spectra.harmonic is not None
    assert spectra.exact is not None
    assert spectra.coexact is not None
    assert spectra.harmonic.mode_count == 1
    assert spectra.total_rank == complex_ir.cell_counts[1]

    exact = _packed_columns(complex_ir, 1, spectra.exact.eigenfunctions)
    coexact = _packed_columns(complex_ir, 1, spectra.coexact.eigenfunctions)
    harmonic = _packed_columns(complex_ir, 1, spectra.harmonic.eigenfunctions)

    exact_derivative = jnp.stack(
        [
            phx.graph.cochain_exterior_derivative(complex_ir.graph, value, 1)
            for value in exact
        ]
    )
    coexact_codifferential = jnp.stack(
        [
            phx.graph.cochain_codifferential(complex_ir.graph, value, 1)
            for value in coexact
        ]
    )
    harmonic_derivative = phx.graph.cochain_exterior_derivative(
        complex_ir.graph, harmonic[0], 1
    )
    harmonic_codifferential = phx.graph.cochain_codifferential(
        complex_ir.graph, harmonic[0], 1
    )

    assert jnp.allclose(exact_derivative, 0.0, atol=1e-8)
    assert jnp.allclose(coexact_codifferential, 0.0, atol=1e-8)
    assert jnp.allclose(harmonic_derivative, 0.0, atol=1e-8)
    assert jnp.allclose(harmonic_codifferential, 0.0, atol=1e-8)


def _hodge_kernel(spectra):
    return phx.kernels.CochainHodgeSpectralKernel(
        spectra,
        harmonic_multiplier=phx.kernels.HeatSpectralMultiplier(0.0),
        exact_multiplier=phx.kernels.MaternSpectralMultiplier(0.6, 1.2),
        coexact_multiplier=phx.kernels.MaternSpectralMultiplier(0.9, 1.7),
        harmonic_amplitude=0.5,
        exact_amplitude=0.8,
        coexact_amplitude=0.7,
    )


def test_hodge_sector_covariance_is_orientation_conjugate_and_finite_feature():
    complex_ir = _annulus_complex()
    spectra = phx.graph.cochain_hodge_sector_spectra(complex_ir, 1)
    kernel = _hodge_kernel(spectra)
    entities = complex_ir.cell_entities(1)
    matrix = kernel.matrix(entities, entities)

    edge_signs = np.where(np.arange(complex_ir.cell_counts[1]) % 2, -1.0, 1.0)
    reoriented = phx.graph.reorient_cochain_complex(
        complex_ir,
        (
            np.ones((complex_ir.cell_counts[0],)),
            edge_signs,
            np.ones((complex_ir.cell_counts[2],)),
        ),
    )
    reoriented_spectra = phx.graph.cochain_hodge_sector_spectra(reoriented, 1)
    reoriented_kernel = _hodge_kernel(reoriented_spectra)
    reoriented_matrix = reoriented_kernel.matrix(
        reoriented.cell_entities(1), reoriented.cell_entities(1)
    )

    assert kernel.feature_rank == spectra.total_rank
    assert jnp.allclose(matrix, matrix.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(np.asarray(matrix))) >= -1e-9
    assert jnp.allclose(
        reoriented_matrix,
        edge_signs[:, None] * matrix * edge_signs[None, :],
        atol=1e-8,
    )


def test_hodge_sector_sum_reuses_weight_space_gp_inference():
    complex_ir = _annulus_complex()
    kernel = _hodge_kernel(phx.graph.cochain_hodge_sector_spectra(complex_ir, 1))
    entities = jnp.tile(complex_ir.cell_entities(1), 2)
    model = phx.uq.ExactGaussianProcessDiscrepancy(
        entities,
        jnp.zeros((entities.size,)),
    )
    state = phx.uq.GaussianProcessLikelihoodState(kernel=kernel, noise_scale=0.1)

    factor = model.factor(state=state)
    dense = phx.uq.ExactGaussianProcessFactor(entities, state=state)
    residual = jnp.zeros((entities.size,))

    assert isinstance(factor, phx.uq.FiniteFeatureGaussianProcessFactor)
    assert jnp.allclose(
        factor.log_probability(residual), dense.log_probability(residual), atol=1e-8
    )
