#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_hermitian_spectrum_functions_and_sylvester_action():
    matrix = jnp.asarray([[2.0 + 0.0j, 0.2j], [-0.2j, 1.0 + 0.0j]])
    spectrum = phx.linalg.HermitianSpectrum(matrix)
    assert bool(spectrum.valid)
    assert jnp.allclose(spectrum.reconstruct(), matrix)
    root = phx.linalg.hermitian_sqrt(matrix)
    assert bool(root.valid)
    assert jnp.allclose(root.value @ root.value, matrix, atol=1e-6)

    operator = phx.linalg.HermitianSylvesterOperator(matrix)
    right = jnp.asarray([[0.3, 0.1j], [-0.1j, -0.2]])
    solution = operator.solve(right)
    assert bool(solution.valid)
    assert solution.residual_norm < 1e-7


def test_bures_density_geometry_sld_distance_and_uhlmann():
    density = 0.5 * jnp.eye(2, dtype="complex128")
    tangent = jnp.asarray([[0.2, 0.1j], [-0.1j, -0.2]])
    manifold = phx.metrix.BuresDensityManifold(2)
    assert bool(manifold.contains(density))
    assert manifold.inner(density, tangent, tangent) > 0.0
    retracted = manifold.retract(density, 0.1 * tangent)
    assert bool(manifold.contains(retracted))
    assert phx.metrix.bures_squared_distance(density, density) < 1e-8

    amplitude = phx.metrix.principal_purification(density)
    alignment = phx.metrix.uhlmann_alignment(amplitude, amplitude)
    assert bool(alignment.valid)
    assert jnp.allclose(jnp.abs(alignment.overlap), 1.0)

    stratum = phx.metrix.FixedRankDensityManifold(2, 1)
    factor = jnp.asarray([[1.0 + 0.0j], [0.0j]])
    assert bool(stratum.contains(factor))
    pure = stratum.density(factor)
    assert stratum.rank_residual(pure) == 0


def test_fixed_rank_density_projection_is_horizontal_for_nonuniform_gram():
    manifold = phx.metrix.FixedRankDensityManifold(3, 2)
    factor = jnp.asarray(
        [
            [jnp.sqrt(0.8) + 0.0j, 0.0j],
            [0.0j, jnp.sqrt(0.2) + 0.0j],
            [0.0j, 0.0j],
        ]
    )
    ambient = jnp.asarray(
        [
            [0.1 + 0.4j, -0.2 + 0.3j],
            [0.5 - 0.1j, 0.2 + 0.6j],
            [-0.3 + 0.2j, 0.7 - 0.4j],
        ]
    )

    projected = manifold.project_tangent(factor, ambient)
    horizontal_residual = jnp.conj(factor.T) @ projected - jnp.conj(projected.T) @ factor
    assert jnp.allclose(horizontal_residual, 0.0, atol=1e-10)
    assert jnp.allclose(jnp.real(jnp.vdot(factor, projected)), 0.0, atol=1e-10)


def test_density_rank_stratification_rejects_non_density_inputs():
    stratification = phx.metrix.DensityRankStratification(2)
    density = jnp.diag(jnp.asarray([0.7, 0.3], dtype="complex128"))
    anti_hermitian = jnp.diag(jnp.asarray([10.0j, -10.0j]))

    assert bool(stratification.classify(density).valid)
    assert not bool(stratification.classify(density + anti_hermitian).valid)
    assert not bool(stratification.classify(jnp.diag(jnp.asarray([0.75, 0.75]))).valid)


def test_homogeneous_hypersurface_patch_residue_and_measure():
    polynomial = phx.geometry.complex.fermat_polynomial(2)
    point = jnp.asarray([1.0 + 0.0j, -1.0 + 0.0j, 0.0j])
    report = polynomial.validate(point)
    assert bool(report.valid)

    hypersurface = phx.geometry.complex.fermat_hypersurface(2)
    patch = phx.geometry.complex.HypersurfacePatchGeometry(hypersurface).evaluate(
        point / jnp.linalg.norm(point)
    )
    assert bool(patch.valid)
    assert patch.induced_metric.shape == (2, 2)
    assert jnp.isfinite(patch.residue_coefficient)

    samples = phx.geometry.complex.ProjectiveLineSamples(
        homogeneous_points=(point / jnp.linalg.norm(point))[None, :],
        chart_indices=jnp.asarray([patch.chart_index]),
        pivot_indices=jnp.asarray([patch.pivot_index]),
        polynomial_residuals=jnp.asarray([patch.polynomial_residual]),
        smoothness_margins=jnp.asarray([patch.smoothness_margin]),
        valid=jnp.asarray([patch.valid]),
        line_ids=jnp.asarray([0]),
        root_ids=jnp.asarray([0]),
    )
    target = phx.integration.projective_measure_target(
        hypersurface, samples, measure_kind="canonical"
    )
    integral = phx.integration.integrate_projective_samples(
        target, lambda homogeneous: jnp.asarray(1.0)
    )
    assert bool(integral.valid)
    assert jnp.allclose(integral.normalized_value, 1.0)


def test_density_manifolds_preserve_product_batch_semantics():
    bures = phx.metrix.BuresDensityManifold(2)
    densities = jnp.stack(
        (
            0.5 * jnp.eye(2, dtype=jnp.complex128),
            jnp.diag(jnp.asarray([0.7, 0.3], dtype=jnp.complex128)),
        )
    )
    tangents = jnp.asarray(
        (
            ((0.1, 0.05j), (-0.05j, -0.1)),
            ((-0.2, 0.03j), (-0.03j, 0.2)),
        ),
        dtype=jnp.complex128,
    )

    batched_inner = bures.inner(densities, tangents, tangents)
    individual_inner = sum(
        bures.inner(density, tangent, tangent)
        for density, tangent in zip(densities, tangents, strict=True)
    )
    retracted = bures.retract(densities, 0.01 * tangents)

    assert jnp.asarray(bures.contains(densities)).shape == ()
    assert bures.contains(densities)
    assert jnp.asarray(bures.constraint_residual(densities)).shape == ()
    assert jnp.allclose(batched_inner, individual_inner)
    assert retracted.shape == densities.shape
    assert bures.contains(retracted)

    fixed_rank = phx.metrix.FixedRankDensityManifold(2, 1)
    factors = jnp.asarray(
        (
            ((1.0 + 0.0j,), (0.0j,)),
            ((0.0j,), (1.0 + 0.0j,)),
        )
    )
    ambient = jnp.asarray(
        (
            ((0.2 + 0.1j,), (0.3 - 0.2j,)),
            ((-0.4 + 0.2j,), (0.1 + 0.3j,)),
        )
    )
    projected = fixed_rank.project_tangent(factors, ambient)
    radial = jnp.sum(jnp.real(jnp.conj(factors) * projected), axis=(-2, -1))
    updated = fixed_rank.retract(factors, 0.01 * projected)

    assert fixed_rank.contains(factors)
    assert jnp.allclose(radial, 0.0, atol=1e-10)
    assert jnp.allclose(
        jnp.sum(jnp.abs(updated) ** 2, axis=(-2, -1)),
        1.0,
    )


@pytest.mark.parametrize("hbar", [0.0, -1.0, float("nan")])
def test_bosonic_gaussian_constructors_reject_invalid_hbar(hbar):
    with pytest.raises(ValueError, match="hbar"):
        phx.metrix.BosonicGaussianState(
            jnp.zeros((2,)),
            jnp.eye(2),
            hbar=hbar,
        )
    with pytest.raises(ValueError, match="hbar"):
        phx.metrix.BosonicGaussianChannel(
            jnp.eye(2),
            jnp.eye(2),
            jnp.zeros((2,)),
            channel_id="invalid-hbar",
            hbar=hbar,
        )
