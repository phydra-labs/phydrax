from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


def _patterned_lattice():
    return LatticeHarmonicPlan.parallelogramic((3,), (9,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )


def test_inverse_factorization_is_uniformly_exact() -> None:
    lattice = _patterned_lattice()
    material = fm.FrequencyMaxwellMaterial(
        jnp.full(lattice.sample_shape, 2.5),
        material_id="uniform-grid",
    )
    direct = fm.prepare_fourier_material(
        material,
        lattice,
        fm.DirectFourierFactorizationPlan(),
    )
    inverse = fm.prepare_fourier_material(
        material,
        lattice,
        fm.InverseFourierFactorizationPlan(),
    )
    np.testing.assert_allclose(
        np.asarray(inverse.permittivity),
        np.asarray(direct.permittivity),
        rtol=1e-11,
        atol=1e-11,
    )


def test_analytic_vector_factorization_returns_normalized_frame() -> None:
    lattice = _patterned_lattice()
    coordinate = lattice.fractional_coordinates[..., 0]
    material = fm.FrequencyMaxwellMaterial(
        jnp.where(coordinate < 0.5, 4.0, 1.0),
        material_id="lamellar",
    )
    tangent = jnp.broadcast_to(jnp.asarray((0.0, 1.0)), lattice.sample_shape + (2,))
    prepared = fm.prepare_fourier_material(
        material,
        lattice,
        fm.VectorFourierFactorizationPlan(
            fm.AnalyticInterfaceFramePlan(tangent, frame_id="analytic")
        ),
    )
    assert prepared.tangent_field is not None
    np.testing.assert_allclose(
        np.asarray(jnp.sum(jnp.abs(prepared.tangent_field) ** 2, axis=-1)),
        1.0,
        atol=1e-12,
    )
    assert not bool(prepared.diagnostics.frame_gradient_omitted)


def test_frozen_jones_frame_reports_omitted_gradient() -> None:
    lattice = _patterned_lattice()
    coordinate = lattice.fractional_coordinates[..., 0]
    material = fm.FrequencyMaxwellMaterial(
        1.0 + 3.0 * jnp.exp(-(((coordinate - 0.5) / 0.15) ** 2)),
        material_id="smooth-pattern",
    )
    prepared = fm.prepare_fourier_material(
        material,
        lattice,
        fm.VectorFourierFactorizationPlan(
            fm.JonesDirectFramePlan(differentiation="frozen")
        ),
    )
    assert prepared.tangent_field is not None
    assert bool(prepared.diagnostics.frame_gradient_omitted)
    assert bool(jnp.all(jnp.isfinite(prepared.tangent_field)))
