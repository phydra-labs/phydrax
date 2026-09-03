from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

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


def test_dynamic_analytic_frame_id_cannot_collide_by_shape() -> None:
    lattice = _patterned_lattice()
    material = fm.FrequencyMaxwellMaterial(2.0, material_id="frame-material")
    first_tangent = jnp.broadcast_to(jnp.asarray((1.0, 0.0)), lattice.sample_shape + (2,))
    second_tangent = jnp.broadcast_to(
        jnp.asarray((0.0, 1.0)), lattice.sample_shape + (2,)
    )
    first = fm.FourierModalLayer(
        material,
        0.1,
        fm.VectorFourierFactorizationPlan(
            fm.AnalyticInterfaceFramePlan(first_tangent, frame_id="shared-frame")
        ),
        layer_id="first",
    )
    second = fm.FourierModalLayer(
        material,
        0.1,
        fm.VectorFourierFactorizationPlan(
            fm.AnalyticInterfaceFramePlan(second_tangent, frame_id="shared-frame")
        ),
        layer_id="second",
    )
    port = fm.HomogeneousMaxwellPort(
        fm.FrequencyMaxwellMaterial(1.0, material_id="frame-vacuum"),
        port_id="port",
    )
    problem = fm.FourierModalMaxwellProblem(
        lattice,
        2.0 * jnp.pi,
        jnp.zeros((2,)),
        port,
        (first, second),
        port,
    )
    with pytest.raises(ValueError, match="frame_id"):
        fm.prepare_fourier_modal_maxwell(problem)
