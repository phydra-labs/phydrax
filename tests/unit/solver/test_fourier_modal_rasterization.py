from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


def _harmonics(points: int = 48) -> Any:
    return LatticeHarmonicPlan.parallelogramic((1, 1), (points, points)).prepare(
        jnp.eye(2)
    )


def _circle(radius: Any = 0.25) -> Any:
    return phx.geometry.Circle(
        center=(0.5, 0.5),
        radius=radius,
        feature_id="inclusion",
    ).compile()


def test_fourier_modal_rasterization_scenario_1() -> None:
    harmonics = _harmonics()
    plan = fm.FourierModalRasterizationPlan(harmonics)
    result = fm.rasterize_fourier_modal_material(
        plan,
        _circle(),
        inside_permittivity=12.0,
        outside_permittivity=1.0,
        material_id="disk",
        passive=True,
        reciprocal=True,
    )

    assert result.material.permittivity.shape == harmonics.sample_shape
    assert result.evidence.samples_per_pixel == 1
    assert not result.evidence.parameter_differentiable
    assert float(jnp.min(result.material.permittivity)) == pytest.approx(1.0)
    assert float(jnp.max(result.material.permittivity)) == pytest.approx(12.0)
    assert float(jnp.mean(result.fill_fraction)) == pytest.approx(
        np.pi * 0.25**2, abs=2.5e-3
    )
    line = LatticeHarmonicPlan.parallelogramic((1,), (8,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    with pytest.raises(ValueError, match="2-D lattice"):
        fm.FourierModalRasterizationPlan(line)

    plan = fm.FourierModalRasterizationPlan(_harmonics(8))
    with pytest.raises(TypeError, match="one numeric scalar"):
        fm.rasterize_fourier_modal_material(
            plan,
            _circle(),
            inside_permittivity=jnp.ones((2,)),
            material_id="invalid",
        )


def test_smoothed_subpixel_rasterization_is_parameter_differentiable() -> None:
    harmonics = _harmonics(32)
    plan = fm.FourierModalRasterizationPlan(
        harmonics,
        fm.FourierModalRasterizationPolicy(
            samples_per_axis=3,
            smoothing_width=0.02,
        ),
    )
    geometry = _circle()
    radius_id = phx.geometry.ParameterId("inclusion", "radius")

    def mean_fill(radius: Any) -> Any:
        current = geometry.with_parameters({radius_id: radius})
        return jnp.mean(
            fm.rasterize_fourier_modal_material(
                plan,
                current,
                inside_permittivity=4.0,
                material_id="smooth-disk",
            ).fill_fraction
        )

    derivative = jax.grad(mean_fill)(jnp.asarray(0.25))
    step = 1.0e-3
    reference = (mean_fill(0.25 + step) - mean_fill(0.25 - step)) / (2.0 * step)
    result = fm.rasterize_fourier_modal_material(
        plan,
        geometry,
        inside_permittivity=4.0,
        material_id="smooth-disk",
    )

    assert result.evidence.samples_per_pixel == 9
    assert result.evidence.parameter_differentiable
    assert float(derivative) == pytest.approx(float(reference), rel=2.0e-2)
    assert float(derivative) == pytest.approx(2.0 * np.pi * 0.25, rel=8.0e-2)
