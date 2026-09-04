#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import (
    LatticeHarmonicDiscretization,
    LatticeHarmonicPlan,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.geometry import RigidFrame
from phydrax.optics.wave._fields import PlaneFieldSpace
from phydrax.optics.wave._maxwell_adapter import (
    fourier_modal_field_to_tangential_plane,
    tile_periodic_plane_to_finite_window,
)
from phydrax.solver.maxwell.fourier_modal import FourierModalFieldResult


def _space(
    shape: tuple[int, int],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    periodic: bool,
) -> PlaneFieldSpace:
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(count, periodic=periodic) for count in shape)
    ).prepare(
        jnp.asarray(
            (
                tuple(axis_bounds[0] for axis_bounds in bounds),
                tuple(axis_bounds[1] for axis_bounds in bounds),
            )
        )
    )
    topology = "periodic-cell" if periodic else "finite-window"
    return PlaneFieldSpace(grid, RigidFrame.identity(3), topology)


def _modal_field(shape: tuple[int, int]) -> FourierModalFieldResult:
    electric = jnp.zeros(shape + (3, 1), dtype=jnp.complex64)
    magnetic = jnp.zeros(shape + (3, 1), dtype=jnp.complex64)
    electric = electric.at[..., 0, 0].set(2.0 + 1.0j)
    electric = electric.at[..., 1, 0].set(3.0 - 1.0j)
    magnetic = magnetic.at[..., 0, 0].set(-1.0j)
    magnetic = magnetic.at[..., 1, 0].set(4.0)
    return FourierModalFieldResult(
        electric_harmonics=jnp.zeros((1, 3, 1), dtype=jnp.complex64),
        magnetic_harmonics=jnp.zeros((1, 3, 1), dtype=jnp.complex64),
        electric_field=electric,
        magnetic_field=magnetic,
        longitudinal_offset=jnp.asarray(0.25),
        boundary_solve_residual=jnp.asarray(0.0),
        local_constitutive_residual=jnp.asarray(0.0),
        continuous_segment_defect=jnp.asarray(0.0),
        continuous_segment_index=jnp.asarray(-1, dtype=jnp.int32),
        continuous_status=jnp.asarray(-1, dtype=jnp.int32),
        layer_id="layer-0",
    )


def test_fourier_modal_adapter_preserves_periodic_cell_and_tangential_values() -> None:
    shape = (2, 2)
    lattice = LatticeHarmonicDiscretization(
        LatticeHarmonicPlan.parallelogramic((1, 1), shape), jnp.eye(2)
    )
    periodic_space = _space(shape, ((0.0, 1.0), (0.0, 1.0)), periodic=True)
    adapted = fourier_modal_field_to_tangential_plane(
        _modal_field(shape),
        lattice,
        periodic_space,
        6.0,
        0.25,
    )
    assert bool(adapted.evidence.accepted)
    assert adapted.evidence.source_topology == "periodic-cell"
    assert adapted.evidence.target_topology == "periodic-cell"
    assert adapted.plane.electric.space.topology == "periodic-cell"
    np.testing.assert_allclose(adapted.plane.electric.values[..., 0], 2.0 + 1.0j)
    np.testing.assert_allclose(adapted.plane.electric.values[..., 1], 3.0 - 1.0j)
    np.testing.assert_allclose(adapted.plane.magnetic.values[..., 0], -1.0j)
    np.testing.assert_allclose(adapted.plane.magnetic.values[..., 1], 4.0)


def test_periodic_tiling_and_windowing_records_finite_support_evidence() -> None:
    shape = (2, 2)
    lattice = LatticeHarmonicDiscretization(
        LatticeHarmonicPlan.parallelogramic((1, 1), shape), jnp.eye(2)
    )
    periodic_space = _space(shape, ((0.0, 1.0), (0.0, 1.0)), periodic=True)
    periodic = fourier_modal_field_to_tangential_plane(
        _modal_field(shape), lattice, periodic_space, 6.0, 0.25
    ).plane
    finite_space = _space((4, 2), ((0.0, 2.0), (0.0, 1.0)), periodic=False)
    window = jnp.asarray([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.25, 0.25]])
    converted = tile_periodic_plane_to_finite_window(
        periodic,
        finite_space,
        lattice.primitive_vectors,
        window,
        tile_counts=(2, 1),
    )
    assert bool(converted.evidence.accepted)
    assert converted.evidence.source_topology == "periodic-cell"
    assert converted.evidence.target_topology == "finite-window"
    assert converted.evidence.tile_counts == (2, 1)
    assert converted.plane.electric.space.topology == "finite-window"
    np.testing.assert_allclose(
        converted.plane.electric.values[..., 0],
        (2.0 + 1.0j) * np.asarray(window),
    )


def test_direct_fourier_modal_to_finite_window_requires_explicit_conversion() -> None:
    shape = (2, 2)
    lattice = LatticeHarmonicDiscretization(
        LatticeHarmonicPlan.parallelogramic((1, 1), shape), jnp.eye(2)
    )
    finite_space = _space(shape, ((0.0, 1.0), (0.0, 1.0)), periodic=False)
    with pytest.raises(ValueError, match="periodic-cell"):
        fourier_modal_field_to_tangential_plane(
            _modal_field(shape), lattice, finite_space, 6.0, 0.25
        )
