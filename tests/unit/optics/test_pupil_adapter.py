#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import TensorGridPlan, UniformCellAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.geometric._interface import OpticalRayState
from phydrax.optics.geometric._sequential import SequentialOpticsResult
from phydrax.optics.wave._fields import PlaneFieldSpace
from phydrax.optics.wave._pupil_adapter import (
    PupilFieldAdapterStatus,
    sequential_pupil_to_scalar_field,
)


def _space(scale: float, *, periodic: bool = False) -> PlaneFieldSpace:
    grid = TensorGridPlan(
        (
            UniformCellAxisSpec(2, periodic=periodic),
            UniformCellAxisSpec(2, periodic=periodic),
        )
    ).prepare(jnp.asarray(((-scale, -scale), (scale, scale))))
    topology = "periodic-cell" if periodic else "finite-window"
    return PlaneFieldSpace(grid, RigidFrame.identity(3), topology)


def _result(
    final_origins: jnp.ndarray, *, successful: bool = True
) -> SequentialOpticsResult:
    flattened = final_origins.reshape((-1, 3))
    count = flattened.shape[0]
    directions = jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (count, 3))
    rays = OpticalRayState(
        flattened,
        directions,
        jnp.ones((count,)),
        jnp.zeros((count,)),
        jnp.zeros((count,)),
    )
    valid = jnp.full((count,), successful)
    return SequentialOpticsResult(
        rays=rays,
        valid=valid,
        status=jnp.zeros((count,), dtype=jnp.int32),
        traversed_surfaces=jnp.ones((count,), dtype=jnp.int32),
        minimum_snell_discriminant=jnp.ones((count,)),
        minimum_aperture_margin=jnp.ones((count,)),
        maximum_intersection_residual=jnp.zeros((count,)),
        finite=jnp.asarray(True),
        successful=jnp.asarray(successful),
        producer_id="sequential-test",
    )


def test_noncaustic_one_to_one_pupil_transport_preserves_power_density() -> None:
    pupil = _space(1.0)
    output = _space(2.0)
    result = _result(output.world_points)
    converted = sequential_pupil_to_scalar_field(
        result,
        pupil,
        output,
        pupil.world_points,
        jnp.ones(pupil.shape, dtype=jnp.complex64),
        6.0,
        0.0,
        3.0,
    )
    assert bool(converted.evidence.accepted)
    assert int(converted.evidence.status) == int(PupilFieldAdapterStatus.SUCCESS)
    np.testing.assert_allclose(converted.evidence.minimum_signed_jacobian_ratio, 4.0)
    np.testing.assert_allclose(converted.field.values, 0.5)


def test_periodic_pupil_topology_is_typed_rejection() -> None:
    pupil = _space(1.0, periodic=True)
    output = _space(1.0, periodic=True)
    converted = sequential_pupil_to_scalar_field(
        _result(output.world_points),
        pupil,
        output,
        pupil.world_points,
        jnp.ones(pupil.shape),
        6.0,
        0.0,
        3.0,
    )
    assert not bool(converted.evidence.accepted)
    assert not bool(converted.evidence.topology_supported)
    assert int(converted.evidence.status) == int(
        PupilFieldAdapterStatus.UNSUPPORTED_TOPOLOGY
    )
    assert np.isnan(np.asarray(converted.field.values)).all()


def test_folded_ray_map_is_typed_caustic_rejection() -> None:
    pupil = _space(1.0)
    output = _space(1.0)
    folded = output.world_points.at[1, :, :].set(output.world_points[1, ::-1, :])
    converted = sequential_pupil_to_scalar_field(
        _result(folded),
        pupil,
        output,
        pupil.world_points,
        jnp.ones(pupil.shape),
        6.0,
        0.0,
        3.0,
        coordinate_tolerance=10.0,
    )
    assert not bool(converted.evidence.accepted)
    assert not bool(converted.evidence.noncaustic)
    assert int(converted.evidence.status) == int(PupilFieldAdapterStatus.CAUSTIC_OR_FOLD)
