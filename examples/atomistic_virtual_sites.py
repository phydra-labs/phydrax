import jax.numpy as jnp
import numpy as np

import phydrax as phx


physical_ids = np.asarray([10, 20, 30])
sites = phx.atomistic.AtomisticInteractionSitePlan(
    [10, 20, 30, 40],
    [8, 1, 1, 0],
    [0, 0, 0, 1],
    [-0.8, 0.4, 0.4, 0.0],
    physical_mask=[True, True, True, False],
)
coordinate_map = phx.atomistic.AtomisticCoordinateMapPlan(
    physical_ids,
    sites,
    [0, 1, 2, -1],
    virtual_rules=(
        phx.atomistic.VirtualSiteRule(
            phx.atomistic.VirtualSiteKind.LOCAL_FRAME,
            40,
            physical_ids,
            [0.15, 0.0, 0.0],
        ),
    ),
)
system = phx.atomistic.AtomisticSystemPlan(
    physical_ids,
    [8, 1, 1],
    [16.0, 1.0, 1.0],
    phx.atomistic.AtomisticUnitSystem.reduced(),
    coordinate_map=coordinate_map,
).prepare()
positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
realized = system.coordinate_map.realize(positions)
site_force = jnp.zeros((4, 3)).at[3].set([1.0, -0.5, 0.25])
physical_force = system.coordinate_map.force_pullback(positions, site_force)
if not bool(realized.successful) or not bool(jnp.all(jnp.isfinite(physical_force))):
    raise RuntimeError("virtual-site realization or force pullback failed")
np.testing.assert_allclose(jnp.sum(physical_force, axis=0), site_force[3], atol=1.0e-12)
print(realized.positions)
print(physical_force)
