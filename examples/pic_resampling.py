#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Vranic merging and splitting inside a reduced 1-D electromagnetic PIC run."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


pic_ = phx.discretization.pic
count = 32
grid = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformCellAxisSpec(16, periodic=True),), axis_names=("x",)
).prepare(jnp.asarray([[0.0], [1.0]]))
solver = phx.solver.ReducedMaxwellPICFieldSolver(
    phx.solver.CompatibleMaxwell1DPlan(grid), pic_.ReducedPICTransferPlan(grid)
)


def species(offset: int, sign: float, name: str) -> pic_.PICSpeciesPlan:
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(offset, offset + count), jnp.ones((count,)), ambient_dimension=1
    ).prepare()
    return pic_.PICSpeciesPlan(
        phx.discretization.ParticlePopulationPlan(support),
        pic_.PICChargeModelPlan(
            sign,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )


binning = pic_.PICCellBinningPlan((0.0,), (1.0,), (4,), (True,))
merge = pic_.ParticleMergePlan(
    binning,
    pic_.PIC_CODE_RELATIVITY,
    species=(0,),
    maximum_per_cell=6,
    momentum_bins=(2, 1, 2),
    minimum_packet_size=3,
    maximum_packet_size=4,
)
split = pic_.ParticleSplitPlan(
    binning,
    pic_.PIC_CODE_RELATIVITY,
    species=(0,),
    minimum_per_cell=2,
    minimum_child_mass=0.05,
    maximum_splits=4,
)
pic = phx.solver.ElectromagneticPICPlan(
    solver,
    species=(species(0, -1.0, "electrons"), species(100, 1.0, "ions")),
    processes=(merge, split),
)

rng = np.random.default_rng(0)
active = np.arange(count) < 20
# A crowded electron cloud in the first quarter and one stray electron.
electrons = np.where(active, rng.uniform(0.02, 0.23, count), 0.0)
electrons[19] = 0.6
velocity = np.zeros((count, 3))
velocity[:, 0] = rng.normal(0.0, 0.05, count)
dt = 0.2 * solver.field.stable_dt
state = pic.initialize(
    (electrons[:, None], rng.uniform(0.0, 1.0, (count, 1))),
    (np.where(active[:, None], velocity, 0.0), np.zeros((count, 3))),
    dt,
    active_masks=(active, active),
    masses=(np.where(active, 1.0, 0.0), np.where(active, 1.0, 0.0)),
)
step = eqx.filter_jit(lambda plan, value: plan.step_detailed(value, dt))

for index in range(3):
    result = step(pic, state)
    state = result.accepted_state
    merged, split_ = result.diagnostics.process_evidence
    projection = result.diagnostics.gauss_projection
    print(
        f"step {index}: accepted={bool(result.successful)} "
        f"merged packets={int(merged[0].events)} splits={int(split_[0].events)} "
        f"electrons={int(split_[0].active_after)} "
        f"max cell count={int(split_[0].maximum_cell_count_after)}"
    )
    print(
        f"  energy defect={float(merged[0].energy_defect):.1e} "
        f"momentum-moment distortion={float(merged[0].momentum_second_moment_distortion):.3f} "
        f"Gauss {projection.route}: {float(projection.divergence_before):.2e} -> "
        f"{float(projection.divergence_after):.1e}"
    )
