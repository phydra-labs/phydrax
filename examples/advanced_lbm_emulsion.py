#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Density-matched emulsion with N-color LBM and near-contact repulsion.

Four oil drops are pressed toward the center of a periodic water box so that every
neighboring pair is separated by a thin water film. The same N-color color-gradient
route (components ``water`` and ``oil``, pairwise tension from an
`InterfaceTensionMatrix`) is advanced twice over one declared interval:

1. control: no near-contact interaction; the films drain and drops merge;
2. repelled: a `NearContactRepulsionPlan` between facing ``oil`` interfaces
   (Montessori et al., J. Fluid Mech. 872, 2019) keeps every film intact.

A third run adds a second dispersed color, ``wax``, with a smaller ``oil|wax`` tension:
the ternary tension matrix is validated on the host and its Neumann-triangle
admissibility is reported, while momentum and every component mass stay conserved.

The report lists drop counts sampled over the interval (host connected components on
the periodic grid), the cumulative repulsion work ledger, and the relative drift of
total momentum and component masses.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

import phydrax as phx
from phydrax.discretization.lattice_boltzmann import (
    BGKCollisionPlan,
    ColorGradientLBMMethod,
    ColorGradientLBMRuntimeParameters,
    ColorGradientLBMState,
    D2Q9,
    GuoForcingPlan,
    LatticeBoltzmannBoundaryPlan,
    LatticeBoltzmannMethodPlan,
    LatticeBoltzmannPlan,
    NearContactRepulsionPlan,
)
from phydrax.equations import (
    ColorGradientLatticeBoltzmannProblem,
    compile_color_gradient_lattice_boltzmann_problem,
    CompiledColorGradientLatticeBoltzmannProblem,
)
from phydrax.interfacial_transport import InterfaceTensionMatrix


GRID = 96
RADIUS = 13.0
GAP = 4.0
APPROACH_SPEED = 0.01
STEPS = 1500
SAMPLE = 250
VISCOSITY = 1.0 / 6.0
TENSION = 0.01
REPULSION = 0.01


def _problem(
    components: tuple[str, ...], near_contact: NearContactRepulsionPlan | None
) -> CompiledColorGradientLatticeBoltzmannProblem:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(GRID, periodic=True),
            phx.discretization.UniformCellAxisSpec(GRID, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (float(GRID), float(GRID)))))
    discretization = LatticeBoltzmannPlan(grid, D2Q9()).prepare()
    method = ColorGradientLBMMethod(
        LatticeBoltzmannMethodPlan(BGKCollisionPlan(), forcing=GuoForcingPlan()),
        components,
        near_contact=near_contact,
        maximum_capillary_number=10.0,
    )
    return compile_color_gradient_lattice_boltzmann_problem(
        ColorGradientLatticeBoltzmannProblem("emulsion", 2),
        discretization,
        method,
        LatticeBoltzmannBoundaryPlan(),
        time_step=1.0,
    )


def _drops() -> tuple[list[np.ndarray], np.ndarray]:
    """Four drops on a square, one gap apart, moving toward the box center."""

    x = np.arange(GRID) + 0.5
    xx, yy = np.meshgrid(x, x, indexing="ij")
    half = RADIUS + 0.5 * GAP
    center = 0.5 * GRID
    drops = []
    velocity = np.zeros((GRID, GRID, 2))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            cx, cy = center + sx * half, center + sy * half
            drop = 0.5 * (1.0 + np.tanh((RADIUS - np.hypot(xx - cx, yy - cy)) / 2.0))
            drops.append(drop)
            velocity[..., 0] -= APPROACH_SPEED * sx * drop / np.sqrt(2.0)
            velocity[..., 1] -= APPROACH_SPEED * sy * drop / np.sqrt(2.0)
    return drops, velocity


def _drop_count(mask: np.ndarray) -> int:
    """Connected components on the doubly periodic grid."""

    labels, count = ndimage.label(mask)
    parent = list(range(count + 1))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for first, second in ((labels[0, :], labels[-1, :]), (labels[:, 0], labels[:, -1])):
        for left, right in zip(first, second, strict=True):
            if left and right:
                parent[find(int(left))] = find(int(right))
    return len({find(label) for label in range(1, count + 1)})


def _advance(
    problem: CompiledColorGradientLatticeBoltzmannProblem,
    state: ColorGradientLBMState,
    parameters: ColorGradientLBMRuntimeParameters,
    dispersed: tuple[int, ...],
) -> dict[str, Any]:
    dynamics = problem.dynamics

    @jax.jit
    def chunk(state: ColorGradientLBMState) -> tuple[ColorGradientLBMState, Any]:
        def body(
            carry: tuple[ColorGradientLBMState, Any], index: Any
        ) -> tuple[tuple[ColorGradientLBMState, Any], None]:
            current, successful = carry
            result = dynamics.step_detailed(index, 0.0, current, 1.0, parameters)
            return (result.accepted_state, successful & result.successful), None

        carry, _ = jax.lax.scan(body, (state, jnp.asarray(True)), jnp.arange(SAMPLE))
        return carry

    initial = dynamics.scalar_diagnostics(0, 0.0, state, parameters)
    counts = []
    successful = True
    for _ in range(STEPS // SAMPLE):
        state, ok = chunk(state)
        successful = successful and bool(ok)
        concentrations = np.asarray(
            problem.macroscopic_state(state, parameters).concentrations
        )
        counts.append(_drop_count(np.sum(concentrations[list(dispersed)], axis=0) > 0.5))
    final = dynamics.scalar_diagnostics(0, 0.0, state, parameters)
    momentum_scale = max(float(jnp.linalg.norm(initial.total_momentum)), 1.0)
    return {
        "successful": successful,
        "drop_counts": counts,
        "near_contact_work": float(state.near_contact_work),
        "momentum_drift": float(
            jnp.linalg.norm(final.total_momentum - initial.total_momentum)
        )
        / momentum_scale,
        "mass_drift": float(
            jnp.max(
                jnp.abs(final.component_masses - initial.component_masses)
                / initial.component_masses
            )
        ),
    }


def run() -> dict[str, Any]:
    drops, velocity = _drops()
    oil = np.sum(drops, axis=0)
    binary = ("water", "oil")
    tension = InterfaceTensionMatrix(binary, np.asarray([[0.0, TENSION], [TENSION, 0.0]]))
    densities = np.stack((1.0 - oil, oil))

    control_problem = _problem(binary, None)
    control_parameters = ColorGradientLBMRuntimeParameters(VISCOSITY, tension)
    control = _advance(
        control_problem,
        control_problem.initialize_state(densities, velocity, control_parameters),
        control_parameters,
        (1,),
    )

    repulsion = NearContactRepulsionPlan((("oil", "oil"),), interaction_range=4)
    repelled_problem = _problem(binary, repulsion)
    repelled_parameters = ColorGradientLBMRuntimeParameters(
        VISCOSITY, tension, near_contact_strength=np.asarray([REPULSION])
    )
    repelled = _advance(
        repelled_problem,
        repelled_problem.initialize_state(densities, velocity, repelled_parameters),
        repelled_parameters,
        (1,),
    )

    ternary = ("water", "oil", "wax")
    ternary_tension = InterfaceTensionMatrix(
        ternary,
        np.asarray(
            [
                [0.0, TENSION, TENSION],
                [TENSION, 0.0, 0.6 * TENSION],
                [TENSION, 0.6 * TENSION, 0.0],
            ]
        ),
    )
    wax = drops[1] + drops[2]
    ternary_problem = _problem(
        ternary,
        NearContactRepulsionPlan(
            (("oil", "oil"), ("oil", "wax"), ("wax", "wax")), interaction_range=4
        ),
    )
    ternary_parameters = ColorGradientLBMRuntimeParameters(
        VISCOSITY,
        ternary_tension,
        near_contact_strength=np.full((3,), REPULSION),
    )
    mixed = _advance(
        ternary_problem,
        ternary_problem.initialize_state(
            np.stack((1.0 - oil, drops[0] + drops[3], wax)), velocity, ternary_parameters
        ),
        ternary_parameters,
        (1, 2),
    )
    admissibility = ternary_tension.admissibility()
    return {
        "declared_interval_steps": STEPS,
        "control": control,
        "repelled": repelled,
        "ternary": {
            **mixed,
            "strict_triangle_inequality": bool(admissibility.strict_triangle_inequality),
        },
        "control_merged": min(control["drop_counts"]) < 4,
        "repelled_kept_four_drops": all(count == 4 for count in repelled["drop_counts"]),
    }


if __name__ == "__main__":
    print(run())
