#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prescribed-angle machine studies and native bounded physical design."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import core
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule
from phydrax.ein import contract
from phydrax.optim import (
    AbstractMinimizationMethod,
    Bounds,
    MinimizationProblem,
    MinimizationResult,
    OptimizationTermination,
    ProjectedLBFGS,
)

from ._magnetostatic import (
    _design_vector,
    MachineFieldResult,
    MachineSolvePolicy,
    solve_planar_machine,
)
from ._model import PlanarMachine, polar_machine


class MachineAngleStudy(StrictModule):
    """Prescribed mechanical angles and fixed-per-angle current excitations.

    ``weights`` specifies normalized angular quadrature. The default is a
    uniform sample average, suitable for a uniform periodic angle grid without
    a duplicated endpoint. Nonuniform quadrature requires explicit weights.
    Currents may vary across samples (e.g. synchronous commutation), but torque
    always differentiates angle at *fixed current*, never the commutation law.
    """

    machines: tuple[PlanarMachine, ...]
    currents: Array
    weights: Array
    angles: Array

    def __init__(
        self,
        machines: Sequence[PlanarMachine],
        currents: ArrayLike,
        *,
        weights: ArrayLike | None = None,
    ):
        models = tuple(machines)
        if not models or any(not isinstance(model, PlanarMachine) for model in models):
            raise ValueError("An angle study requires at least one prepared machine.")
        first = models[0]
        topology = first.discretization.mesh.topology_id
        if any(model.discretization.mesh.topology_id != topology for model in models[1:]):
            raise ValueError(
                "Angle models must preserve one explicit finite-element topology."
            )
        region_contract = tuple(
            (
                region.name,
                region.relative_permeability,
                tuple(np.asarray(region.remanence).tolist()),
                tuple(np.asarray(region.winding_turn_density).tolist()),
                region.rotating,
            )
            for region in first.regions
        )
        region_counts = np.bincount(
            np.asarray(first.cell_regions), minlength=len(first.regions)
        )
        if any(
            tuple(
                (
                    region.name,
                    region.relative_permeability,
                    tuple(np.asarray(region.remanence).tolist()),
                    tuple(np.asarray(region.winding_turn_density).tolist()),
                    region.rotating,
                )
                for region in model.regions
            )
            != region_contract
            or not np.array_equal(
                np.bincount(np.asarray(model.cell_regions), minlength=len(model.regions)),
                region_counts,
            )
            for model in models[1:]
        ):
            raise ValueError(
                "Angle models must preserve one material and winding profile."
            )
        if any(
            model.winding_count != first.winding_count
            or model.reference_radius != first.reference_radius
            or model.radius_bounds != first.radius_bounds
            or model.angle_window != first.angle_window
            or model.axial_length != first.axial_length
            or not np.array_equal(model.airgap_cells, first.airgap_cells)
            or not np.array_equal(model.rotation_weights, first.rotation_weights)
            or not np.array_equal(model.contour_edges, first.contour_edges)
            or not np.array_equal(model.contour_cells, first.contour_cells)
            for model in models
        ):
            raise ValueError(
                "Angle models must share winding count, radius and angle "
                "contracts, stack length, and air-gap kinematics."
            )
        excitation = np.asarray(currents, dtype=float)
        if excitation.shape == (first.winding_count,):
            excitation = np.broadcast_to(excitation, (len(models), first.winding_count))
        if excitation.shape != (
            len(models),
            first.winding_count,
        ) or not np.all(np.isfinite(excitation)):
            raise ValueError(
                "Supply finite currents per winding or per angle and winding."
            )
        quadrature = (
            np.ones(len(models)) if weights is None else np.asarray(weights, dtype=float)
        )
        if (
            quadrature.shape != (len(models),)
            or np.any(~np.isfinite(quadrature) | (quadrature < 0))
            or not np.sum(quadrature) > 0
        ):
            raise ValueError(
                "Angle weights must be finite, nonnegative, and have positive total."
            )
        self.machines = models
        self.currents = jnp.asarray(excitation)
        self.weights = jnp.asarray(quadrature / np.sum(quadrature))
        self.angles = jnp.asarray([model.reference_angle for model in models])


class MachineAngleResult(StrictModule):
    """Angle-resolved fields, mean torque, and absolute ripple (N m)."""

    fields: tuple[MachineFieldResult, ...]
    angles: Array
    torques: Array
    average_torque: Array
    rms_torque_ripple: Array
    peak_to_peak_torque_ripple: Array
    average_energy: Array
    accepted: Array


class MachineDesignResult(StrictModule):
    """Native optimizer evidence plus a fresh final physical angle scan.

    Numerical field acceptance and optimizer success are distinct; ``accepted``
    requires both, and bounds satisfaction. No outer-iteration differentiation.
    """

    design: Array
    optimization: MinimizationResult
    final_evaluation: MachineAngleResult
    accepted: Array


def polar_machine_study(
    angles: ArrayLike,
    currents: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    **machine_options: Any,
) -> MachineAngleStudy:
    """Prepare a genuine rotating polar FEM mesh for every requested angle."""
    samples = np.asarray(angles, dtype=float)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("Prescribed rotor angles must be a nonempty finite vector.")
    return MachineAngleStudy(
        tuple(polar_machine(float(angle), **machine_options) for angle in samples),
        currents,
        weights=weights,
    )


def scan_machine_angles(
    study: MachineAngleStudy,
    design: ArrayLike | None = None,
    *,
    policy: MachineSolvePolicy | None = None,
) -> MachineAngleResult:
    """Solve every prescribed angle and aggregate physical torque statistics."""
    if not isinstance(study, MachineAngleStudy):
        raise TypeError("study must be a MachineAngleStudy.")
    parameters = _design_vector(study.machines[0], design)
    fields = tuple(
        solve_planar_machine(
            machine,
            study.currents[index],
            design=parameters,
            policy=policy,
        )
        for index, machine in enumerate(study.machines)
    )
    torques = jnp.stack(tuple(field.torque for field in fields))
    mean = contract("i,i->", study.weights, torques)
    deviations = torques - mean
    variance = contract("i,i,i->", study.weights, deviations, deviations)
    # Keep zero-ripple values differentiable: sqrt(0) otherwise creates NaNs
    # when a design objective accesses other fields in the same result tree.
    positive = variance > 0.0
    ripple = jnp.where(positive, jnp.sqrt(jnp.where(positive, variance, 1.0)), 0.0)
    return MachineAngleResult(
        fields=fields,
        angles=study.angles,
        torques=torques,
        average_torque=mean,
        rms_torque_ripple=ripple,
        peak_to_peak_torque_ripple=jnp.max(torques) - jnp.min(torques),
        average_energy=contract(
            "i,i->",
            study.weights,
            jnp.stack(tuple(field.energy for field in fields)),
        ),
        accepted=jnp.all(jnp.stack(tuple(field.accepted for field in fields))),
    )


def optimize_machine_design(
    study: MachineAngleStudy,
    initial_design: ArrayLike,
    bounds: Bounds,
    *,
    objective: Callable[[MachineAngleResult, Array, Any], Array] | None = None,
    args: Any = None,
    method: AbstractMinimizationMethod | None = None,
    termination: OptimizationTermination | None = None,
    policy: MachineSolvePolicy | None = None,
) -> MachineDesignResult:
    """Optimize [rotor radius, remanence multiplier, winding-turn multiplier].

    The default objective maximizes average torque. A custom scalar objective
    consumes the full angle scan and design, e.g. torque target, ripple, and
    material use. Bounds must lie inside every angle mesh's admissible radius
    interval, with nonnegative remanence and strictly positive winding turns.
    Design coordinates are normalized internally for the native bounded method,
    retaining SI radius at all application boundaries. Final fields are independently solved.
    """
    if not isinstance(study, MachineAngleStudy) or not isinstance(bounds, Bounds):
        raise TypeError("Machine design requires MachineAngleStudy and native Bounds.")
    if objective is not None and not callable(objective):
        raise TypeError("objective must be callable or None.")
    initial = _design_vector(study.machines[0], initial_design)
    if isinstance(initial, core.Tracer):
        raise ValueError(
            "Differentiate accepted field responses, not the outer machine optimizer."
        )
    lower_tree, upper_tree = bounds.materialize(initial)
    lower, upper = np.asarray(lower_tree), np.asarray(upper_tree)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or np.any(~np.isfinite(lower) | ~np.isfinite(upper) | (lower >= upper))
    ):
        raise ValueError(
            "Machine design requires finite positive-width bounds on all three "
            "coordinates."
        )
    if (
        lower[0] < study.machines[0].radius_bounds[0]
        or upper[0] > study.machines[0].radius_bounds[1]
        or lower[1] < 0
        or lower[2] <= 0
    ):
        raise ValueError(
            "Design bounds must preserve the machine airgap, remanence, and "
            "winding domains."
        )
    if not bool(bounds.contains(initial)):
        raise ValueError("Initial machine design must lie inside the supplied bounds.")
    optimizer = ProjectedLBFGS() if method is None else method
    if not isinstance(optimizer, AbstractMinimizationMethod):
        raise TypeError("method must be an AbstractMinimizationMethod instance.")
    stopping = OptimizationTermination() if termination is None else termination
    if not isinstance(stopping, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination or None.")
    offset, span = jnp.asarray(lower), jnp.asarray(upper - lower)

    def physical_objective(normalized, context):
        candidate = offset + span * normalized
        evaluated = scan_machine_angles(study, candidate, policy=policy)
        value = (
            -evaluated.average_torque
            if objective is None
            else jnp.asarray(objective(evaluated, candidate, context))
        )
        return eqx.error_if(
            value,
            ~evaluated.accepted,
            "Machine design objective requires accepted field and independent "
            "torque evidence.",
        )

    optimization = optimizer.solve(
        MinimizationProblem(
            physical_objective,
            bounds=Bounds(jnp.zeros(3), jnp.ones(3)),
            problem_id="planar-machine-design",
        ),
        (initial - offset) / span,
        termination=stopping,
        args=args,
    )
    final_design = offset + span * optimization.parameters
    final = scan_machine_angles(study, final_design, policy=policy)
    return MachineDesignResult(
        design=final_design,
        optimization=optimization,
        final_evaluation=final,
        accepted=optimization.successful & final.accepted & bounds.contains(final_design),
    )
