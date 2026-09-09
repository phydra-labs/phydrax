#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Linear Az finite elements and independently evaluated mechanical torque."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule
from phydrax.ein import contract
from phydrax.linalg import (
    ArraySpace,
    DenseLinearOperator,
    FactorizationPolicy,
    OperatorProperties,
    factorize,
)

from ._model import PlanarMachine, VACUUM_PERMEABILITY


class MachineSolvePolicy(StrictModule):
    """Independent equation and energy/air-stress torque acceptance tolerances."""

    residual_absolute: float = eqx.field(static=True)
    residual_relative: float = eqx.field(static=True)
    torque_absolute: float = eqx.field(static=True)
    torque_relative: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual_absolute: float = 1e-9,
        residual_relative: float = 1e-8,
        torque_absolute: float = 1e-8,
        torque_relative: float = 1e-6,
    ):
        values = tuple(
            map(
                float,
                (
                    residual_absolute,
                    residual_relative,
                    torque_absolute,
                    torque_relative,
                ),
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Machine acceptance tolerances must be finite and nonnegative."
            )
        (
            self.residual_absolute,
            self.residual_relative,
            self.torque_absolute,
            self.torque_relative,
        ) = values


class MachineFieldResult(StrictModule):
    """Accepted-point physical fields and unrounded diagnostic evidence.

    ``magnetic_field`` is B in tesla; ``field_strength`` is H in A/m. Energy
    uses the constitutive potential 0.5*nu*|B-Br|^2, including its fixed-remanence
    reference. Coenergy is current*flux minus that potential, not 0.5*I*flux for
    a permanent-magnet machine. Positive torque is counterclockwise (+z).

    ``stress_torque`` is the air-only Maxwell tensor contracted with a nodal
    virtual-rotation gradient, independent of energy differentiation.
    ``contour_torque`` is a further polygon contour/averaged-P1-trace estimate;
    its discrepancy is a spatial-resolution indicator, not a convergence claim.
    """

    potential: Array
    coordinates: Array
    magnetic_field: Array
    field_strength: Array
    current_density: Array
    flux_linkage: Array
    energy: Array
    coenergy: Array
    torque: Array
    stress_torque: Array
    contour_torque: Array | None
    residual_norm: Array
    relative_residual: Array
    gauge_error: Array
    torque_discrepancy: Array
    linear_accepted: Array
    accepted: Array


def _design_vector(machine: PlanarMachine, design: ArrayLike | None) -> Array:
    # Coordinates: rotor radius [m], remanence multiplier, winding-turn multiplier.
    parameters = jnp.asarray(
        (machine.reference_radius, 1.0, 1.0) if design is None else design
    )
    if parameters.shape != (3,):
        raise ValueError(
            "Machine design is [rotor_radius, remanence_scale, turns_scale]."
        )
    return eqx.error_if(
        parameters,
        jnp.any(~jnp.isfinite(parameters))
        | (parameters[1] < 0.0)
        | (parameters[2] <= 0.0),
        "Machine design requires finite nonnegative remanence and positive "
        "winding turns.",
    )


def _material_arrays(machine, parameters, angle_delta):
    ids = machine.cell_regions
    reluctivity = 1.0 / (
        VACUUM_PERMEABILITY
        * jnp.asarray([region.relative_permeability for region in machine.regions])
    )
    remanence = parameters[1] * jnp.stack(
        tuple(region.remanence for region in machine.regions)
    )
    rotation = jnp.asarray([region.rotating for region in machine.regions])
    angle = (machine.reference_angle + angle_delta) * rotation
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    remanence = jnp.stack(
        (
            cosine * remanence[:, 0] - sine * remanence[:, 1],
            sine * remanence[:, 0] + cosine * remanence[:, 1],
        ),
        axis=-1,
    )
    windings = parameters[2] * jnp.stack(
        tuple(region.winding_turn_density for region in machine.regions)
    )
    return reluctivity[ids], remanence[ids], windings[ids]


def _geometry(machine, parameters, angle_delta):
    points = machine.coordinates(parameters[0], angle_delta)
    # P1 affine triangles: one centroid integrates all assembly and field terms
    # exactly, avoiding general high-order quadrature storage for this profile.
    geometry = machine.discretization.evaluate_block_geometry(
        "Az",
        0,
        points,
        jnp.asarray(((1.0 / 3.0, 1.0 / 3.0),)),
        jnp.asarray((0.5,)),
    )
    gradients = geometry.physical_gradients[:, 0]
    curls = jnp.stack((gradients[:, :, 1], -gradients[:, :, 0]), axis=-1)
    return points, geometry.measure, gradients, curls


def _field_quantities(
    machine,
    potential,
    areas,
    curls,
    reluctivity,
    remanence,
    windings,
    currents,
):
    local = potential[machine.discretization.mesh.blocks[0].vertices]
    magnetic_field = contract("cid,ci->cd", curls, local)
    field_strength = reluctivity[:, None] * (magnetic_field - remanence)
    current_density = contract("ck,k->c", windings, currents)
    mean_potential = jnp.mean(local, axis=1)
    flux = machine.axial_length * contract(
        "c,ck,c->k", areas, windings, mean_potential
    )
    magnetic_offset = magnetic_field - remanence
    energy = 0.5 * machine.axial_length * contract(
        "c,c,cd,cd->", areas, reluctivity, magnetic_offset, magnetic_offset
    )
    coenergy = contract("k,k->", currents, flux) - energy
    return magnetic_field, field_strength, current_density, flux, energy, coenergy


def machine_coenergy(
    machine: PlanarMachine,
    potential: ArrayLike,
    currents: ArrayLike,
    *,
    design: ArrayLike | None = None,
    angle_delta: ArrayLike = 0.0,
) -> Array:
    """Variational coenergy at any Az; stationary only at a solved field.

    Differentiating this expression with respect to angle at *fixed* accepted
    nodal Az gives torque. Design derivatives may differentiate the native
    implicit linear solve, never the outer optimization iterations.
    """
    if not isinstance(machine, PlanarMachine):
        raise TypeError("machine must be a prepared PlanarMachine.")
    parameters = _design_vector(machine, design)
    values, excitation = jnp.asarray(potential), jnp.asarray(currents)
    if (
        values.shape != (machine.discretization.mesh.coordinates.shape[0],)
        or excitation.shape != (machine.winding_count,)
    ):
        raise ValueError(
            "Potential or current vector has an incompatible machine shape."
        )
    _, areas, _, curls = _geometry(machine, parameters, angle_delta)
    reluctivity, remanence, windings = _material_arrays(
        machine, parameters, angle_delta
    )
    return _field_quantities(
        machine,
        values,
        areas,
        curls,
        reluctivity,
        remanence,
        windings,
        excitation,
    )[-1]


def solve_planar_machine(
    machine: PlanarMachine,
    currents: ArrayLike,
    *,
    design: ArrayLike | None = None,
    angle_delta: ArrayLike = 0.0,
    boundary_potential: ArrayLike = 0.0,
    policy: MachineSolvePolicy | None = None,
) -> MachineFieldResult:
    """Solve -div(nu grad Az)=Jz+remanence weak source with fixed boundary Az.

    No penalty physics or failed-solve substitution is used. Invalid mesh or
    excitation raises; numerical evidence remains explicit in ``accepted``.
    """
    if not isinstance(machine, PlanarMachine):
        raise TypeError("machine must be a prepared PlanarMachine.")
    controls = MachineSolvePolicy() if policy is None else policy
    if not isinstance(controls, MachineSolvePolicy):
        raise TypeError("policy must be MachineSolvePolicy or None.")
    parameters = _design_vector(machine, design)
    excitation = jnp.asarray(currents)
    boundary = jnp.asarray(boundary_potential)
    if (
        excitation.shape != (machine.winding_count,)
        or boundary.shape != ()
    ):
        raise ValueError(
            "Currents must match winding count and boundary potential must be scalar."
        )
    excitation = eqx.error_if(
        excitation,
        jnp.any(~jnp.isfinite(excitation)),
        "Winding currents must be finite.",
    )
    boundary = eqx.error_if(
        boundary, ~jnp.isfinite(boundary), "Boundary potential must be finite."
    )
    points, areas, gradients, curls = _geometry(machine, parameters, angle_delta)
    reluctivity, remanence, windings = _material_arrays(
        machine, parameters, angle_delta
    )
    cells = machine.discretization.mesh.blocks[0].vertices
    local_matrix = contract(
        "c,c,cid,cjd->cij", areas, reluctivity, curls, curls
    )
    current_density = contract("ck,k->c", windings, excitation)
    local_source = contract(
        "c,c,cid,cd->ci", areas, reluctivity, curls, remanence
    )
    local_source = local_source + (areas * current_density / 3.0)[:, None]
    count = points.shape[0]
    matrix = (
        jnp.zeros((count, count), dtype=points.dtype)
        .at[cells[:, :, None], cells[:, None, :]]
        .add(local_matrix)
    )
    source = jnp.zeros((count,), dtype=points.dtype).at[cells].add(local_source)
    free = machine.free_nodes
    reduced = matrix[free[:, None], free[None, :]]
    prescribed = jnp.zeros_like(source).at[machine.boundary_nodes].set(boundary)
    right_hand_side = (
        source - contract("ij,j->i", matrix, prescribed)
    )[free]
    space = ArraySpace((free.shape[0],), dtype=reduced.dtype)
    operator = DenseLinearOperator(
        reduced,
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    linear = factorize(
        operator, FactorizationPolicy("cholesky")
    ).solve(right_hand_side)
    potential = prescribed.at[free].set(linear.value)
    (
        magnetic_field,
        field_strength,
        current_density,
        flux,
        energy,
        coenergy,
    ) = _field_quantities(
        machine,
        potential,
        areas,
        curls,
        reluctivity,
        remanence,
        windings,
        excitation,
    )
    # Envelope derivative of the stationary constitutive potential. Az stays
    # fixed for this partial derivative, but retains its implicit design tangent
    # when an enclosing differentiation asks for torque design sensitivities.
    torque = jax.grad(
        lambda delta: machine_coenergy(
            machine,
            potential,
            excitation,
            design=parameters,
            angle_delta=delta,
        )
    )(jnp.asarray(angle_delta, dtype=points.dtype))
    identity = jnp.eye(2, dtype=magnetic_field.dtype)
    magnetic_norm_squared = contract(
        "ci,ci->c", magnetic_field, magnetic_field
    )
    stress = (
        contract("ci,cj->cij", magnetic_field, magnetic_field)
        - 0.5 * magnetic_norm_squared[:, None, None] * identity
    ) / VACUUM_PERMEABILITY
    velocity = machine.rotation_weights[:, None] * jnp.stack(
        (-points[:, 1], points[:, 0]), axis=-1
    )
    velocity_gradient = contract("cid,cij->cdj", velocity[cells], gradients)
    stress_torque = -machine.axial_length * contract(
        "c,c,cij,cij->",
        areas,
        machine.airgap_cells.astype(areas.dtype),
        stress,
        velocity_gradient,
    )
    contour_torque = None
    if machine.contour_edges.shape[0]:
        endpoints = points[machine.contour_edges]
        tangent = endpoints[:, 1] - endpoints[:, 0]
        normal_measure = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
        mean_stress = jnp.mean(stress[machine.contour_cells], axis=1)
        traction = contract("eij,ej->ei", mean_stress, normal_measure)
        midpoint = jnp.mean(endpoints, axis=1)
        contour_torque = machine.axial_length * jnp.sum(
            midpoint[:, 0] * traction[:, 1]
            - midpoint[:, 1] * traction[:, 0]
        )
    residual = contract("ij,j->i", reduced, linear.value) - right_hand_side
    residual_norm = jnp.sqrt(contract("i,i->", residual, residual))
    source_norm = jnp.sqrt(contract("i,i->", right_hand_side, right_hand_side))
    relative_residual = residual_norm / jnp.maximum(
        source_norm, jnp.finfo(points.dtype).tiny
    )
    gauge_error = jnp.max(
        jnp.abs(potential[machine.boundary_nodes] - boundary)
    )
    discrepancy = jnp.abs(torque - stress_torque)
    linear_accepted = jnp.all(linear.successful) & (
        residual_norm
        <= controls.residual_absolute + controls.residual_relative * source_norm
    )
    contour_finite = (
        jnp.asarray(True)
        if contour_torque is None
        else jnp.isfinite(contour_torque)
    )
    accepted = (
        linear_accepted
        & (gauge_error == 0.0)
        & jnp.all(jnp.isfinite(potential))
        & jnp.all(jnp.isfinite(magnetic_field))
        & jnp.all(jnp.isfinite(field_strength))
        & jnp.all(jnp.isfinite(current_density))
        & jnp.all(jnp.isfinite(flux))
        & jnp.isfinite(energy)
        & jnp.isfinite(coenergy)
        & jnp.isfinite(torque)
        & jnp.isfinite(stress_torque)
        & contour_finite
        & (
            discrepancy
            <= controls.torque_absolute
            + controls.torque_relative
            * jnp.maximum(jnp.abs(torque), jnp.abs(stress_torque))
        )
    )
    return MachineFieldResult(
        potential=potential,
        coordinates=points,
        magnetic_field=magnetic_field,
        field_strength=field_strength,
        current_density=current_density,
        flux_linkage=flux,
        energy=energy,
        coenergy=coenergy,
        torque=torque,
        stress_torque=stress_torque,
        contour_torque=contour_torque,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        gauge_error=gauge_error,
        torque_discrepancy=discrepancy,
        linear_accepted=linear_accepted,
        accepted=accepted,
    )
