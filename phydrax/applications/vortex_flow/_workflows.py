#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.vortex import (
    VortexFilamentState,
    VortexFilamentTopology,
    VortexParticleStateLayout,
)
from ...solver import WienerTerm


class PassiveVortexProbes(StrictModule, NonTrainableState):
    position: Array
    probes_id: str = eqx.field(static=True)

    def __init__(self, position: ArrayLike, /):
        points = jnp.asarray(position, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] not in (2, 3)
            or not bool(jnp.all(jnp.isfinite(points)))
        ):
            raise ValueError("Passive probe positions require finite shape (N,2|3).")
        self.position = points
        self.probes_id = canonical_fingerprint(
            {
                "kind": "passive-vortex-probes",
                "count": int(points.shape[0]),
                "dimension": int(points.shape[1]),
            }
        )

    def sample(self, prepared_velocity, source_position, strength, core_radius, /):
        return prepared_velocity.evaluate(
            source_position,
            strength,
            core_radius,
            targets=self.position,
        )


def actuator_line_sources(
    position: ArrayLike,
    circulation: ArrayLike,
    core_radius: ArrayLike,
    /,
) -> VortexFilamentState:
    points = jnp.asarray(position)
    gamma = jnp.asarray(circulation, dtype=points.dtype)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError("Actuator line positions require shape (vertices >= 2, 3).")
    segment_count = int(points.shape[0] - 1)
    if gamma.shape == ():
        gamma = jnp.full((segment_count,), gamma, dtype=points.dtype)
    if gamma.shape != (segment_count,):
        raise ValueError("Actuator line circulation must be scalar or per segment.")
    core = jnp.asarray(core_radius, dtype=points.dtype)
    if core.shape == ():
        core = jnp.full((segment_count,), core, dtype=points.dtype)
    topology = VortexFilamentTopology.from_segments(
        tuple((index, index + 1) for index in range(segment_count)),
        vertex_capacity=int(points.shape[0]),
    )
    return VortexFilamentState(topology, points, gamma, core)


def actuator_surface_sources(
    panel_vertices: ArrayLike,
    circulation: ArrayLike,
    core_radius: ArrayLike,
    /,
) -> VortexFilamentState:
    panels = jnp.asarray(panel_vertices)
    gamma = jnp.asarray(circulation, dtype=panels.dtype)
    if panels.ndim != 3 or panels.shape[1:] != (4, 3):
        raise ValueError("Actuator surface panels require shape (panels,4,3).")
    if gamma.shape != (panels.shape[0],):
        raise ValueError("Actuator surface circulation requires one value per panel.")
    vertices = panels.reshape((-1, 3))
    segments = []
    strengths = []
    for panel in range(int(panels.shape[0])):
        base = 4 * panel
        for local in range(4):
            segments.append((base + local, base + (local + 1) % 4))
            strengths.append(gamma[panel])
    topology = VortexFilamentTopology.from_segments(
        tuple(segments),
        vertex_capacity=int(vertices.shape[0]),
    )
    core = jnp.asarray(core_radius, dtype=panels.dtype)
    if core.shape == ():
        core = jnp.full((len(segments),), core, dtype=panels.dtype)
    return VortexFilamentState(topology, vertices, jnp.stack(tuple(strengths)), core)


class VortexRigidMotionState(StrictModule):
    position: Array
    velocity: Array
    angle: Array
    angular_velocity: Array


class PrescribedVortexRigidMotion(StrictModule, NonTrainableState):
    law: Callable[[Array, Any], VortexRigidMotionState]
    law_id: str = eqx.field(static=True)

    def __init__(
        self, law: Callable[[Array, Any], VortexRigidMotionState], law_id: str, /
    ):
        if not callable(law) or not str(law_id):
            raise ValueError("Prescribed motion requires callable law and stable ID.")
        self.law = law
        self.law_id = str(law_id)

    def evaluate(self, time: ArrayLike, args: Any = None, /) -> VortexRigidMotionState:
        state = self.law(jnp.asarray(time), args)
        if not isinstance(state, VortexRigidMotionState):
            raise TypeError("Prescribed motion law must return VortexRigidMotionState.")
        return state


class CoupledVortexRigidMotion(StrictModule, NonTrainableState):
    mass: float = eqx.field(static=True)
    inertia: float = eqx.field(static=True)
    motion_id: str = eqx.field(static=True)

    def __init__(self, mass: float, inertia: float, /):
        if mass <= 0.0 or inertia <= 0.0:
            raise ValueError("Coupled rigid mass/inertia must be positive.")
        self.mass = float(mass)
        self.inertia = float(inertia)
        self.motion_id = canonical_fingerprint(
            {
                "kind": "coupled-vortex-rigid-motion",
                "mass": self.mass,
                "inertia": self.inertia,
            }
        )

    def step(
        self,
        state: VortexRigidMotionState,
        force: ArrayLike,
        torque: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> VortexRigidMotionState:
        force_ = jnp.asarray(force, dtype=state.position.dtype)
        torque_ = jnp.asarray(torque, dtype=state.position.dtype)
        dt = jnp.asarray(time_step, dtype=state.position.dtype)
        if force_.shape != state.position.shape or torque_.shape != () or dt.shape != ():
            raise ValueError("Coupled rigid load/time shapes are invalid.")
        velocity = state.velocity + dt * force_ / self.mass
        angular_velocity = state.angular_velocity + dt * torque_ / self.inertia
        return VortexRigidMotionState(
            state.position + dt * velocity,
            velocity,
            state.angle + dt * angular_velocity,
            angular_velocity,
        )


class RandomVortexDiffusion(StrictModule, NonTrainableState):
    viscosity: float = eqx.field(static=True)
    noise_name: str = eqx.field(static=True)
    diffusion_id: str = eqx.field(static=True)

    def __init__(
        self, viscosity: float, /, *, noise_name: str = "vortex-brownian-diffusion"
    ):
        if viscosity <= 0.0 or not str(noise_name):
            raise ValueError(
                "Random vortex diffusion requires positive viscosity and a name."
            )
        self.viscosity = float(viscosity)
        self.noise_name = str(noise_name)
        self.diffusion_id = canonical_fingerprint(
            {
                "kind": "random-vortex-diffusion",
                "viscosity": self.viscosity,
                "noise_name": self.noise_name,
            }
        )

    def wiener_term(self, layout: VortexParticleStateLayout, /) -> WienerTerm:
        if not isinstance(layout, VortexParticleStateLayout):
            raise TypeError("layout must be VortexParticleStateLayout.")
        coefficient = (
            jnp.zeros((layout.state_size,), dtype=float)
            .at[: layout.position_size]
            .set(jnp.sqrt(2.0 * self.viscosity))
        )

        def diagonal(time, state, args):
            del time, args
            return coefficient.astype(jnp.asarray(state).dtype)

        return WienerTerm(
            self.noise_name,
            diagonal,
            (layout.state_size,),
            structure="additive",
            basis_id=self.diffusion_id,
            representation="diagonal",
        )


class LearnedVorticityResult(StrictModule):
    model: Any
    vorticity: Array
    velocity: Array
    finite: Array
    workflow_id: str = eqx.field(static=True)


class LearnedVorticityWorkflow(StrictModule, NonTrainableState):
    trainer: Callable[[Array, Array, Any], Any]
    evaluator: Callable[[Any, Array], ArrayLike]
    velocity_reconstruction: Callable[[Array, Array, Any], ArrayLike]
    workflow_id: str = eqx.field(static=True)

    def __init__(
        self, trainer, evaluator, velocity_reconstruction, /, *, workflow_id: str
    ):
        if (
            not callable(trainer)
            or not callable(evaluator)
            or not callable(velocity_reconstruction)
            or not str(workflow_id)
        ):
            raise ValueError(
                "Learned vorticity workflow requires real callbacks and stable ID."
            )
        self.trainer = trainer
        self.evaluator = evaluator
        self.velocity_reconstruction = velocity_reconstruction
        self.workflow_id = str(workflow_id)

    def fit_and_reconstruct(
        self,
        sample_position: ArrayLike,
        sample_weight: ArrayLike,
        targets: ArrayLike,
        args: Any = None,
        /,
    ) -> LearnedVorticityResult:
        samples = jnp.asarray(sample_position)
        weights = jnp.asarray(sample_weight)
        target = jnp.asarray(targets)
        if (
            samples.ndim != 2
            or weights.shape != samples.shape[:1]
            or target.ndim != 2
            or target.shape[1] != samples.shape[1]
        ):
            raise ValueError("Learned-vorticity sample/target shapes are invalid.")
        model = self.trainer(samples, weights, args)
        vorticity = jnp.asarray(self.evaluator(model, target))
        velocity = jnp.asarray(self.velocity_reconstruction(vorticity, target, args))
        finite = jnp.all(jnp.isfinite(vorticity)) & jnp.all(jnp.isfinite(velocity))
        return LearnedVorticityResult(
            model, vorticity, velocity, finite, self.workflow_id
        )


__all__ = [
    "CoupledVortexRigidMotion",
    "LearnedVorticityResult",
    "LearnedVorticityWorkflow",
    "PassiveVortexProbes",
    "PrescribedVortexRigidMotion",
    "RandomVortexDiffusion",
    "VortexRigidMotionState",
    "actuator_line_sources",
    "actuator_surface_sources",
]
