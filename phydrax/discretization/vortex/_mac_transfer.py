#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import ParticleDiscretization
from ...discretization.splatting import (
    ParticleGridSplatPlan,
    TensorBSplineSplatAssignment,
)
from ...equations._mac_incompressible import CompiledMACIncompressibleDynamics
from ...operators.integral.vortex._gaussian2d import gaussian_vortex_velocity_2d
from ...operators.integral.vortex._gaussian3d import GaussianErfVortexKernel3D
from ._source import VortexSourceState


class MACVortexTransferEvidence(StrictModule):
    deposited_strength: Array
    recovered_strength: Array
    circulation_residual: Array
    first_moment_residual: Array
    divergence_norm: Array
    transfer_successful: Array
    finite: Array


class MACVortexGridState(StrictModule):
    vorticity: Array
    velocity_state: Array
    evidence: MACVortexTransferEvidence
    transfer_id: str = eqx.field(static=True)


class MACVortexParticleTransferPlan(StrictModule, NonTrainableState):
    particles: ParticleDiscretization
    dynamics: CompiledMACIncompressibleDynamics
    degree: int = eqx.field(static=True)
    transfer: object
    dimension: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        dynamics: CompiledMACIncompressibleDynamics,
        /,
        *,
        degree: int = 2,
    ):
        if (
            not isinstance(particles, ParticleDiscretization)
            or not isinstance(dynamics, CompiledMACIncompressibleDynamics)
            or int(degree) not in (1, 2, 3)
        ):
            raise ValueError(
                "MAC vortex transfer requires particles, dynamics, and degree 1-3."
            )
        dimension = particles.ambient_dimension
        grid = dynamics.momentum.operators.discretization.grid
        if len(grid.axis_names) != dimension:
            raise ValueError("MAC grid and particle dimensions differ.")
        location = grid.centered_location
        transfer = ParticleGridSplatPlan(
            grid,
            location=location,
            assignment=TensorBSplineSplatAssignment(int(degree)),
            boundary="reject",
        ).prepare(particles)
        self.particles, self.dynamics, self.degree, self.transfer, self.dimension = (
            particles,
            dynamics,
            int(degree),
            transfer,
            dimension,
        )
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "mac-vortex-particle-transfer",
                "particles": particles.prepared_id,
                "dynamics": dynamics.compilation_id,
                "degree": int(degree),
                "transfer": transfer.prepared_id,
            }
        )

    def deposit(self, source: VortexSourceState, /) -> tuple[Array, object]:
        if (
            source.capacity != self.particles.capacity
            or source.dimension != self.dimension
        ):
            raise ValueError("MAC vortex source does not match particle support.")
        state = self.transfer.build(source.safe_positions())
        deposition = self.transfer.deposit_content(state, source.safe_strength())
        return deposition.density, state

    def _layout_points(self, layout) -> Array:
        mesh = jnp.meshgrid(*layout.coordinates_by_axis, indexing="ij")
        return jnp.stack(tuple(component.reshape(-1) for component in mesh), axis=-1)

    def vorticity_to_velocity(self, vorticity: Array, /) -> Array:
        operators = self.dynamics.momentum.operators
        cell_layout = operators.discretization.grid.cells()
        if vorticity.shape != cell_layout.shape + (() if self.dimension == 2 else (3,)):
            raise ValueError("MAC vorticity shape is incompatible.")
        cell_points = self._layout_points(cell_layout)
        cell_measure = cell_layout.measure.reshape(-1)
        strength = vorticity.reshape(
            (cell_points.shape[0],) + (() if self.dimension == 2 else (3,))
        ) * (cell_measure if self.dimension == 2 else cell_measure[:, None])
        widths = tuple(
            axis.interval_widths for axis in operators.discretization.grid.structured_axes
        )
        core_value = sum(jnp.mean(width) for width in widths) / self.dimension
        core = jnp.full((cell_points.shape[0],), core_value, dtype=cell_points.dtype)
        faces = []
        for layout in operators.discretization.face_layouts:
            target = self._layout_points(layout)
            displacement = target[:, None, :] - cell_points[None, :, :]
            if self.dimension == 2:
                pair_shape = displacement.shape[:-1]
                pair = gaussian_vortex_velocity_2d(
                    displacement,
                    jnp.broadcast_to(strength[None, :], pair_shape),
                    jnp.broadcast_to(core[None, :], pair_shape),
                )
            else:
                kernel = GaussianErfVortexKernel3D()
                pair = kernel.evaluate(
                    displacement,
                    jnp.broadcast_to(strength[None, :, :], displacement.shape),
                    jnp.broadcast_to(core[None, :], displacement.shape[:-1]),
                ).velocity
            vector_velocity = jnp.sum(pair, axis=1)
            axis = len(faces)
            faces.append(vector_velocity[:, axis].reshape(layout.shape))
        projected = self.dynamics.project_state(tuple(faces))
        return projected

    def velocity_to_vorticity(self, velocity_state: Array, /) -> Array:
        velocity = self.dynamics.unpack_velocity(velocity_state)
        axes = self.dynamics.momentum.operators.discretization.grid.structured_axes
        centered = []
        for axis, component in enumerate(velocity):
            if axes[axis].periodic:
                centered.append(0.5 * (component + jnp.roll(component, -1, axis=axis)))
            else:
                lower = [slice(None)] * self.dimension
                upper = [slice(None)] * self.dimension
                lower[axis] = slice(0, -1)
                upper[axis] = slice(1, None)
                centered.append(0.5 * (component[tuple(lower)] + component[tuple(upper)]))
        spacing = tuple(float(jnp.mean(axis.interval_widths)) for axis in axes)
        if self.dimension == 2:
            du_dy = jnp.gradient(centered[0], spacing[1], axis=1)
            dv_dx = jnp.gradient(centered[1], spacing[0], axis=0)
            return dv_dx - du_dy
        derivatives = tuple(
            tuple(
                jnp.gradient(centered[component], spacing[axis], axis=axis)
                for axis in range(3)
            )
            for component in range(3)
        )
        return jnp.stack(
            (
                derivatives[2][1] - derivatives[1][2],
                derivatives[0][2] - derivatives[2][0],
                derivatives[1][0] - derivatives[0][1],
            ),
            axis=-1,
        )

    def gather(
        self, transfer_state, vorticity: Array, source: VortexSourceState, /
    ) -> VortexSourceState:
        gathered = self.transfer.gather(transfer_state, vorticity).values
        if source.volume is None:
            raise ValueError("Particle recovery requires source volume.")
        strength = gathered * (
            source.safe_volume() if self.dimension == 2 else source.safe_volume()[:, None]
        )
        return VortexSourceState(
            source.positions,
            strength,
            core_radius=source.core_radius,
            volume=source.volume,
            active_mask=source.active_mask,
            source_kind=source.source_kind,
            source_id=source.source_id,
        )


__all__ = [
    "MACVortexGridState",
    "MACVortexParticleTransferPlan",
    "MACVortexTransferEvidence",
]
