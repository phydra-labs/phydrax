#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lorentz-boosted-frame electromagnetic PIC over Galilean PSATD.

A boosted-frame run (Vay 2007) simulates a lab-frame interaction — a laser
wakefield stage, an undulator — in a frame moving with ``v_b = β_b c ê`` along
one grid axis ``ê``. Lab plasma at rest becomes a plasma of density ``γ_b n``
drifting at ``−v_b``; laser wavelengths stretch by ``γ_b(1 + β_b)`` while plasma
lengths contract by ``γ_b``, so the scale range of the problem, and with it the
number of cells and steps, shrinks by roughly ``γ_b²(1 + β_b)``.

`BoostedFramePlan(frame, lab_domain)` owns the frame (a pure boost
`phydrax.LorentzFrame` along one axis, the passive map from lab to boosted
four-vectors) and the lab-frame region of interest. Its transforms are exact
Lorentz transforms: particles through `phydrax.boost_proper_velocity` and a
ballistic drift onto one boosted time slice; prescribed fields gather-only
through `BoostedExternalField` (`phydrax.boost_fields` of the lab source at
each particle's lab event; never deposited, so a static undulator, whose
boosted field is a sub-luminal, magnetic-type (``E² < c²B²``) evanescent wave,
never radiates on the grid); sheet antennas through a moving B5
`SampledPlaneCurrentAntennaPlan`.

``prepare(pic, ...)`` binds one `ElectromagneticPICPlan` whose field solver is
a `PreparedSpectralMaxwell`. The field grid is either Galilean, comoving with
the boosted lab plasma (``v_gal = −v_b``: the NCI of the drifting plasma is
suppressed and lab plasma is at rest in grid coordinates, ``z_lab = γ_b z``),
or standard (vacuum and beam-only runs; plasma drifting at ``−v_b`` through a
standard grid is refused at initialization). The numerical-Cherenkov guard is
mandatory: every step samples the high-``|k|`` shell energy ``W_k`` of the
candidate field with `SpectralNCIMonitorPlan` and rejects the step when
``W_k`` has grown beyond ``nci_growth_limit`` times its reference (the initial
high-``|k|`` energy, or the first nonzero one) while also exceeding
``nci_energy_fraction`` of the electromagnetic energy — the exponential
high-``|k|`` growth that characterizes the instability.

Back-transformed diagnostics. A lab snapshot at time ``T`` is assembled from
the boosted run: lab sample plane ``z`` is reached at boosted time
``t′ = γ_b(T − β_b z/c)``; the fields of the two bracketing boosted steps are
interpolated linearly in time (and along the axis in grid coordinates) and
Lorentz transformed back. Particles are recorded when their lab time crosses
``T`` inside the lab domain. Snapshots live in a bounded ring of
``ring_capacity`` slots (snapshot ``s`` in slot ``s mod K``); the plan refuses
a ring whose next occupant of a slot would start before its previous occupant
completes, and a completed snapshot stays resident until its slot's next
occupant starts. Track recorders are converted to lab `ChargedTrajectory`
lanes with per-lane lab times for the trajectory-radiation route (A1).

A boosted Huygens far field is admitted only for a standard (non-Galilean)
grid whose observers saw ``J = 0`` on the surface at every attempted step and
whose acquisition window closed, in vacuum; the boosted spectrum is relabeled
into the lab through `phydrax.transform_spectral_energy` (complete emission).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._lorentz import (
    boost_event,
    boost_fields,
    boost_matrix,
    boost_proper_velocity,
    LorentzFrame,
    LorentzSpectralTransform,
    SpectralEmissionCompleteness,
    transform_spectral_energy,
)
from .._physical import ElectromagneticScaleContract, RelativityScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge
from ..discretization.pic import (
    ExternalFieldSample,
    ExternalFieldSource,
    PIC_CODE_RELATIVITY,
    PICTrackRecorder,
)
from ..electromagnetics import ChargedTrajectory
from ..geometry import Box
from ..typing import checked
from ._electromagnetic_pic import (
    ElectromagneticPICPlan,
    ElectromagneticPICState,
    ElectromagneticPICStepResult,
    PICFieldHistory,
    PICRestartCheckpoint,
)
from ._maxwell_antenna import SampledPlaneCurrentAntennaPlan
from ._maxwell_far_field import MaxwellFarFieldResult
from ._pic_field_solver import (
    PICRestartComponent,
    restart_component,
    restore_component,
)
from .maxwell.spectral._nci import SpectralNCIMonitorPlan, SpectralNCISample
from .maxwell.spectral._psatd import (
    _component_offsets,
    PreparedSpectralMaxwell,
    SpectralMaxwellDiagnostics,
    SpectralMaxwellState,
)


_RESTART_NAME = "boosted-frame"
_FRAME_TOLERANCE = 1.0e-12


def _frame_boost(frame: LorentzFrame, /) -> tuple[int, float]:
    """Axis and signed speed of a pure boost along one Cartesian axis."""
    matrix = np.asarray(frame.matrix, dtype=np.float64)
    beta = -matrix[0, 1:] / matrix[0, 0]
    expected = np.asarray(boost_matrix(jnp.asarray(beta, dtype=jnp.float64)))
    if np.max(np.abs(matrix - expected)) > _FRAME_TOLERANCE * float(
        np.max(np.abs(matrix))
    ):
        raise ValueError(
            "frame must be a pure Lorentz boost; rotations and rotated boosts are "
            "refused."
        )
    moving = np.flatnonzero(beta != 0.0)
    if moving.size != 1:
        raise ValueError(
            "frame must boost along exactly one Cartesian axis with nonzero speed."
        )
    axis = int(moving[0])
    return axis, float(beta[axis])


class BoostedParticles(StrictModule):
    """Particles transformed onto one boosted time slice.

    ``positions`` are boosted-frame positions at ``time``; ``velocities`` and
    ``proper_velocities`` are boosted-frame physical and proper velocities.
    ``drift_intervals`` is the boosted time each particle was drifted
    ballistically from its own boosted event onto the slice (zero-field free
    streaming is assumed over that interval).
    """

    positions: Array
    velocities: Array
    proper_velocities: Array
    drift_intervals: Array
    time: Array


class BoostedFramePlan(StrictModule, NonTrainableState):
    """Pure Lorentz boost along one grid axis bound to a lab-frame region.

    ``frame`` maps lab four-vectors to boosted ones,
    ``frame.apply(p_lab) = p_boosted``: for a boosted frame moving with
    ``β_b ê`` relative to the lab this is ``LorentzFrame(boost_matrix(β_b ê))``
    (equivalently ``LorentzFrame.boost(−β_b ê)``). ``lab_domain`` is the lab
    region whose fields and particles the back-transformed diagnostics
    reconstruct; ``relativity`` fixes the units of positions, times, and the
    speed of light (the PIC pusher's relativity scale).
    """

    frame: LorentzFrame
    lab_domain: Box
    relativity: RelativityScaleContract = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    lorentz_factor: float = eqx.field(static=True)
    lab_lower: tuple[float, float, float] = eqx.field(static=True)
    lab_upper: tuple[float, float, float] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        frame: LorentzFrame,
        lab_domain: Box,
        /,
        *,
        relativity: RelativityScaleContract = PIC_CODE_RELATIVITY,
    ) -> None:
        axis, beta = _frame_boost(frame)
        center = np.asarray(lab_domain.center, dtype=np.float64)
        size = np.asarray(lab_domain.size, dtype=np.float64)
        if not (np.all(np.isfinite(center)) and np.all(np.isfinite(size))):
            raise ValueError("lab_domain must be finite.")
        lower = center - 0.5 * size
        upper = center + 0.5 * size
        self.frame = frame
        self.lab_domain = lab_domain
        self.relativity = relativity
        self.axis = axis
        self.beta = beta
        self.lorentz_factor = 1.0 / math.sqrt(1.0 - beta * beta)
        self.lab_lower = (float(lower[0]), float(lower[1]), float(lower[2]))
        self.lab_upper = (float(upper[0]), float(upper[1]), float(upper[2]))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "boosted-frame-plan",
                "frame": frame.frame_id,
                "lab_lower": list(self.lab_lower),
                "lab_upper": list(self.lab_upper),
                "relativity": relativity.scale_id,
            }
        )

    # -- kinematics ---------------------------------------------------------------

    @property
    def speed_of_light(self) -> float:
        return float(self.relativity.speed_of_light)

    @property
    def boost_velocity(self) -> tuple[float, float, float]:
        """Velocity ``β_b c ê`` of the boosted frame in the lab."""
        value = [0.0, 0.0, 0.0]
        value[self.axis] = self.beta * self.speed_of_light
        return (value[0], value[1], value[2])

    @property
    def galilean_velocity(self) -> tuple[float, float, float]:
        """Drift ``−β_b c ê`` of lab-resting plasma in the boosted frame.

        A Galilean PSATD grid comoving with the boosted plasma declares this
        ``galilean_velocity``.
        """
        velocity = self.boost_velocity
        return (-velocity[0], -velocity[1], -velocity[2])

    @property
    def grid_lower(self) -> tuple[float, float, float]:
        """Lower corner of ``lab_domain`` in comoving Galilean grid coordinates."""
        return self._contracted(self.lab_lower)

    @property
    def grid_upper(self) -> tuple[float, float, float]:
        """Upper corner of ``lab_domain`` in comoving Galilean grid coordinates."""
        return self._contracted(self.lab_upper)

    def _contracted(
        self, point: tuple[float, float, float], /
    ) -> tuple[float, float, float]:
        value = list(point)
        value[self.axis] = value[self.axis] / self.lorentz_factor
        return (value[0], value[1], value[2])

    def _beta(self, sign: float, /) -> Array:
        return jnp.zeros((3,), dtype=jnp.float64).at[self.axis].set(sign * self.beta)

    def to_lab(self, times: ArrayLike, positions: ArrayLike, /) -> tuple[Array, Array]:
        """Lab events ``(t, x)`` of boosted events ``(t′, x′[..., 3])``."""
        return self._transform(-1.0, times, positions)

    def from_lab(self, times: ArrayLike, positions: ArrayLike, /) -> tuple[Array, Array]:
        """Boosted events ``(t′, x′[..., 3])`` of lab events ``(t, x)``."""
        return self._transform(1.0, times, positions)

    def _transform(
        self, sign: float, times: ArrayLike, positions: ArrayLike, /
    ) -> tuple[Array, Array]:
        position = jnp.asarray(positions, dtype=jnp.float64)
        if position.shape[-1:] != (3,):
            raise ValueError("Event positions require a trailing axis of length three.")
        time = jnp.broadcast_to(
            jnp.asarray(times, dtype=jnp.float64), position.shape[:-1]
        )
        c = self.speed_of_light
        event = jnp.concatenate((c * time[..., None], position), axis=-1)
        boosted = boost_event(self._beta(sign), event)
        return boosted[..., 0] / c, boosted[..., 1:]

    def boost_fields(
        self, electric: ArrayLike, magnetic: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Boosted ``(E′, B′)`` of lab fields at the same event."""
        return boost_fields(
            self._beta(1.0),
            jnp.asarray(electric, dtype=jnp.float64),
            jnp.asarray(magnetic, dtype=jnp.float64),
            speed_of_light=jnp.asarray(self.speed_of_light, dtype=jnp.float64),
        )

    def lab_fields(
        self, electric: ArrayLike, magnetic: ArrayLike, /
    ) -> tuple[Array, Array]:
        """Lab ``(E, B)`` of boosted fields at the same event."""
        return boost_fields(
            self._beta(-1.0),
            jnp.asarray(electric, dtype=jnp.float64),
            jnp.asarray(magnetic, dtype=jnp.float64),
            speed_of_light=jnp.asarray(self.speed_of_light, dtype=jnp.float64),
        )

    def boost_proper_velocity(self, proper_velocity: ArrayLike, /) -> Array:
        """Boosted proper velocity ``u′ = γ′v′`` of lab ``u`` (velocity units)."""
        c = self.speed_of_light
        return c * boost_proper_velocity(
            self._beta(1.0), jnp.asarray(proper_velocity, dtype=jnp.float64) / c
        )

    def lab_proper_velocity(self, proper_velocity: ArrayLike, /) -> Array:
        """Lab proper velocity of boosted ``u′`` (velocity units)."""
        c = self.speed_of_light
        return c * boost_proper_velocity(
            self._beta(-1.0), jnp.asarray(proper_velocity, dtype=jnp.float64) / c
        )

    def boost_particles(
        self,
        lab_positions: ArrayLike,
        lab_velocities: ArrayLike,
        /,
        *,
        boosted_time: ArrayLike,
        lab_times: ArrayLike = 0.0,
    ) -> BoostedParticles:
        """Transform lab particles onto the boosted slice ``t′ = boosted_time``.

        Each particle's lab event ``(lab_times, x)`` maps to its own boosted event
        ``(t′_e, x′_e)``; the proper velocity is boosted exactly and the particle
        drifts ballistically to the common slice,
        ``x′ = x′_e + v′ (boosted_time − t′_e)``. Lab-resting plasma becomes the
        boosted plasma of density ``γ_b n`` (its ``z`` spacing contracts by
        ``γ_b`` at unchanged macroparticle weights) drifting at ``−v_b``; beams
        receive the relativistic velocity addition. The ballistic drift assumes
        free streaming between the particle's event and the slice: load beams
        in vacuum.
        """
        position = jnp.asarray(lab_positions, dtype=jnp.float64)
        velocity = jnp.asarray(lab_velocities, dtype=jnp.float64)
        if (
            position.ndim != 2
            or position.shape[1] != 3
            or velocity.shape != position.shape
        ):
            raise ValueError("lab positions and velocities must have shape (N, 3).")
        c = self.speed_of_light
        speed2 = jnp.sum(velocity * velocity, axis=-1)
        velocity = eqx.error_if(
            velocity,
            ~jnp.all(jnp.isfinite(velocity)) | jnp.any(speed2 >= c * c),
            "Boosted particles need finite subluminal lab velocities.",
        )
        proper = velocity / jnp.sqrt(1.0 - speed2 / (c * c))[:, None]
        boosted_proper = self.boost_proper_velocity(proper)
        gamma = jnp.sqrt(1.0 + jnp.sum(boosted_proper**2, axis=-1) / (c * c))
        boosted_velocity = boosted_proper / gamma[:, None]
        event_time, event_position = self.from_lab(lab_times, position)
        slice_time = jnp.asarray(boosted_time, dtype=jnp.float64).reshape(())
        interval = slice_time - event_time
        return BoostedParticles(
            event_position + interval[:, None] * boosted_velocity,
            boosted_velocity,
            boosted_proper,
            interval,
            slice_time,
        )

    # -- prescribed fields and antennas ----------------------------------------------

    def boost_external_field(
        self, source: ExternalFieldSource, /
    ) -> BoostedExternalField:
        """Gather-only boosted view of a lab-frame `ExternalFieldSource`."""
        return BoostedExternalField(self, source)

    @checked
    def boost_antenna(
        self,
        antenna: SampledPlaneCurrentAntennaPlan,
        bridge: StructuredCochainBridge,
        /,
        *,
        scale: ElectromagneticScaleContract | None = None,
    ) -> SampledPlaneCurrentAntennaPlan:
        """Moving boosted-frame antenna of a lab antenna on a boosted ``bridge``.

        The antenna's rest frame and rest-frame samples are unchanged; its
        velocity composes relativistically with the boost, and its rest-frame
        origin moves to the event where the sheet crosses its boosted plane at
        boosted time zero. Requires the normal axis to be the boost axis and the
        vacuum of ``scale`` (the antenna's own scale by default).
        """
        if antenna.normal_axis != self.axis:
            raise ValueError(
                "Only antennas normal to the boost axis move along their normal in the "
                "boosted frame; the antenna's normal axis differs from the boost axis."
            )
        resolved = antenna.scale if scale is None else scale
        if not isinstance(resolved, ElectromagneticScaleContract):
            raise ValueError(
                "A boosted antenna moves; it needs an ElectromagneticScaleContract "
                "vacuum (pass scale= or bind the lab antenna to a scale)."
            )
        if float(resolved.speed_of_light) != self.speed_of_light:
            raise ValueError(
                "The antenna scale's speed of light differs from the frame's."
            )
        c = self.speed_of_light
        lab_beta = antenna.beta
        beta = (lab_beta - self.beta) / (1.0 - lab_beta * self.beta)
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        origin = np.zeros(3)
        origin[self.axis] = antenna.plane_coordinate
        time, position = self.from_lab(0.0, jnp.asarray(origin))
        origin_time = float(time)
        # Rest-frame (proper) time elapsed along the sheet from its old origin
        # event to its crossing of the boosted slice t′ = 0.
        proper_shift = -origin_time / gamma
        plane = float(position[self.axis]) - beta * c * origin_time
        return SampledPlaneCurrentAntennaPlan(
            bridge,
            self.axis,
            plane,
            antenna.first_coordinates,
            antenna.second_coordinates,
            antenna.times - proper_shift,
            antenna.electric,
            magnetic=antenna.magnetic,
            carrier_angular_frequency=antenna.carrier_angular_frequency,
            direction=antenna.direction,
            beta=beta,
            scale=resolved,
            provenance_id=antenna.provenance_id,
        )

    # -- runtime --------------------------------------------------------------------

    def prepare(
        self,
        pic: ElectromagneticPICPlan,
        /,
        *,
        snapshots: BoostedSnapshotPlan | None = None,
        nci_shell_count: int = 8,
        nci_high_fraction: float = 0.5,
        nci_growth_limit: float = 1.0e2,
        nci_energy_fraction: float = 1.0e-2,
    ) -> PreparedBoostedFrame:
        return PreparedBoostedFrame(
            self,
            pic,
            snapshots,
            nci_shell_count,
            nci_high_fraction,
            nci_growth_limit,
            nci_energy_fraction,
        )


class BoostedExternalField(StrictModule, NonTrainableState):
    """Lab-frame external field seen in the boosted frame; gather-only.

    Boosted events ``(t′, x′)`` map to lab events where ``source`` is sampled
    at per-particle lab times; the fields transform with ``F′ = Λ F Λᵀ`` and
    support is the lab support. Being an `ExternalFieldSource`, the field is
    only gathered by particle pushes and never deposited on the grid.
    """

    frame: BoostedFramePlan
    source: ExternalFieldSource
    source_id: str = eqx.field(static=True)

    @checked
    def __init__(self, frame: BoostedFramePlan, source: ExternalFieldSource, /) -> None:
        if not isinstance(source, ExternalFieldSource):
            raise TypeError("source must implement ExternalFieldSource.")
        self.frame = frame
        self.source = source
        self.source_id = canonical_fingerprint(
            {
                "kind": "boosted-external-field",
                "frame": frame.plan_id,
                "source": source.source_id,
            }
        )

    def external_fields(self, positions: Array, times: Array, /) -> ExternalFieldSample:
        lab_times, lab_positions = self.frame.to_lab(times, positions)
        sample = self.source.external_fields(lab_positions, lab_times)
        electric, magnetic = self.frame.boost_fields(sample.electric, sample.magnetic)
        return ExternalFieldSample(electric, magnetic, sample.support)


# -- back-transformed snapshots ----------------------------------------------------------


class BoostedSnapshotPlan(StrictModule, NonTrainableState):
    """Lab-frame snapshot request of a boosted run.

    ``lab_times`` are strictly increasing lab times; field snapshots are
    reconstructed on the lab images of the grid's node planes inside the lab
    domain, and particle snapshots for the run species listed in ``species``.
    ``ring_capacity`` bounds the resident snapshots (default: all).
    """

    lab_times: tuple[float, ...] = eqx.field(static=True)
    ring_capacity: int = eqx.field(static=True)
    species: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lab_times: Sequence[float],
        /,
        *,
        ring_capacity: int | None = None,
        species: Sequence[int] = (),
    ) -> None:
        times = tuple(float(value) for value in lab_times)
        if not times or not all(math.isfinite(value) for value in times):
            raise ValueError("lab_times must be a nonempty sequence of finite times.")
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("lab_times must increase strictly.")
        capacity = len(times) if ring_capacity is None else int(ring_capacity)
        if capacity < 1:
            raise ValueError("ring_capacity must be positive.")
        indices = tuple(int(value) for value in species)
        if len(set(indices)) != len(indices) or any(value < 0 for value in indices):
            raise ValueError("species must be distinct nonnegative species indices.")
        self.lab_times = times
        self.ring_capacity = min(capacity, len(times))
        self.species = indices
        self.plan_id = canonical_fingerprint(
            {
                "kind": "boosted-snapshot-plan",
                "lab_times": [value.hex() for value in times],
                "ring_capacity": self.ring_capacity,
                "species": list(indices),
            }
        )


class BoostedParticleSnapshotBuffer(StrictModule):
    """Ring rows ``[K, capacity]`` of one species' lab-frame particle snapshots."""

    positions: Array
    proper_velocities: Array
    id_hi: Array
    id_lo: Array
    filled: Array


class BoostedSnapshotState(StrictModule):
    """Resident lab snapshots: slot ``k`` holds snapshot ``owner[k]``.

    ``electric``/``magnetic`` are ``[K, Z, T₁, T₂, 3]`` lab fields on the lab
    sample planes (boost axis first, then the transverse axes in grid order);
    ``field_filled[K, Z]`` marks reconstructed planes. ``conflicts`` counts
    slot hand-overs that evicted an incomplete snapshot.
    """

    owner: Array
    electric: Array
    magnetic: Array
    field_filled: Array
    particles: tuple[BoostedParticleSnapshotBuffer, ...]
    conflicts: Array


class BoostedFrameEvidence(StrictModule):
    """Run evidence carried with the boosted state.

    ``nci_reference`` is the high-``|k|`` energy the NCI guard measures growth
    against; ``nci_growth`` and ``nci_fraction`` are the largest growth ratio
    and largest high-``|k|`` fraction of the electromagnetic energy over
    accepted fields; ``nci_rejections`` counts steps the guard rejected.
    ``surface_current`` is the largest current met on a Huygens surface at any
    attempted step. ``vacuum_divergence`` is the Gauss residual removed from the
    sampled initial vacuum fields by the solver's Poisson projection.
    """

    nci_reference: Array
    nci_growth: Array
    nci_fraction: Array
    nci_rejections: Array
    surface_current: Array
    vacuum_divergence: Array


class BoostedFrameState(StrictModule):
    pic: ElectromagneticPICState
    snapshots: BoostedSnapshotState | None
    evidence: BoostedFrameEvidence


class BoostedFrameStepResult(StrictModule):
    candidate_state: BoostedFrameState
    accepted_state: BoostedFrameState
    pic: ElectromagneticPICStepResult
    nci: SpectralNCISample
    nci_growth: Array
    nci_fraction: Array
    nci_admissible: Array
    successful: Array


class BoostedLabFieldSnapshot(StrictModule):
    """Back-transformed lab fields at ``lab_time``.

    Arrays follow the grid axis order with the boost axis replaced by the lab
    sample planes ``axis_coordinates``; ``transverse_coordinates`` are the node
    coordinates of the other two axes in increasing axis order. ``filled`` marks
    reconstructed planes (planes whose boosted time preceded the run start or
    has not been reached are unfilled).
    """

    lab_time: float = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    axis_coordinates: Array
    transverse_coordinates: tuple[Array, Array]
    electric: Array
    magnetic: Array
    filled: Array


class BoostedLabParticleSnapshot(StrictModule):
    """Lab-frame particles of one species crossing ``lab_time`` inside the lab domain."""

    lab_time: float = eqx.field(static=True)
    species: int = eqx.field(static=True)
    positions: Array
    proper_velocities: Array
    id_hi: Array
    id_lo: Array
    filled: Array


def _select(predicate: Array, candidate: Any, current: Any, /) -> Any:
    return jax.tree.map(
        lambda proposed, old: jnp.where(predicate, proposed, old), candidate, current
    )


def _initial_evidence(reference: Array, divergence: Array, /) -> BoostedFrameEvidence:
    zero = jnp.zeros((), dtype=jnp.float64)
    return BoostedFrameEvidence(
        nci_reference=jnp.asarray(reference, dtype=jnp.float64),
        nci_growth=zero,
        nci_fraction=zero,
        nci_rejections=jnp.zeros((), dtype=jnp.int32),
        surface_current=zero,
        vacuum_divergence=jnp.asarray(divergence, dtype=jnp.float64),
    )


class _SnapshotLayout(StrictModule, NonTrainableState):
    """Prepared schedule and sampling geometry of the snapshot ring."""

    lab_times: Array
    samples: Array
    owners: Array
    starts: Array
    completes: Array
    species: tuple[int, ...] = eqx.field(static=True)
    ring_capacity: int = eqx.field(static=True)


def _snapshot_layout(
    plan: BoostedSnapshotPlan,
    frame: BoostedFramePlan,
    solver: PreparedSpectralMaxwell,
    species_count: int,
    /,
) -> _SnapshotLayout:
    if any(index >= species_count for index in plan.species):
        raise ValueError("Snapshot species reference a species outside the run.")
    axis = frame.axis
    gamma = frame.lorentz_factor
    c = frame.speed_of_light
    nodes = solver.plan.origin[axis] + solver.plan.spacing[axis] * np.arange(
        solver.plan.counts[axis]
    )
    lower, upper = frame.lab_lower[axis], frame.lab_upper[axis]
    samples = gamma * nodes
    samples = samples[(samples >= lower) & (samples <= upper)]
    if samples.size == 0:
        raise ValueError("No grid node plane images into the lab domain.")
    times = np.asarray(plan.lab_times)
    edges = np.asarray([lower, upper])
    boosted = gamma * (times[:, None] - frame.beta * edges[None, :] / c)
    starts = np.min(boosted, axis=1)
    completes = np.max(boosted, axis=1)
    capacity = plan.ring_capacity
    count = times.size
    for index in range(count - capacity):
        if starts[index + capacity] < completes[index]:
            raise ValueError(
                f"Snapshot ring capacity {capacity} is too small: lab snapshot "
                f"{index + capacity} starts at boosted time "
                f"{starts[index + capacity]:.6g} before snapshot {index} completes at "
                f"{completes[index]:.6g}."
            )
    rows = -(-count // capacity)
    owners = np.zeros((capacity, rows), dtype=np.int32)
    table = np.full((capacity, rows), np.inf)
    for slot in range(capacity):
        members = np.arange(slot, count, capacity)
        owners[slot, : members.size] = members
        owners[slot, members.size :] = members[-1]
        table[slot, : members.size] = starts[members]
    return _SnapshotLayout(
        lab_times=jnp.asarray(times),
        samples=jnp.asarray(samples),
        owners=jnp.asarray(owners),
        starts=jnp.asarray(table),
        completes=jnp.asarray(completes),
        species=plan.species,
        ring_capacity=capacity,
    )


class PreparedBoostedFrame(StrictModule, NonTrainableState):
    """Boosted-frame PIC run: NCI-guarded steps and back-transformed diagnostics."""

    frame: BoostedFramePlan
    pic: ElectromagneticPICPlan
    nci: SpectralNCIMonitorPlan
    snapshot_layout: _SnapshotLayout | None
    grid_velocity: tuple[float, float, float] = eqx.field(static=True)
    galilean: bool = eqx.field(static=True)
    nci_growth_limit: float = eqx.field(static=True)
    nci_energy_fraction: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        frame: BoostedFramePlan,
        pic: ElectromagneticPICPlan,
        snapshots: BoostedSnapshotPlan | None,
        nci_shell_count: int,
        nci_high_fraction: float,
        nci_growth_limit: float,
        nci_energy_fraction: float,
        /,
    ) -> None:
        solver = pic.solver
        if not isinstance(solver, PreparedSpectralMaxwell):
            raise TypeError(
                "Boosted-frame PIC requires a PreparedSpectralMaxwell (PSATD) field "
                "solver: the Galilean scheme and the NCI guard are spectral."
            )
        if pic.pusher.relativity.scale_id != frame.relativity.scale_id:
            raise ValueError(
                "The PIC pusher's relativity scale differs from the frame's."
            )
        c = frame.speed_of_light
        if abs(solver.plan.speed_of_light - c) > _FRAME_TOLERANCE * c:
            raise ValueError(
                "The spectral solver's medium is not the vacuum of the frame's scale: "
                "Lorentz boosts of fields and spectra hold only in vacuum."
            )
        grid_velocity = solver.plan.galilean_velocity
        match solver.plan.variant:
            case "standard":
                galilean = False
            case "galilean" | "averaged-galilean":
                expected = np.asarray(frame.galilean_velocity)
                if np.max(np.abs(np.asarray(grid_velocity) - expected)) > (
                    _FRAME_TOLERANCE * c
                ):
                    raise ValueError(
                        "The Galilean grid must comove with the boosted lab plasma: "
                        f"galilean_velocity must be {frame.galilean_velocity}."
                    )
                galilean = True
            case _:
                raise ValueError("The spectral solver variant is invalid.")
        if galilean:
            _require_covered(frame, solver)
        for index, recorder in enumerate(pic.recorders):
            if isinstance(recorder, PICTrackRecorder) and recorder.radiation is not None:
                raise ValueError(
                    f"Recorder {index} streams trajectory radiation in the boosted "
                    "frame; radiate lab-frame tracks from lab_trajectory instead."
                )
        for index, source in enumerate(pic.external_fields):
            if not (
                isinstance(source, BoostedExternalField)
                and source.frame.plan_id == frame.plan_id
            ):
                raise ValueError(
                    f"External field {index} is not a lab source boosted through this "
                    "frame (BoostedFramePlan.boost_external_field)."
                )
        fraction = float(nci_energy_fraction)
        growth = float(nci_growth_limit)
        if not (math.isfinite(fraction) and 0.0 < fraction < 1.0):
            raise ValueError("nci_energy_fraction must lie in (0, 1).")
        if not (math.isfinite(growth) and growth > 1.0):
            raise ValueError("nci_growth_limit must be finite and exceed one.")
        monitor = SpectralNCIMonitorPlan(
            solver, shell_count=nci_shell_count, high_fraction=nci_high_fraction
        )
        layout = None
        if snapshots is not None:
            if not isinstance(snapshots, BoostedSnapshotPlan):
                raise TypeError("snapshots must be a BoostedSnapshotPlan or None.")
            layout = _snapshot_layout(snapshots, frame, solver, len(pic.species))
        self.frame = frame
        self.pic = pic
        self.nci = monitor
        self.snapshot_layout = layout
        self.grid_velocity = grid_velocity
        self.galilean = galilean
        self.nci_growth_limit = growth
        self.nci_energy_fraction = fraction
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-boosted-frame",
                "frame": frame.plan_id,
                "pic": pic.plan_id,
                "snapshots": None if snapshots is None else snapshots.plan_id,
                "nci": monitor.plan_id,
                "nci_growth_limit": growth,
                "nci_energy_fraction": fraction,
            }
        )

    @property
    def solver(self) -> PreparedSpectralMaxwell:
        """The run's PSATD field solver (validated at preparation)."""
        solver = self.pic.solver
        if not isinstance(solver, PreparedSpectralMaxwell):
            raise TypeError("The boosted PIC run lost its PreparedSpectralMaxwell.")
        return solver

    # -- coordinates -----------------------------------------------------------------

    def _boosted_positions(self, grid: Array, time: Array, /) -> Array:
        """Boosted positions ``x′ = x_grid + v_grid t′`` of grid coordinates."""
        velocity = jnp.asarray(self.grid_velocity, dtype=jnp.float64)
        return grid + jnp.asarray(time, dtype=jnp.float64)[..., None] * velocity

    def _component_points(self, offsets: np.ndarray, /) -> Array:
        """Grid coordinates ``[N₀, N₁, N₂, 3]`` of one component's locations."""
        plan = self.solver.plan
        axes = [
            origin + (np.arange(count) + offset) * spacing
            for origin, count, spacing, offset in zip(
                plan.origin, plan.counts, plan.spacing, offsets, strict=True
            )
        ]
        return jnp.asarray(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1))

    # -- initialization ----------------------------------------------------------------

    def _vacuum_field(
        self, sources: tuple[ExternalFieldSource, ...], time: Array, /
    ) -> tuple[Array, Array, Array]:
        """Boosted fields of lab vacuum sources at the solver's component locations."""
        electric_offsets, magnetic_offsets = _component_offsets(self.solver.plan.grid)
        counts = self.solver.plan.counts
        electric = jnp.zeros((*counts, 3), dtype=jnp.float64)
        magnetic = jnp.zeros((*counts, 3), dtype=jnp.float64)
        supported = jnp.asarray(True)
        for kind, offsets in (
            ("electric", electric_offsets),
            ("magnetic", magnetic_offsets),
        ):
            for component in range(3):
                points = self._boosted_positions(
                    self._component_points(offsets[component]), time
                ).reshape((-1, 3))
                lab_time, lab_position = self.frame.to_lab(time, points)
                for source in sources:
                    sample = source.external_fields(lab_position, lab_time)
                    boosted_e, boosted_b = self.frame.boost_fields(
                        sample.electric, sample.magnetic
                    )
                    supported = supported & jnp.all(sample.support)
                    value = (boosted_e if kind == "electric" else boosted_b)[
                        :, component
                    ].reshape(counts)
                    if kind == "electric":
                        electric = electric.at[..., component].add(value)
                    else:
                        magnetic = magnetic.at[..., component].add(value)
        return electric, magnetic, supported

    def initialize(
        self,
        positions: Sequence[ArrayLike],
        velocities: Sequence[ArrayLike],
        step_size: ArrayLike,
        /,
        *,
        time: ArrayLike,
        active_masks: Sequence[ArrayLike | None] | None = None,
        masses: Sequence[ArrayLike | None] | None = None,
        vacuum_fields: Sequence[ExternalFieldSource] = (),
        vacuum_overlap_tolerance: float = 1.0e-6,
    ) -> BoostedFrameState:
        """Boosted initial state on the slice ``t′ = time``.

        ``positions``/``velocities`` are boosted-frame positions and physical
        velocities on the slice (`BoostedFramePlan.boost_particles`).
        ``vacuum_fields`` are lab-frame vacuum solutions (laser pulses) sampled
        at the lab events of the slice, boosted, added to the grid field, and
        Gauss-projected. They must not overlap particles: the leapfrog bootstrap
        of the PIC initialization does not see them, so a field above
        ``vacuum_overlap_tolerance`` of its grid maximum at an active particle is
        refused.
        """
        start = jnp.asarray(time, dtype=jnp.float64).reshape(())
        sources = tuple(vacuum_fields)
        if any(not isinstance(value, ExternalFieldSource) for value in sources):
            raise TypeError("vacuum_fields must implement ExternalFieldSource.")
        tolerance = float(vacuum_overlap_tolerance)
        if not (math.isfinite(tolerance) and tolerance > 0.0):
            raise ValueError("vacuum_overlap_tolerance must be positive.")
        velocity_values = tuple(
            jnp.asarray(value, dtype=jnp.float64) for value in velocities
        )
        grid_positions = tuple(
            jnp.asarray(value, dtype=jnp.float64)
            - start * jnp.asarray(self.grid_velocity, dtype=jnp.float64)
            for value in positions
        )
        state = self.pic.initialize(
            grid_positions,
            velocity_values,
            step_size,
            active_masks=active_masks,
            masses=masses,
            time=start,
        )
        if not self.galilean:
            state = self._refuse_drifting_plasma(state, velocity_values)
        field = state.field
        divergence = jnp.zeros((), dtype=jnp.float64)
        if sources:
            field, divergence = self._add_vacuum_fields(state, sources, start, tolerance)
            history = state.field_history
            state = dataclasses.replace(
                state,
                field=field,
                field_history=None
                if history is None
                else PICFieldHistory(field, history.time),
            )
        return BoostedFrameState(
            state,
            None if self.snapshot_layout is None else self._initial_snapshots(),
            _initial_evidence(self.nci.sample(state.field).high_energy, divergence),
        )

    def _refuse_drifting_plasma(
        self, state: ElectromagneticPICState, velocities: tuple[Array, ...], /
    ) -> ElectromagneticPICState:
        """Refuse lab-resting plasma (mean drift ``−v_b``) on a standard grid."""
        drift = -self.frame.beta * self.frame.speed_of_light
        plasma = jnp.asarray(False)
        for species, velocity in zip(state.species, velocities, strict=True):
            active = species.population.active
            count = jnp.sum(active)
            mean = jnp.sum(jnp.where(active, velocity[:, self.frame.axis], 0.0)) / (
                jnp.maximum(count, 1)
            )
            plasma = plasma | (
                (count > 0)
                & (jnp.abs(mean - drift) <= 1.0e-3 * self.frame.speed_of_light)
            )
        return eqx.error_if(
            state,
            plasma,
            "Plasma drifting at −v_b through a standard PSATD grid is refused: the "
            "boosted plasma requires a Galilean grid comoving at galilean_velocity.",
        )

    def _add_vacuum_fields(
        self,
        state: ElectromagneticPICState,
        sources: tuple[ExternalFieldSource, ...],
        time: Array,
        tolerance: float,
        /,
    ) -> tuple[SpectralMaxwellState, Array]:
        electric, magnetic, supported = self._vacuum_field(sources, time)
        field = state.field
        c = self.frame.speed_of_light
        vacuum = dataclasses.replace(
            field,
            electric=electric,
            magnetic=magnetic,
            averaged_electric=None if field.averaged_electric is None else electric,
            averaged_magnetic=None if field.averaged_magnetic is None else magnetic,
        )
        peak = jnp.maximum(
            jnp.max(jnp.abs(electric), initial=0.0),
            c * jnp.max(jnp.abs(magnetic), initial=0.0),
        )
        overlap = jnp.zeros((), dtype=jnp.float64)
        for index, species in enumerate(state.species):
            active = species.population.active
            e, b, _ = self.solver.gather_fields(
                index, species.particles.position, active, vacuum
            )
            local = jnp.maximum(jnp.abs(e), c * jnp.abs(b))
            overlap = jnp.maximum(
                overlap, jnp.max(jnp.where(active[:, None], local, 0.0), initial=0.0)
            )
        combined = dataclasses.replace(
            field,
            electric=field.electric + electric,
            magnetic=field.magnetic + magnetic,
            averaged_electric=None
            if field.averaged_electric is None
            else field.averaged_electric + electric,
            averaged_magnetic=None
            if field.averaged_magnetic is None
            else field.averaged_magnetic + magnetic,
        )
        projection = self.solver.project_gauss(combined, field.charge)
        projected = eqx.error_if(
            projection.field,
            ~supported | ~projection.successful | (overlap > tolerance * peak),
            "Boosted vacuum fields must be supported on the whole initial slice and "
            "must not overlap particles (their leapfrog bootstrap would miss them).",
        )
        return projected, projection.divergence_before

    def _initial_snapshots(self) -> BoostedSnapshotState:
        layout = self.snapshot_layout
        if layout is None:
            raise ValueError("This boosted run records no snapshots.")
        capacity = layout.ring_capacity
        counts = list(self.solver.plan.counts)
        del counts[self.frame.axis]
        zone = (capacity, layout.samples.shape[0], counts[0], counts[1], 3)
        return BoostedSnapshotState(
            owner=layout.owners[:, 0],
            electric=jnp.zeros(zone, dtype=jnp.float64),
            magnetic=jnp.zeros(zone, dtype=jnp.float64),
            field_filled=jnp.zeros(zone[:2], dtype=jnp.bool_),
            particles=tuple(
                BoostedParticleSnapshotBuffer(
                    jnp.zeros((capacity, self.pic.species[index].capacity, 3)),
                    jnp.zeros((capacity, self.pic.species[index].capacity, 3)),
                    jnp.zeros(
                        (capacity, self.pic.species[index].capacity), dtype=jnp.uint32
                    ),
                    jnp.zeros(
                        (capacity, self.pic.species[index].capacity), dtype=jnp.uint32
                    ),
                    jnp.zeros(
                        (capacity, self.pic.species[index].capacity), dtype=jnp.bool_
                    ),
                )
                for index in layout.species
            ),
            conflicts=jnp.zeros((), dtype=jnp.int32),
        )

    # -- stepping ------------------------------------------------------------------------

    @checked
    def step_detailed(
        self, state: BoostedFrameState, step_size: ArrayLike, /
    ) -> BoostedFrameStepResult:
        """One PIC step under the NCI guard, recording back-transformed snapshots."""
        result = self.pic.step_detailed(state.pic, step_size)
        candidate_pic = result.candidate_state
        sample = self.nci.sample(candidate_pic.field)
        tiny = jnp.finfo(jnp.float64).tiny
        evidence = state.evidence
        reference = evidence.nci_reference
        growth = sample.high_energy / jnp.maximum(reference, tiny)
        fraction = sample.high_energy / jnp.maximum(sample.total_energy, tiny)
        # NCI is exponential high-|k| growth: refuse a field whose high-|k| energy
        # outgrew its reference and dominates beyond the declared fraction. A
        # field-free start has no reference yet; its first nonzero field sets it.
        admissible = (
            (reference == 0.0)
            | (growth <= self.nci_growth_limit)
            | (fraction <= self.nci_energy_fraction)
        )
        successful = result.successful & admissible
        diagnostics = result.diagnostics.field
        if not isinstance(diagnostics, SpectralMaxwellDiagnostics):
            raise TypeError("The spectral solver returned foreign field diagnostics.")
        attempted = dataclasses.replace(
            evidence,
            nci_rejections=evidence.nci_rejections
            + (result.successful & ~admissible).astype(jnp.int32),
            surface_current=jnp.maximum(
                evidence.surface_current, diagnostics.surface_current
            ),
        )
        snapshots = (
            None
            if state.snapshots is None
            else self._record(state.snapshots, state.pic, candidate_pic)
        )
        candidate = BoostedFrameState(
            candidate_pic,
            snapshots,
            dataclasses.replace(
                attempted,
                nci_reference=jnp.where(reference == 0.0, sample.high_energy, reference),
                nci_growth=jnp.where(
                    reference == 0.0,
                    attempted.nci_growth,
                    jnp.maximum(attempted.nci_growth, growth),
                ),
                nci_fraction=jnp.maximum(attempted.nci_fraction, fraction),
            ),
        )
        rejected = BoostedFrameState(state.pic, state.snapshots, attempted)
        accepted = _select(successful, candidate, rejected)
        return BoostedFrameStepResult(
            candidate,
            accepted,
            result,
            sample,
            jnp.where(reference == 0.0, 1.0, growth),
            fraction,
            admissible,
            successful,
        )

    def _record(
        self,
        snapshots: BoostedSnapshotState,
        before: ElectromagneticPICState,
        after: ElectromagneticPICState,
        /,
    ) -> BoostedSnapshotState:
        layout = self.snapshot_layout
        if layout is None:
            raise ValueError("This boosted run records no snapshots.")
        start, stop = before.time, after.time
        started = jnp.sum(layout.starts < stop, axis=1) - 1
        owner = jnp.take_along_axis(
            layout.owners, jnp.maximum(started, 0)[:, None], axis=1
        )[:, 0]
        changed = owner != snapshots.owner
        conflicts = snapshots.conflicts + jnp.sum(
            changed & (layout.completes[snapshots.owner] > start)
        ).astype(jnp.int32)
        keep = ~changed
        electric, magnetic, filled = self._record_fields(
            layout,
            owner,
            jnp.where(keep[:, None, None, None, None], snapshots.electric, 0.0),
            jnp.where(keep[:, None, None, None, None], snapshots.magnetic, 0.0),
            snapshots.field_filled & keep[:, None],
            before,
            after,
        )
        particles = tuple(
            self._record_particles(
                layout,
                owner,
                species,
                BoostedParticleSnapshotBuffer(
                    buffer.positions,
                    buffer.proper_velocities,
                    buffer.id_hi,
                    buffer.id_lo,
                    buffer.filled & keep[:, None],
                ),
                before,
                after,
            )
            for species, buffer in zip(layout.species, snapshots.particles, strict=True)
        )
        return BoostedSnapshotState(
            owner, electric, magnetic, filled, particles, conflicts
        )

    def _record_fields(
        self,
        layout: _SnapshotLayout,
        owner: Array,
        electric: Array,
        magnetic: Array,
        filled: Array,
        before: ElectromagneticPICState,
        after: ElectromagneticPICState,
        /,
    ) -> tuple[Array, Array, Array]:
        frame = self.frame
        axis = frame.axis
        gamma = frame.lorentz_factor
        c = frame.speed_of_light
        plan = self.solver.plan
        count = plan.counts[axis]
        spacing = plan.spacing[axis]
        start, stop = before.time, after.time
        lab_time = layout.lab_times[owner][:, None]
        samples = layout.samples[None, :]
        boosted_time = gamma * (lab_time - frame.beta * samples / c)
        boosted_position = gamma * (samples - frame.beta * c * lab_time)
        grid_position = boosted_position - self.grid_velocity[axis] * boosted_time
        inside = (boosted_time >= start) & (boosted_time < stop)
        weight = jnp.clip((boosted_time - start) / (stop - start), 0.0, 1.0)
        electric_offsets, magnetic_offsets = _component_offsets(plan.grid)
        transverse = tuple(value for value in range(3) if value != axis)

        def planes(values: Array, offsets: np.ndarray) -> Array:
            columns = []
            for component in range(3):
                array = values[..., component]
                for other in transverse:
                    if offsets[component][other] != 0.0:
                        array = 0.5 * (array + jnp.roll(array, 1, axis=other))
                array = jnp.moveaxis(array, axis, 0)
                fractional = (grid_position - plan.origin[axis]) / spacing - offsets[
                    component
                ][axis]
                lower = jnp.floor(fractional)
                fraction = (fractional - lower)[..., None, None]
                first = jnp.mod(lower.astype(jnp.int32), count)
                second = jnp.mod(first + 1, count)
                columns.append((1.0 - fraction) * array[first] + fraction * array[second])
            return jnp.stack(columns, axis=-1)

        time_weight = weight[..., None, None, None]
        boosted_e = (1.0 - time_weight) * planes(
            before.field.electric, electric_offsets
        ) + time_weight * planes(after.field.electric, electric_offsets)
        boosted_b = (1.0 - time_weight) * planes(
            before.field.magnetic, magnetic_offsets
        ) + time_weight * planes(after.field.magnetic, magnetic_offsets)
        lab_e, lab_b = frame.lab_fields(boosted_e, boosted_b)
        mask = inside[..., None, None, None]
        return (
            jnp.where(mask, lab_e, electric),
            jnp.where(mask, lab_b, magnetic),
            filled | inside,
        )

    def _record_particles(
        self,
        layout: _SnapshotLayout,
        owner: Array,
        species: int,
        buffer: BoostedParticleSnapshotBuffer,
        before: ElectromagneticPICState,
        after: ElectromagneticPICState,
        /,
    ) -> BoostedParticleSnapshotBuffer:
        frame = self.frame
        start_state, stop_state = before.species[species], after.species[species]
        start_population = start_state.population
        stop_population = stop_state.population
        same = (
            start_population.active
            & stop_population.active
            & (start_population.id_hi == stop_population.id_hi)
            & (start_population.id_lo == stop_population.id_lo)
        )
        first = self._boosted_positions(start_state.particles.position, before.time)
        second = self._boosted_positions(stop_state.particles.position, after.time)
        first_time, _ = frame.to_lab(before.time, first)
        second_time, _ = frame.to_lab(after.time, second)
        target = layout.lab_times[owner][:, None]
        crossing = (
            same[None, :]
            & (first_time[None, :] <= target)
            & (target < second_time[None, :])
        )
        weight = jnp.clip(
            (target - first_time[None, :])
            / jnp.maximum(second_time - first_time, jnp.finfo(jnp.float64).tiny)[None, :],
            0.0,
            1.0,
        )
        position = first[None] + weight[..., None] * (second - first)[None]
        time = before.time + weight * (after.time - before.time)
        _, lab_position = frame.to_lab(time, position)
        lower = jnp.asarray(frame.lab_lower)
        upper = jnp.asarray(frame.lab_upper)
        inside = jnp.all((lab_position >= lower) & (lab_position <= upper), axis=-1)
        record = crossing & inside & ~buffer.filled
        proper = frame.lab_proper_velocity(stop_state.particles.proper_velocity)
        mask = record[..., None]
        return BoostedParticleSnapshotBuffer(
            jnp.where(mask, lab_position, buffer.positions),
            jnp.where(mask, proper[None], buffer.proper_velocities),
            jnp.where(record, stop_population.id_hi[None], buffer.id_hi),
            jnp.where(record, stop_population.id_lo[None], buffer.id_lo),
            buffer.filled | record,
        )

    # -- consumers -----------------------------------------------------------------------

    def _slot(
        self, state: BoostedFrameState, index: int, /
    ) -> tuple[BoostedSnapshotState, int]:
        layout = self.snapshot_layout
        snapshots = state.snapshots
        if layout is None or snapshots is None:
            raise ValueError("This boosted run records no snapshots.")
        count = layout.lab_times.shape[0]
        if not 0 <= index < count:
            raise ValueError("Snapshot index outside the requested lab times.")
        slot = index % layout.ring_capacity
        if int(snapshots.owner[slot]) != index:
            raise ValueError(
                f"Lab snapshot {index} is not resident: ring slot {slot} holds snapshot "
                f"{int(snapshots.owner[slot])}."
            )
        return snapshots, slot

    def lab_field_snapshot(
        self, state: BoostedFrameState, index: int, /
    ) -> BoostedLabFieldSnapshot:
        """Resident back-transformed lab field snapshot ``index``."""
        snapshots, slot = self._slot(state, index)
        layout = self.snapshot_layout
        if layout is None:
            raise ValueError("This boosted run records no snapshots.")
        axis = self.frame.axis
        plan = self.solver.plan
        transverse = tuple(value for value in range(3) if value != axis)
        coordinates = tuple(
            jnp.asarray(
                plan.origin[value] + plan.spacing[value] * np.arange(plan.counts[value])
            )
            for value in transverse
        )
        return BoostedLabFieldSnapshot(
            lab_time=float(layout.lab_times[index]),
            axis=axis,
            axis_coordinates=layout.samples,
            transverse_coordinates=(coordinates[0], coordinates[1]),
            electric=jnp.moveaxis(snapshots.electric[slot], 0, axis),
            magnetic=jnp.moveaxis(snapshots.magnetic[slot], 0, axis),
            filled=snapshots.field_filled[slot],
        )

    def lab_particle_snapshot(
        self, state: BoostedFrameState, index: int, species: int, /
    ) -> BoostedLabParticleSnapshot:
        """Resident lab particle snapshot ``index`` of one recorded species."""
        snapshots, slot = self._slot(state, index)
        layout = self.snapshot_layout
        if layout is None:
            raise ValueError("This boosted run records no snapshots.")
        if species not in layout.species:
            raise ValueError(f"Species {species} is not recorded in the snapshots.")
        buffer = snapshots.particles[layout.species.index(species)]
        return BoostedLabParticleSnapshot(
            lab_time=float(layout.lab_times[index]),
            species=species,
            positions=buffer.positions[slot],
            proper_velocities=buffer.proper_velocities[slot],
            id_hi=buffer.id_hi[slot],
            id_lo=buffer.id_lo[slot],
            filled=buffer.filled[slot],
        )

    def lab_trajectory(
        self,
        state: BoostedFrameState,
        recorder: int,
        scale: ElectromagneticScaleContract,
        /,
    ) -> ChargedTrajectory:
        """Lab-frame lanes of one track recorder with per-lane lab times."""
        if not 0 <= recorder < len(self.pic.recorders):
            raise ValueError("recorder is not a recorder index of the run.")
        tracks = self.pic.recorders[recorder]
        if not isinstance(tracks, PICTrackRecorder):
            raise TypeError("lab_trajectory requires a PICTrackRecorder.")
        boosted = tracks.to_charged_trajectory(state.pic.recorders[recorder], scale)
        positions = self._boosted_positions(boosted.positions, boosted.times)
        times, lab_positions = self.frame.to_lab(boosted.times, positions)
        return ChargedTrajectory(
            times,
            lab_positions,
            self.frame.lab_proper_velocity(boosted.proper_velocities),
            boosted.charges,
            boosted.multiplicities,
            boosted.active,
            (boosted.id_hi, boosted.id_lo),
        )

    @checked
    def lab_far_field(
        self,
        state: BoostedFrameState,
        far_field: MaxwellFarFieldResult,
        /,
        *,
        emission: SpectralEmissionCompleteness,
    ) -> LorentzSpectralTransform:
        """Relabel a boosted Huygens far field into the lab frame.

        Admitted only for a standard grid carrying Huygens observers, in vacuum,
        after every observer's acquisition window closed, with ``J = 0`` met on
        the surfaces at every attempted step.
        """
        if self.galilean or not self.solver.huygens:
            raise ValueError(
                "Boosted Huygens far fields require a standard spectral grid with "
                "Huygens observers; Galilean grids carry no Huygens sampling."
            )
        if float(state.evidence.surface_current) != 0.0:
            raise ValueError(
                "A Huygens surface carried current during the run: the boosted far "
                "field requires J = 0 on the surface over the whole window."
            )
        now = float(state.pic.time)
        for box in self.solver.huygens:
            stop = box.acquisition.stop_time
            if stop is None or now < stop:
                raise ValueError(
                    "A Huygens acquisition window is still open; the boosted spectrum "
                    "transforms only as complete emission."
                )
        frequencies = far_field.angular_frequencies[:, None]
        directions = far_field.directions[None, :, :]
        return transform_spectral_energy(
            self.frame._beta(-1.0),
            jnp.broadcast_to(frequencies, far_field.spectral_energy.shape),
            jnp.broadcast_to(directions, (*far_field.spectral_energy.shape, 3)),
            far_field.spectral_energy,
            emission=emission,
        )

    # -- restart -------------------------------------------------------------------------

    def restart_component(self, state: BoostedFrameState, /) -> PICRestartComponent:
        """Snapshot ring and evidence as one component owned by this prepared run."""
        return restart_component(
            _RESTART_NAME, self.prepared_id, (state.snapshots, state.evidence)
        )

    def restore_component(
        self, component: PICRestartComponent, /
    ) -> tuple[BoostedSnapshotState | None, BoostedFrameEvidence]:
        template = (
            None if self.snapshot_layout is None else self._initial_snapshots(),
            _initial_evidence(
                jnp.zeros((), dtype=jnp.float64), jnp.zeros((), dtype=jnp.float64)
            ),
        )
        return restore_component(component, _RESTART_NAME, self.prepared_id, template)

    def checkpoint(self, state: BoostedFrameState, /) -> PICRestartCheckpoint:
        """PIC restart components plus the boosted-frame component."""
        pic = self.pic.checkpoint(state.pic)
        return PICRestartCheckpoint((*pic.components, self.restart_component(state)))

    @checked
    def restore(self, checkpoint: PICRestartCheckpoint, /) -> BoostedFrameState:
        boosted = [
            value for value in checkpoint.components if value.name == _RESTART_NAME
        ]
        if len(boosted) != 1:
            raise ValueError("The checkpoint carries no single boosted-frame component.")
        snapshots, evidence = self.restore_component(boosted[0])
        pic = self.pic.restore(
            PICRestartCheckpoint(
                tuple(
                    value
                    for value in checkpoint.components
                    if value.name != _RESTART_NAME
                )
            )
        )
        return BoostedFrameState(pic, snapshots, evidence)


def _require_covered(frame: BoostedFramePlan, solver: PreparedSpectralMaxwell, /) -> None:
    """Refuse a Galilean grid that does not cover the contracted lab domain."""
    plan = solver.plan
    for axis, (low, high) in enumerate(
        zip(frame.grid_lower, frame.grid_upper, strict=True)
    ):
        first = plan.origin[axis]
        last = first + plan.counts[axis] * plan.spacing[axis]
        slack = 1.0e-9 * max(1.0, abs(first), abs(last))
        if low < first - slack or high > last + slack:
            raise ValueError(
                f"The Galilean grid does not cover the lab domain along axis {axis}: "
                f"grid [{first:.6g}, {last:.6g}] versus contracted lab domain "
                f"[{low:.6g}, {high:.6g}]."
            )


__all__ = [
    "BoostedExternalField",
    "BoostedFrameEvidence",
    "BoostedFramePlan",
    "BoostedFrameState",
    "BoostedFrameStepResult",
    "BoostedLabFieldSnapshot",
    "BoostedLabParticleSnapshot",
    "BoostedParticleSnapshotBuffer",
    "BoostedParticles",
    "BoostedSnapshotPlan",
    "BoostedSnapshotState",
    "PreparedBoostedFrame",
]
