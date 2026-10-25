#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nudged elastic bands, saddle qualification, and intrinsic reaction paths.

Two NEB routes share one force projection (`nudged_elastic_band_forces`: the
improved energy-weighted tangent, tangential spring force and climbing image)
and the native optimizer owners (`FIRE`, `FIRE2`, `TrustScaledLBFGS`; a
downhill-checked FIRE is refused because the NEB force is not a gradient):

* `NudgedElasticBandPlan` is the honest host-surface route.  Each round
  evaluates every interior image through the explicit provider surface and
  consumes the evaluation with one native optimizer step; no provider call is
  hidden inside compiled code.
* `AtomisticNudgedElasticBandPlan` is the bounded device route over a native
  `AtomisticPotentialEnergySurface`: whole bands of ``G`` path groups are
  evaluated by vmapped native energy/force passes against per-image accepted
  resources, moved by one optimizer layout (one group per path, fixed
  endpoints as inactive sites) and committed band-atomically.

Reported images, energies, physical forces and NEB forces always belong to
the same accepted iterate.  Periodic fixed-cell paths use unwrapped Cartesian
coordinates on a declared image branch (`ReactionPathBranch`); the shortest
image branch is a named policy with an ambiguity certificate.  Variable-cell
NEB is not provided.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from enum import StrEnum
from math import isfinite
from typing import Any, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import IdentityWords
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomicStructure, AtomisticSystemPlan
from ...atomistic._neighborhood_bundle import AtomisticNeighborhoodBundleState
from ...atomistic._potential_program import AbstractPreparedAtomisticHamiltonian
from ...atomistic._relaxation import (
    _identity_rows,
    _identity_strings,
    _step_relaxation_method,
    atomistic_stationary_inertia,
    AtomisticGaugePolicy,
    AtomisticHamiltonianTransition,
    AtomisticRelaxationStatus,
    AtomisticStationaryEvidence,
    AtomisticStationaryPolicy,
    AtomisticTrialOutcome,
    evaluate_cartesian_trial,
    hamiltonian_transition_kind,
    initialize_capacity,
    program_energy,
    replay_capacity,
    trial_geometry,
)
from ...atomistic._system import PreparedAtomisticSystem
from ...optim import (
    FIRE,
    FIRE2,
    FIREState,
    OptimizationStatus,
    OptimizerGroupLayout,
    TrustScaledLBFGS,
    TrustScaledLBFGSState,
)
from ...typing import checked, ConvertibleToArray
from .._atomistic import AtomisticPotentialEnergySurface
from .._optimization import _require_structure_matches_system
from .._surface import AbstractPreparedPotentialEnergySurface
from ..vibration._harmonic import StationaryPointKind, VibrationalAnalysisResult


_SUCCESS = int(OptimizationStatus.SUCCESS)


# -- shared force projection ------------------------------------------------------


class NudgedElasticBandProjection(NamedTuple):
    """NEB forces of one band; endpoints carry zero force."""

    forces: Array
    tangents: Array
    climbing_image: Array
    image_residual: Array
    valid: Array


def nudged_elastic_band_forces(
    images: ArrayLike,
    energies: ArrayLike,
    forces: ArrayLike,
    path_mask: ArrayLike,
    /,
    *,
    spring_constant: ArrayLike,
    climbing: ArrayLike,
) -> NudgedElasticBandProjection:
    """Project physical forces of one band ``(M, N, 3)`` onto NEB forces.

    Tangents follow Henkelman and Jónsson (J. Chem. Phys. 113, 9978, 2000):
    the uphill neighbor difference at monotone images and the
    energy-weighted combination ``tau+ dV_max + tau- dV_min`` (or its mirror) at
    extrema.  Interior images receive the perpendicular physical force plus
    the tangential spring force ``k (|tau+| - |tau-|) tau``; when ``climbing``
    the highest interior image receives ``F - 2 (F.tau) tau`` instead.  Only
    ``path_mask`` atoms enter tangents, distances and forces; differences are
    taken in the supplied (unwrapped) coordinates.  ``image_residual`` is the
    largest per-atom NEB force norm of each image and ``valid`` is false for a
    zero or nonfinite tangent or nonfinite input.
    """
    mask = jnp.asarray(path_mask, dtype=jnp.bool_)[None, :, None]
    band = jnp.where(mask, jnp.asarray(images), 0.0)
    physical = jnp.where(mask, jnp.asarray(forces, dtype=band.dtype), 0.0)
    energy = jnp.asarray(energies, dtype=band.dtype)
    forward = band[2:] - band[1:-1]
    backward = band[1:-1] - band[:-2]
    previous, current, following = energy[:-2], energy[1:-1], energy[2:]
    ascending = ((following > current) & (current > previous))[:, None, None]
    descending = ((following < current) & (current < previous))[:, None, None]
    rise = jnp.abs(following - current)
    fall = jnp.abs(previous - current)
    largest = jnp.maximum(rise, fall)[:, None, None]
    smallest = jnp.minimum(rise, fall)[:, None, None]
    mixed = jnp.where(
        (following > previous)[:, None, None],
        forward * largest + backward * smallest,
        forward * smallest + backward * largest,
    )
    tangent = jnp.where(ascending, forward, jnp.where(descending, backward, mixed))
    norm = jnp.sqrt(jnp.sum(tangent * tangent, axis=(1, 2)))
    valid = (
        jnp.all(jnp.isfinite(norm) & (norm > 0.0))
        & jnp.all(jnp.isfinite(energy))
        & jnp.all(jnp.isfinite(physical))
    )
    unit = tangent / jnp.where(norm > 0.0, norm, 1.0)[:, None, None]
    physical = physical[1:-1]
    parallel = jnp.sum(physical * unit, axis=(1, 2))[:, None, None]
    separation = jnp.sqrt(jnp.sum(forward * forward, axis=(1, 2))) - jnp.sqrt(
        jnp.sum(backward * backward, axis=(1, 2))
    )
    spring = jnp.asarray(spring_constant, dtype=band.dtype)
    nudged = physical - parallel * unit + spring * separation[:, None, None] * unit
    climber = jnp.argmax(current)
    climbs = jnp.asarray(climbing, dtype=jnp.bool_) & (
        jnp.arange(current.shape[0]) == climber
    )
    interior = jnp.where(climbs[:, None, None], physical - 2.0 * parallel * unit, nudged)
    interior = jnp.where(mask, interior, 0.0)
    zero = jnp.zeros_like(band[:1])
    projected = jnp.concatenate((zero, interior, zero))
    return NudgedElasticBandProjection(
        projected,
        jnp.concatenate((zero, unit, zero)),
        (climber + 1).astype(jnp.int32),
        jnp.max(jnp.linalg.norm(projected, axis=-1), axis=-1),
        valid,
    )


# -- endpoints, correspondence and periodic branch ------------------------------------


class ReactionPathBranch(StrEnum):
    """Declared image branch of the product endpoint of a periodic path."""

    EXPLICIT = "explicit"
    SHORTEST_IMAGE = "shortest-image"


@final
class ReactionPathBranchCertificate(StrictModule, NonTrainableState):
    """Product image shifts with the shortest-image ambiguity margins.

    The unwrapped product is ``product + image_shifts @ H``.  ``margins`` is,
    per atom, the distance gap between the chosen and the next-best image
    (``inf`` for an explicit branch or a finite system).
    """

    image_shifts: np.ndarray
    margins: np.ndarray
    policy: ReactionPathBranch = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        image_shifts: ConvertibleToArray,
        margins: ConvertibleToArray,
        /,
        *,
        policy: ReactionPathBranch,
    ) -> None:
        shifts = np.asarray(image_shifts, dtype=np.int64)
        gaps = np.asarray(margins, dtype=np.float64)
        if shifts.ndim != 2 or gaps.shape != shifts.shape[:1]:
            raise ValueError("Branch shifts and margins must align with atoms.")
        self.image_shifts = shifts
        self.margins = gaps
        self.policy = ReactionPathBranch(policy)
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "reaction-path-branch-certificate",
                "policy": self.policy.value,
                "image_shifts": shifts.tolist(),
            }
        )


def resolve_reaction_path_branch(
    system: AtomisticSystemPlan,
    reactant: ConvertibleToArray,
    product: ConvertibleToArray,
    /,
    *,
    policy: ReactionPathBranch | str = ReactionPathBranch.EXPLICIT,
    image_shifts: ConvertibleToArray | None = None,
    ambiguity_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, ReactionPathBranchCertificate]:
    """Return the unwrapped product on the declared branch and its certificate.

    ``EXPLICIT`` uses the supplied product coordinates plus optional integer
    ``image_shifts`` ``(N, r)``.  ``SHORTEST_IMAGE`` chooses, per active atom,
    the product image nearest the reactant among the rounded fractional image
    and its neighbors, and refuses when the next-best image is within
    ``ambiguity_tolerance``.  It never silently replaces a declared branch.
    """
    policy_ = ReactionPathBranch(policy)
    left = np.asarray(reactant, dtype=np.float64)
    right = np.asarray(product, dtype=np.float64)
    active = np.asarray(system.active_mask, dtype=np.bool_)
    cell = system.cell
    if cell is None:
        if image_shifts is not None or policy_ is ReactionPathBranch.SHORTEST_IMAGE:
            raise ValueError("A finite system has no periodic image branch.")
        return right.copy(), ReactionPathBranchCertificate(
            np.zeros((left.shape[0], 0), dtype=np.int64),
            np.full((left.shape[0],), np.inf),
            policy=policy_,
        )
    vectors = np.asarray(cell.vectors, dtype=np.float64)
    rank = vectors.shape[0]
    periodic = np.asarray(cell.periodic_mask, dtype=np.bool_)
    if policy_ is ReactionPathBranch.EXPLICIT:
        shifts = (
            np.zeros((left.shape[0], rank), dtype=np.int64)
            if image_shifts is None
            else np.asarray(image_shifts)
        )
        if not np.issubdtype(shifts.dtype, np.integer) or shifts.shape != (
            left.shape[0],
            rank,
        ):
            raise ValueError(
                f"image_shifts must be integers of shape {(left.shape[0], rank)}."
            )
        if np.any(shifts[:, ~periodic] != 0):
            raise ValueError("image_shifts must vanish on nonperiodic axes.")
        shifts = np.where(active[:, None], shifts, 0).astype(np.int64)
        margins = np.full((left.shape[0],), np.inf)
    else:
        if image_shifts is not None:
            raise ValueError("The shortest-image policy derives its own image shifts.")
        tolerance = float(ambiguity_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("ambiguity_tolerance must be finite and non-negative.")
        delta = np.where(active[:, None], right - left, 0.0)
        fractional = delta @ np.asarray(cell.inverse_vectors, dtype=np.float64)
        base = np.where(periodic, np.round(fractional), 0.0).astype(np.int64)
        offsets = np.asarray(
            [
                offset
                for offset in itertools.product((-1, 0, 1), repeat=rank)
                if all(periodic[axis] or offset[axis] == 0 for axis in range(rank))
            ],
            dtype=np.int64,
        )
        candidates = base[:, None, :] + offsets[None]
        distance = np.linalg.norm(
            delta[:, None, :] - candidates.astype(np.float64) @ vectors, axis=-1
        )
        order = np.argsort(distance, axis=1)
        best = np.take_along_axis(distance, order[:, :1], axis=1)[:, 0]
        second = (
            np.take_along_axis(distance, order[:, 1:2], axis=1)[:, 0]
            if offsets.shape[0] > 1
            else np.full_like(best, np.inf)
        )
        margins = np.where(active, second - best, np.inf)
        ambiguous = np.flatnonzero(active & (margins <= tolerance))
        if ambiguous.size:
            raise ValueError(
                "The shortest-image product branch is ambiguous for atoms "
                f"{ambiguous.tolist()}; declare explicit image shifts."
            )
        chosen = np.take_along_axis(candidates, order[:, :1, None], axis=1)[:, 0]
        shifts = np.where(active[:, None], -chosen, 0).astype(np.int64)
    unwrapped = right + shifts.astype(np.float64) @ vectors
    unwrapped = np.where(active[:, None], unwrapped, right)
    return unwrapped, ReactionPathBranchCertificate(shifts, margins, policy=policy_)


def _corresponding_product(
    reactant: AtomicStructure,
    product: AtomicStructure,
    system: AtomisticSystemPlan,
    correspondence: ConvertibleToArray | None,
    /,
) -> np.ndarray:
    """Product coordinates in reactant atom order under an explicit correspondence."""
    if product.scale.scale_id != system.units.scale.scale_id:
        raise ValueError("Product and path system scales differ.")
    if system.cell is None:
        if product.cell is not None:
            raise ValueError("A finite path system takes a finite product.")
    elif product.cell is None or not np.array_equal(
        np.asarray(product.cell), np.asarray(system.cell.vectors)
    ):
        raise ValueError("Every path endpoint must share the fixed system cell.")
    if correspondence is None:
        if not np.array_equal(
            np.asarray(reactant.particle_ids), np.asarray(product.particle_ids)
        ):
            raise ValueError("Reaction endpoints must preserve stable particle order.")
        return np.asarray(product.positions)
    order = np.asarray(correspondence)
    count = np.asarray(reactant.positions).shape[0]
    if (
        not np.issubdtype(order.dtype, np.integer)
        or order.shape != (count,)
        or not np.array_equal(np.sort(order), np.arange(count))
    ):
        raise ValueError("correspondence must be a permutation of the product atoms.")
    if not np.array_equal(
        np.asarray(product.atomic_numbers)[order], np.asarray(reactant.atomic_numbers)
    ):
        raise ValueError("correspondence maps atoms of different elements.")
    if not np.array_equal(
        np.asarray(product.active_mask)[order], np.asarray(reactant.active_mask)
    ):
        raise ValueError("correspondence maps active atoms onto padding.")
    return np.asarray(product.positions)[order]


def _path_endpoints(
    system: AtomisticSystemPlan,
    reactant: AtomicStructure,
    product: AtomicStructure,
    /,
    *,
    correspondence: ConvertibleToArray | None,
    branch: ReactionPathBranch | str,
    image_shifts: ConvertibleToArray | None,
) -> tuple[np.ndarray, np.ndarray, ReactionPathBranchCertificate]:
    _require_structure_matches_system(reactant, system)
    dtype = np.dtype(system.coordinate_dtype)
    left = np.asarray(reactant.positions, dtype=dtype)
    raw = _corresponding_product(reactant, product, system, correspondence)
    right, certificate = resolve_reaction_path_branch(
        system, left, raw, policy=branch, image_shifts=image_shifts
    )
    active = np.asarray(system.active_mask, dtype=np.bool_)
    fixed = active & ~np.asarray(system.mobile_mask, dtype=np.bool_)
    if not np.array_equal(left[fixed], right[fixed], equal_nan=True):
        raise ValueError("Reaction endpoints must agree at every nonmobile active atom.")
    return left, right.astype(dtype), certificate


def _initial_band(
    system: AtomisticSystemPlan,
    left: np.ndarray,
    right: np.ndarray,
    image_count: int,
    initial_images: ConvertibleToArray | None,
    /,
) -> np.ndarray:
    active = np.asarray(system.active_mask, dtype=np.bool_)
    fixed = active & ~np.asarray(system.mobile_mask, dtype=np.bool_)
    if initial_images is None:
        fractions = np.linspace(0.0, 1.0, image_count)
        return left[None] + fractions[:, None, None] * (right - left)[None]
    images = np.asarray(initial_images, dtype=left.dtype).copy()
    if images.shape != (image_count, *left.shape):
        raise ValueError("initial_images has an incompatible shape.")
    if not np.array_equal(images[0][active], left[active], equal_nan=True) or not (
        np.array_equal(images[-1][active], right[active], equal_nan=True)
    ):
        raise ValueError("initial_images must preserve active endpoint coordinates.")
    if not np.array_equal(
        images[:, fixed],
        np.broadcast_to(left[fixed], (image_count, int(np.count_nonzero(fixed)), 3)),
        equal_nan=True,
    ):
        raise ValueError("initial_images must keep nonmobile active coordinates fixed.")
    return images


def _path_method(
    method: FIRE | FIRE2 | TrustScaledLBFGS | None, /
) -> FIRE | FIRE2 | TrustScaledLBFGS:
    method_ = FIRE() if method is None else method
    if not isinstance(method_, (FIRE, FIRE2, TrustScaledLBFGS)):
        raise TypeError("method must be FIRE, FIRE2 or TrustScaledLBFGS.")
    if isinstance(method_, FIRE) and method_.downhill_check:
        raise ValueError(
            "NEB forces are not the gradient of a scalar; a downhill-checked FIRE "
            "would compare meaningless objective values."
        )
    return method_


def _method_identity(method: FIRE | FIRE2 | TrustScaledLBFGS, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "reaction-path-method",
            "method": method.method_id,
            "controls": {
                name: getattr(method, name)
                for name in sorted(vars(method))
                if isinstance(getattr(method, name), (int, float, bool, str))
            },
        }
    )


def _path_values(
    spring_constant: float, force_tolerance: float, image_count: int, climbing_start: int
) -> tuple[float, float, int, int]:
    spring = float(spring_constant)
    tolerance = float(force_tolerance)
    images = int(image_count)
    climb = int(climbing_start)
    if images < 3 or climb < 0:
        raise ValueError("NEB needs at least three images and climbing_start >= 0.")
    if any(not isfinite(value) or value <= 0.0 for value in (spring, tolerance)):
        raise ValueError("NEB spring and force tolerance must be positive finite.")
    return spring, tolerance, images, climb


# -- result -------------------------------------------------------------------------


class ReactionPathResult(StrictModule, NonTrainableState):
    """One accepted band with its own energies, physical and NEB forces.

    ``images``, ``energies``, ``forces`` (physical), ``neb_forces``,
    ``tangents`` and ``image_residual`` (largest per-atom NEB force norm) all
    belong to one accepted iterate.  ``status`` is an
    `AtomisticRelaxationStatus`; ``climbing_active`` records whether the
    climbing-image force was applied in that iterate.
    """

    images: Array
    energies: Array
    forces: Array
    neb_forces: Array
    tangents: Array
    image_residual: Array
    iterations: Array
    provider_evaluations: Array
    successful: Array
    status: Array
    climbing_image: int = eqx.field(static=True)
    climbing_active: bool = eqx.field(static=True)
    source_result_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        images: ArrayLike,
        energies: ArrayLike,
        forces: ArrayLike,
        neb_forces: ArrayLike,
        tangents: ArrayLike,
        image_residual: ArrayLike,
        iterations: int,
        provider_evaluations: int,
        successful: ArrayLike,
        /,
        *,
        status: int,
        climbing_image: int,
        climbing_active: bool,
        source_result_ids: tuple[str, ...],
        plan_id: str,
    ) -> None:
        images_ = jnp.asarray(images)
        energies_ = jnp.asarray(energies, dtype=images_.dtype)
        forces_ = jnp.asarray(forces, dtype=images_.dtype)
        neb_ = jnp.asarray(neb_forces, dtype=images_.dtype)
        tangents_ = jnp.asarray(tangents, dtype=images_.dtype)
        residual_ = jnp.asarray(image_residual, dtype=images_.dtype)
        if images_.ndim != 3 or images_.shape[2] != 3:
            raise ValueError("Reaction path images must have shape (image, atom, 3).")
        count = images_.shape[0]
        if (
            energies_.shape != (count,)
            or residual_.shape != (count,)
            or forces_.shape != images_.shape
            or neb_.shape != images_.shape
            or tangents_.shape != images_.shape
        ):
            raise ValueError("Reaction path arrays must align with images.")
        climbing = int(climbing_image)
        if climbing <= 0 or climbing >= count - 1:
            raise ValueError("Climbing image must be an interior image.")
        sources = tuple(str(value).strip() for value in source_result_ids)
        if not sources or any(not value for value in sources):
            raise ValueError("Reaction path source result IDs must be non-empty.")
        status_ = int(status)
        AtomisticRelaxationStatus(status_)
        self.images = images_
        self.energies = energies_
        self.forces = forces_
        self.neb_forces = neb_
        self.tangents = tangents_
        self.image_residual = residual_
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.provider_evaluations = jnp.asarray(
            provider_evaluations, dtype=jnp.int32
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.status = jnp.asarray(status_, dtype=jnp.int32)
        self.climbing_image = climbing
        self.climbing_active = bool(climbing_active)
        self.source_result_ids = sources
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "reaction-path-result",
                "plan": self.plan_id,
                "climbing_image": climbing,
                "climbing_active": self.climbing_active,
                "status": status_,
                "sources": list(sources),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "images": np.asarray(images_),
                        "energies": np.asarray(energies_),
                        "forces": np.asarray(forces_),
                        "neb_forces": np.asarray(neb_),
                        "iterations": np.asarray(self.iterations),
                        "provider_evaluations": np.asarray(self.provider_evaluations),
                    }
                ),
            }
        )

    @property
    def transition_state_positions(self) -> Array:
        return self.images[self.climbing_image]


# -- host-surface route -----------------------------------------------------------------


class NudgedElasticBandPlan(StrictModule, NonTrainableState):
    """Host-provider NEB over an explicit potential-energy surface.

    Each round evaluates every interior image with ``surface.evaluate`` on the
    host (endpoints once), projects NEB forces with
    `nudged_elastic_band_forces` and consumes them with one step of the native
    ``method`` over a single-group layout whose endpoint and nonmobile atoms
    are inactive sites.  Climbing starts once ``climbing_start`` rounds have
    passed; convergence requires an active climbing image and every interior
    image residual at most ``force_tolerance``.  A failed image evaluation
    fails the whole band at its last accepted iterate.
    """

    system: AtomisticSystemPlan
    surface: AbstractPreparedPotentialEnergySurface
    method: FIRE | FIRE2 | TrustScaledLBFGS
    image_count: int = eqx.field(static=True)
    spring_constant: float = eqx.field(static=True)
    force_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    climbing_start: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        system: AtomisticSystemPlan,
        surface: AbstractPreparedPotentialEnergySurface,
        /,
        *,
        image_count: int = 7,
        spring_constant: float = 1.0,
        force_tolerance: float = 1.0e-3,
        maximum_steps: int = 500,
        climbing_start: int = 20,
        method: FIRE | FIRE2 | TrustScaledLBFGS | None = None,
    ) -> None:
        if surface.system_id != system.system_id or not surface.capabilities.forces:
            raise ValueError("NEB requires a force-capable surface for the same system.")
        if surface.units.unit_system_id != system.units.unit_system_id:
            raise ValueError("NEB surface and system unit identities differ.")
        if not np.any(
            np.asarray(system.active_mask, dtype=np.bool_)
            & np.asarray(system.mobile_mask, dtype=np.bool_)
        ):
            raise ValueError("NEB requires at least one active mobile atom.")
        spring, tolerance, images, climb = _path_values(
            spring_constant, force_tolerance, image_count, climbing_start
        )
        steps = int(maximum_steps)
        if steps <= 0:
            raise ValueError("maximum_steps must be positive.")
        method_ = _path_method(method)
        self.system = system
        self.surface = surface
        self.method = method_
        self.image_count = images
        self.spring_constant = spring
        self.force_tolerance = tolerance
        self.maximum_steps = steps
        self.climbing_start = climb
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nudged-elastic-band-plan",
                "system": system.system_id,
                "surface": surface.surface_id,
                "method": _method_identity(method_),
                "image_count": images,
                "spring_constant": spring,
                "force_tolerance": tolerance,
                "maximum_steps": steps,
                "climbing_start": climb,
            }
        )

    def run(
        self,
        reactant: AtomicStructure,
        product: AtomicStructure,
        /,
        *,
        initial_images: ConvertibleToArray | None = None,
        correspondence: ConvertibleToArray | None = None,
        branch: ReactionPathBranch | str = ReactionPathBranch.EXPLICIT,
        product_image_shifts: ConvertibleToArray | None = None,
    ) -> ReactionPathResult:
        left, right, _ = _path_endpoints(
            self.system,
            reactant,
            product,
            correspondence=correspondence,
            branch=branch,
            image_shifts=product_image_shifts,
        )
        images = _initial_band(self.system, left, right, self.image_count, initial_images)
        active = np.asarray(self.system.active_mask, dtype=np.bool_)
        path_mask = active & np.asarray(self.system.mobile_mask, dtype=np.bool_)
        cell = None if reactant.cell is None else np.asarray(reactant.cell)
        count = self.image_count
        sources: list[str] = []
        endpoints = (
            self.surface.evaluate(left, cell),
            self.surface.evaluate(right, cell),
        )
        evaluations = 2
        sources.extend(value.source_result_id for value in endpoints)
        record: tuple[np.ndarray, np.ndarray, np.ndarray, Any, bool] | None = None
        status = AtomisticRelaxationStatus.EVALUATION_LIMIT
        iterations = 0
        if not all(bool(value.successful) for value in endpoints):
            status = AtomisticRelaxationStatus.SCIENTIFIC_FAILURE
        else:
            boundary = np.asarray([float(value.energy) for value in endpoints])
            layout = OptimizerGroupLayout(
                np.broadcast_to(
                    np.where(path_mask, 0, -1), (count - 2, path_mask.shape[0])
                )
                .astype(np.int32)
                .copy(),
                group_count=1,
            )
            state = self.method.init(jnp.asarray(images[1:-1]), layout=layout)
            for step in range(self.maximum_steps):
                interior = np.asarray(state.parameters)
                evaluated = tuple(
                    self.surface.evaluate(image, cell) for image in interior
                )
                evaluations += count - 2
                sources.extend(value.source_result_id for value in evaluated)
                if not all(bool(value.successful) for value in evaluated):
                    status = AtomisticRelaxationStatus.SCIENTIFIC_FAILURE
                    break
                energies = np.concatenate(
                    (
                        boundary[:1],
                        [float(value.energy) for value in evaluated],
                        boundary[1:],
                    )
                )
                zero = np.zeros_like(left)[None]
                physical = np.concatenate(
                    (
                        zero,
                        np.stack([np.asarray(value.forces) for value in evaluated]),
                        zero,
                    )
                )
                band = np.concatenate((left[None], interior, right[None]))
                climbing = step >= self.climbing_start
                projection = nudged_elastic_band_forces(
                    band,
                    energies,
                    physical,
                    path_mask,
                    spring_constant=self.spring_constant,
                    climbing=climbing,
                )
                if not bool(projection.valid):
                    status = AtomisticRelaxationStatus.SCIENTIFIC_FAILURE
                    break
                converged = (
                    climbing
                    and float(jnp.max(projection.image_residual)) <= self.force_tolerance
                )
                state = _step_relaxation_method(
                    self.method,
                    state,
                    jnp.asarray([np.sum(energies[1:-1])], dtype=band.dtype),
                    -projection.forces[1:-1],
                    converged=jnp.asarray([converged]),
                )
                iterations = step + 1
                if bool(np.asarray(state.evaluation_accepted)[0]):
                    record = (band, energies, physical, projection, climbing)
                if converged:
                    status = AtomisticRelaxationStatus.CONVERGED
                    break
        if record is None:
            nan = np.full((count,), np.nan)
            zeros = np.zeros_like(images)
            return ReactionPathResult(
                images,
                nan,
                zeros,
                zeros,
                zeros,
                nan,
                iterations,
                evaluations,
                False,
                status=int(status),
                climbing_image=max(1, count // 2),
                climbing_active=False,
                source_result_ids=tuple(sources),
                plan_id=self.plan_id,
            )
        band, energies, physical, projection, climbing = record
        return ReactionPathResult(
            band,
            energies,
            physical,
            projection.forces,
            projection.tangents,
            projection.image_residual,
            iterations,
            evaluations,
            status is AtomisticRelaxationStatus.CONVERGED,
            status=int(status),
            climbing_image=int(projection.climbing_image),
            climbing_active=climbing,
            source_result_ids=tuple(sources),
            plan_id=self.plan_id,
        )


# -- native device route ------------------------------------------------------------------


@final
class AtomisticReactionPathState(StrictModule):
    """Device state of ``G`` whole path groups (leading axis ``G``).

    ``optimizer.parameters`` holds the pending interior band ``(G, M - 2, N,
    3)`` in unwrapped Cartesian coordinates; ``resources`` the accepted
    case-one bundle state of every interior image ``(G, M - 2)``.  ``images``
    to ``climbing_active`` describe the accepted band of each group (the same
    iterate throughout); ``status`` is an `AtomisticRelaxationStatus`.
    ``evaluations``/``iterations`` are run totals across Hamiltonian
    transitions (optimizer counters plus the carried bases) and
    ``hamiltonian_lineage`` the transition ids of the bank's Hamiltonian.

    PyTree: the plan and resource-binding digests are dynamic
    ``plan_identity``/``binding_identity`` and the lineage the stacked
    ``(L, 8)`` ``lineage_identities`` (host properties ``plan_id``,
    ``binding_id`` and ``hamiltonian_lineage`` decode the exact strings).
    Compiled `step` folds the binding check into the band status; the lineage
    depth ``L`` is an array shape (one compile per transition depth).
    """

    optimizer: FIREState | TrustScaledLBFGSState
    endpoints: Array
    endpoint_energies: Array
    resources: AtomisticNeighborhoodBundleState
    candidate_resources: AtomisticNeighborhoodBundleState
    images: Array
    energies: Array
    forces: Array
    neb_forces: Array
    tangents: Array
    image_residual: Array
    climbing_image: Array
    climbing_active: Array
    status: Array
    trial_outcome: Array
    active: Array
    capacity_replays: Array
    rejected_trials: Array
    reprimes: Array
    evaluation_base: Array
    iteration_base: Array
    plan_identity: IdentityWords
    binding_identity: IdentityWords
    lineage_identities: IdentityWords

    @property
    def plan_id(self) -> str:
        return self.plan_identity.fingerprint()

    @property
    def binding_id(self) -> str:
        return self.binding_identity.fingerprint()

    @property
    def hamiltonian_lineage(self) -> tuple[str, ...]:
        return _identity_strings(self.lineage_identities)

    @property
    def evaluations(self) -> Array:
        """Consumed band evaluations of the whole run per group."""
        return self.optimizer.evaluations + self.evaluation_base

    @property
    def iterations(self) -> Array:
        """Optimizer steps of the whole run per group."""
        return self.optimizer.iterations + self.iteration_base


@final
class AtomisticNudgedElasticBandPlan(StrictModule, NonTrainableState):
    """Bounded device NEB of ``group_count`` fixed-cell paths on a native surface.

    Every path has ``image_count`` images with fixed endpoints.  Climbing
    begins after ``climbing_start`` consumed band evaluations of a group;
    convergence requires the climbing image and every interior residual at
    most ``force_tolerance``.  ``maximum_evaluations`` bounds consumed band
    evaluations per group.

    PyTree: ``spring_constant``/``force_tolerance`` (exact ``float64``) and
    ``maximum_evaluations``/``climbing_start`` (``int64``) are dynamic 0-d
    leaves; the plan digest is dynamic ``plan_identity`` (host ``plan_id``).
    ``image_count`` and ``group_count`` define shapes and stay static.
    """

    surface: AtomisticPotentialEnergySurface
    method: FIRE | FIRE2 | TrustScaledLBFGS
    spring_constant: np.ndarray
    force_tolerance: np.ndarray
    maximum_evaluations: np.ndarray
    climbing_start: np.ndarray
    plan_identity: IdentityWords
    image_count: int = eqx.field(static=True)
    group_count: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        surface: AtomisticPotentialEnergySurface,
        /,
        *,
        image_count: int = 7,
        spring_constant: float = 1.0,
        force_tolerance: float = 1.0e-3,
        maximum_evaluations: int = 500,
        climbing_start: int = 20,
        group_count: int = 1,
        method: FIRE | FIRE2 | TrustScaledLBFGS | None = None,
    ) -> None:
        system = surface.system
        if system.topology.constraint_count > 0:
            raise ValueError("Native NEB refuses holonomic topology constraints.")
        if not np.any(
            np.asarray(system.active_mask, dtype=np.bool_)
            & np.asarray(system.mobile_mask, dtype=np.bool_)
        ):
            raise ValueError("NEB requires at least one active mobile atom.")
        spring, tolerance, images, climb = _path_values(
            spring_constant, force_tolerance, image_count, climbing_start
        )
        if int(maximum_evaluations) <= 0 or int(group_count) <= 0:
            raise ValueError("maximum_evaluations and group_count must be positive.")
        evaluations = int(maximum_evaluations)
        method_ = _path_method(method)
        self.surface = surface
        self.method = method_
        self.image_count = images
        self.spring_constant = np.asarray(spring, dtype=np.float64)
        self.force_tolerance = np.asarray(tolerance, dtype=np.float64)
        self.maximum_evaluations = np.asarray(evaluations, dtype=np.int64)
        self.climbing_start = np.asarray(climb, dtype=np.int64)
        self.group_count = int(group_count)
        self.plan_identity = IdentityWords.of(
            {
                "kind": "atomistic-nudged-elastic-band-plan",
                "surface": surface.surface_id,
                "method": _method_identity(method_),
                "image_count": images,
                "spring_constant": spring,
                "force_tolerance": tolerance,
                "maximum_evaluations": evaluations,
                "climbing_start": climb,
                "group_count": self.group_count,
            }
        )

    @property
    def plan_id(self) -> str:
        return self.plan_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        """Content-free layout: surface program layout, method, images, groups."""
        return canonical_fingerprint(
            {
                "kind": "atomistic-nudged-elastic-band-plan-layout",
                "program": self.surface.potential.layout_id,
                "method": f"{type(self.method).__qualname__}:{self.method.method_id}",
                "image_count": self.image_count,
                "group_count": self.group_count,
            }
        )

    def prepare(self) -> PreparedAtomisticNudgedElasticBand:
        return PreparedAtomisticNudgedElasticBand(self, self.surface.potential)


class _BandEvaluation(NamedTuple):
    candidate: AtomisticNeighborhoodBundleState
    band: Array
    energies: Array
    forces: Array
    projection: NudgedElasticBandProjection
    capacity: Array
    scientific: Array


@final
class PreparedAtomisticNudgedElasticBand(StrictModule, NonTrainableState):
    """Device NEB runtime bound to one program resource binding.

    `step` and `run` are pure; `init`, `grow` and `solve` are host
    transactions.  A capacity failure of any image leaves the whole group
    unconsumed (``CAPACITY_PENDING``) and `grow` replays it; any other image
    failure fails the band at its accepted iterate.

    PyTree: the binding digest is dynamic ``prepared_identity`` (host
    ``prepared_id``); ``layout_id`` is the content-free executable layout.
    """

    plan: AtomisticNudgedElasticBandPlan
    program: AbstractPreparedAtomisticHamiltonian
    layout: OptimizerGroupLayout
    prepared_identity: IdentityWords

    def __init__(
        self,
        plan: AtomisticNudgedElasticBandPlan,
        program: AbstractPreparedAtomisticHamiltonian,
        /,
    ) -> None:
        system = plan.surface.system
        if program.system.prepared_id != system.prepared_id:
            raise ValueError("The rebound program belongs to another prepared system.")
        groups = plan.group_count
        mask = np.asarray(system.active_mask, dtype=np.bool_) & np.asarray(
            system.mobile_mask, dtype=np.bool_
        )
        interior = plan.image_count - 2
        index = np.where(
            np.broadcast_to(mask, (groups, interior, mask.shape[0])),
            np.arange(groups)[:, None, None],
            -1,
        ).astype(np.int32)
        self.plan = plan
        self.program = program
        self.layout = OptimizerGroupLayout(index, group_count=groups)
        self.prepared_identity = IdentityWords.of(
            {
                "kind": "prepared-atomistic-nudged-elastic-band",
                "plan": plan.plan_id,
                "program_binding": program.prepared_id,
            }
        )

    @property
    def prepared_id(self) -> str:
        return self.prepared_identity.fingerprint()

    @property
    def layout_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "prepared-atomistic-nudged-elastic-band-layout",
                "plan": self.plan.layout_id,
                "program": self.program.layout_id,
                "bundle": self.program.neighborhood_bundle.layout_id,
            }
        )

    @property
    def system(self) -> PreparedAtomisticSystem:
        return self.program.system

    def _cell(self) -> Array | None:
        cell = self.system.cell
        return (
            None
            if cell is None
            else cell.vectors.astype(jnp.dtype(self.system.plan.coordinate_dtype))
        )

    def _path_mask(self) -> Array:
        return jnp.asarray(self.system.active_mask, dtype=jnp.bool_) & jnp.asarray(
            self.system.mobile_mask, dtype=jnp.bool_
        )

    def _evaluate(
        self,
        interior: Array,
        endpoints: Array,
        endpoint_energies: Array,
        resources: AtomisticNeighborhoodBundleState,
        climbing: Array,
        /,
    ) -> _BandEvaluation:
        bundle = self.program.neighborhood_bundle
        cell = self._cell()
        trial = jax.vmap(
            lambda position, accepted: evaluate_cartesian_trial(
                self.program, bundle, accepted, position, cell
            )
        )(interior, resources)
        band = jnp.concatenate((endpoints[:1], interior, endpoints[1:]))
        energies = jnp.concatenate(
            (endpoint_energies[:1], trial.energy, endpoint_energies[1:])
        )
        zero = jnp.zeros_like(endpoints[:1])
        forces = jnp.concatenate((zero, trial.forces, zero))
        projection = nudged_elastic_band_forces(
            band,
            energies,
            forces,
            self._path_mask(),
            spring_constant=self.plan.spring_constant,
            climbing=climbing,
        )
        capacity = jnp.any(trial.capacity)
        scientific = ~capacity & (jnp.any(trial.scientific) | ~projection.valid)
        return _BandEvaluation(
            trial.candidate, band, energies, forces, projection, capacity, scientific
        )

    def _endpoint_trial(
        self, images: Array, resources: AtomisticNeighborhoodBundleState, /
    ) -> Any:
        """Evaluate both endpoints of every band against their own resources."""
        count = self.plan.image_count
        bundle = self.program.neighborhood_bundle
        cell = self._cell()
        ends = (0, count - 1)
        return jax.vmap(
            jax.vmap(
                lambda position, accepted: evaluate_cartesian_trial(
                    self.program, bundle, accepted, position, cell
                )
            )
        )(images[:, jnp.asarray(ends)], _index_images(resources, ends))

    # -- lifecycle ---------------------------------------------------------------

    def init(
        self,
        endpoints: Sequence[tuple[AtomicStructure, AtomicStructure]],
        /,
        *,
        correspondences: Sequence[ConvertibleToArray | None] | None = None,
        branch: ReactionPathBranch | str = ReactionPathBranch.EXPLICIT,
        image_shifts: Sequence[ConvertibleToArray | None] | None = None,
        initial_images: ConvertibleToArray | None = None,
        active: ArrayLike | None = None,
        hamiltonian_lineage: tuple[str, ...] = (),
    ) -> tuple[
        PreparedAtomisticNudgedElasticBand,
        AtomisticReactionPathState,
        tuple[ReactionPathBranchCertificate, ...],
    ]:
        """Admit ``G`` whole paths, initialize per-image resources and endpoints.

        Resource capacities are grown on the host until every image of every
        active lane fits; an active lane's endpoint evaluation failure refuses
        the paths.  ``hamiltonian_lineage`` is the Hamiltonian history of the
        bank the paths join; their own counters start at zero.  Returns the
        possibly rebound runtime, the state and the branch certificates.
        """
        lineage = _identity_rows(tuple(hamiltonian_lineage))
        groups = self.plan.group_count
        count = self.plan.image_count
        plan_system = self.system.plan
        if len(endpoints) != groups:
            raise ValueError(f"endpoints must name {groups} reactant/product pairs.")
        correspondence_ = (
            (None,) * groups if correspondences is None else tuple(correspondences)
        )
        shifts_ = (None,) * groups if image_shifts is None else tuple(image_shifts)
        if len(correspondence_) != groups or len(shifts_) != groups:
            raise ValueError(
                "Per-group correspondences and shifts must name every group."
            )
        bands = []
        certificates = []
        active_atoms = np.asarray(plan_system.active_mask, dtype=np.bool_)
        for group, (reactant, product) in enumerate(endpoints):
            left, right, certificate = _path_endpoints(
                plan_system,
                reactant,
                product,
                correspondence=correspondence_[group],
                branch=branch,
                image_shifts=shifts_[group],
            )
            band = _initial_band(
                plan_system,
                left,
                right,
                count,
                None if initial_images is None else np.asarray(initial_images)[group],
            )
            bands.append(np.where(active_atoms[None, :, None], band, 0.0))
            certificates.append(certificate)
        stacked = jnp.asarray(np.stack(bands), dtype=plan_system.coordinate_dtype)
        lanes = (
            jnp.ones((groups,), dtype=jnp.bool_)
            if active is None
            else jnp.asarray(active, dtype=jnp.bool_)
        )
        if lanes.shape != (groups,):
            raise ValueError("active must have shape (group_count,).")
        cell = self._cell()
        program, resources, exhausted = _band_resources(
            self.program, stacked, np.asarray(lanes), cell
        )
        if resources is None:
            raise ValueError(
                f"The declared capacity ladder cannot hold the initial path of group {exhausted}."
            )
        runtime = (
            self
            if program.prepared_id == self.program.prepared_id
            else PreparedAtomisticNudgedElasticBand(self.plan, program)
        )
        terminal = runtime._endpoint_trial(stacked, resources)
        failed = (
            np.asarray(terminal.capacity | terminal.scientific)
            & np.asarray(lanes)[:, None]
        )
        if bool(np.any(failed)):
            raise ValueError("A path endpoint evaluation failed; the path is refused.")
        interior = stacked[:, 1:-1]
        interior_resources = _index_images(resources, tuple(range(1, count - 1)))
        dtype = interior.dtype
        optimizer = self.plan.method.init(
            interior, layout=runtime.layout, value_dtype=dtype
        )
        nan = jnp.full((groups, count), jnp.nan, dtype=dtype)
        zeros = jnp.zeros_like(stacked)
        none = jnp.zeros((groups,), dtype=jnp.int32)
        state = AtomisticReactionPathState(
            optimizer=optimizer,
            endpoints=stacked[:, jnp.asarray([0, count - 1])],
            endpoint_energies=terminal.energy,
            resources=interior_resources,
            candidate_resources=interior_resources,
            images=stacked,
            energies=nan,
            forces=zeros,
            neb_forces=zeros,
            tangents=zeros,
            image_residual=nan,
            climbing_image=jnp.full((groups,), max(1, count // 2), dtype=jnp.int32),
            climbing_active=jnp.zeros((groups,), dtype=jnp.bool_),
            status=jnp.where(
                lanes,
                int(AtomisticRelaxationStatus.ITERATING),
                int(AtomisticRelaxationStatus.INACTIVE),
            ).astype(jnp.int32),
            trial_outcome=none,
            active=lanes,
            capacity_replays=none,
            rejected_trials=none,
            reprimes=none,
            evaluation_base=none,
            iteration_base=none,
            plan_identity=self.plan.plan_identity,
            binding_identity=runtime.prepared_identity,
            lineage_identities=lineage,
        )
        return runtime, state, tuple(certificates)

    def _require_state(self, state: AtomisticReactionPathState, /) -> None:
        """Host transactions: exact refusal of a state of another plan or binding."""
        if not isinstance(state, AtomisticReactionPathState):
            raise TypeError("state must be an AtomisticReactionPathState.")
        if state.plan_id != self.plan.plan_id:
            raise ValueError("Path state belongs to another plan.")
        if state.binding_id != self.prepared_id:
            raise ValueError(
                "Path state belongs to another resource binding; use grow()."
            )

    def _bound(self, state: AtomisticReactionPathState, /) -> Array:
        """Traced: whether ``state`` belongs to this plan and resource binding."""
        if not isinstance(state, AtomisticReactionPathState):
            raise TypeError("state must be an AtomisticReactionPathState.")
        return state.plan_identity.equals(self.plan.plan_identity) & (
            state.binding_identity.equals(self.prepared_identity)
        )

    def step(self, state: AtomisticReactionPathState, /) -> AtomisticReactionPathState:
        """Evaluate every pending band once and commit accepted bands atomically.

        Iterating bands of a state bound to another plan or resource binding
        are not evaluated: they terminate with ``SCIENTIFIC_FAILURE`` (trial
        outcome ``SCIENTIFIC``) without consuming budget.
        """
        iterating = state.status == int(AtomisticRelaxationStatus.ITERATING)
        unbound = iterating & ~self._bound(state)
        pending = iterating & ~unbound
        climbing = state.evaluations >= self.plan.climbing_start
        trial = jax.vmap(self._evaluate)(
            state.optimizer.parameters,
            state.endpoints,
            state.endpoint_energies,
            state.resources,
            climbing,
        )
        consume = pending & ~trial.capacity
        valid = ~trial.capacity & ~trial.scientific
        residual = jnp.max(trial.projection.image_residual[:, 1:-1], axis=1)
        converged = valid & climbing & (residual <= self.plan.force_tolerance)
        nan = jnp.asarray(jnp.nan, dtype=trial.band.dtype)
        value = jnp.where(valid, jnp.sum(trial.energies[:, 1:-1], axis=1), nan)
        gradient = jnp.where(
            valid[:, None, None, None], -trial.projection.forces[:, 1:-1], nan
        )
        optimizer = _step_relaxation_method(
            self.plan.method,
            state.optimizer,
            value,
            gradient,
            converged=converged,
            active=consume,
        )
        accept = consume & valid & optimizer.evaluation_accepted
        bundle = self.program.neighborhood_bundle
        interior = self.plan.image_count - 2
        resources = jax.vmap(jax.vmap(bundle.select_cases))(
            jnp.broadcast_to(accept[:, None, None], (accept.shape[0], interior, 1)),
            trial.candidate,
            state.resources,
        )

        def keep(new: Array, old: Array) -> Array:
            shape = accept.shape + (1,) * (new.ndim - 1)
            return jnp.where(jnp.reshape(accept, shape), new, old)

        budget = (
            optimizer.evaluations + state.evaluation_base >= self.plan.maximum_evaluations
        )
        status = jnp.select(
            [
                unbound,
                pending & trial.capacity,
                consume & trial.scientific,
                consume & (optimizer.status == _SUCCESS),
                consume & budget,
            ],
            [
                int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE),
                int(AtomisticRelaxationStatus.CAPACITY_PENDING),
                int(AtomisticRelaxationStatus.SCIENTIFIC_FAILURE),
                int(AtomisticRelaxationStatus.CONVERGED),
                int(AtomisticRelaxationStatus.EVALUATION_LIMIT),
            ],
            state.status,
        ).astype(jnp.int32)
        outcome = jnp.select(
            [
                unbound,
                accept,
                pending & trial.capacity,
                consume & trial.scientific,
                consume,
            ],
            [
                int(AtomisticTrialOutcome.SCIENTIFIC),
                int(AtomisticTrialOutcome.ACCEPTED),
                int(AtomisticTrialOutcome.CAPACITY),
                int(AtomisticTrialOutcome.SCIENTIFIC),
                int(AtomisticTrialOutcome.ROLLED_BACK),
            ],
            int(AtomisticTrialOutcome.NONE),
        ).astype(jnp.int32)
        return AtomisticReactionPathState(
            optimizer=optimizer,
            endpoints=state.endpoints,
            endpoint_energies=state.endpoint_energies,
            resources=resources,
            candidate_resources=trial.candidate,
            images=keep(trial.band, state.images),
            energies=keep(trial.energies, state.energies),
            forces=keep(trial.forces, state.forces),
            neb_forces=keep(trial.projection.forces, state.neb_forces),
            tangents=keep(trial.projection.tangents, state.tangents),
            image_residual=keep(trial.projection.image_residual, state.image_residual),
            climbing_image=keep(trial.projection.climbing_image, state.climbing_image),
            climbing_active=keep(climbing, state.climbing_active),
            status=status,
            trial_outcome=jnp.where(iterating, outcome, state.trial_outcome).astype(
                jnp.int32
            ),
            active=state.active,
            capacity_replays=state.capacity_replays
            + (pending & trial.capacity).astype(jnp.int32),
            rejected_trials=state.rejected_trials + (consume & ~accept).astype(jnp.int32),
            reprimes=state.reprimes,
            evaluation_base=state.evaluation_base,
            iteration_base=state.iteration_base,
            plan_identity=state.plan_identity,
            binding_identity=state.binding_identity,
            lineage_identities=state.lineage_identities,
        )

    def run(
        self, state: AtomisticReactionPathState, /, *, maximum_rounds: int
    ) -> AtomisticReactionPathState:
        """Step on device until no group iterates or ``maximum_rounds`` pass."""
        rounds = int(maximum_rounds)
        if rounds < 1:
            raise ValueError("maximum_rounds must be positive.")
        if not isinstance(state, AtomisticReactionPathState):
            raise TypeError("state must be an AtomisticReactionPathState.")

        def condition(carry: tuple[Array, AtomisticReactionPathState]) -> Array:
            count, current = carry
            return (count < rounds) & jnp.any(
                current.status == int(AtomisticRelaxationStatus.ITERATING)
            )

        def body(
            carry: tuple[Array, AtomisticReactionPathState],
        ) -> tuple[Array, AtomisticReactionPathState]:
            count, current = carry
            return count + 1, self.step(current)

        _, final = jax.lax.while_loop(
            condition, body, (jnp.asarray(0, dtype=jnp.int32), state)
        )
        return final

    def grow(
        self, state: AtomisticReactionPathState, /
    ) -> tuple[PreparedAtomisticNudgedElasticBand, AtomisticReactionPathState]:
        """Host transaction: grow resources for capacity-pending bands and replay.

        Every capacity-failed image of a pending group is replayed from its
        accepted resources until it fits (an exhausted ladder terminates the
        group with ``RESOURCE_EXHAUSTED``); the program is rebound, all image
        resources are rebased and accepted band evidence is reprimed under the
        new binding without consuming budget.
        """
        self._require_state(state)
        status = np.asarray(state.status)
        pending = status == int(AtomisticRelaxationStatus.CAPACITY_PENDING)
        if not np.any(pending):
            return self, state
        bundle = self.program.neighborhood_bundle
        cell = self._cell()
        failed = np.asarray(state.candidate_resources.capacity_only)[..., 0]
        exhausted = np.zeros(status.shape, dtype=np.bool_)
        for group in np.flatnonzero(pending):
            for image in np.flatnonzero(failed[group]):
                position = state.optimizer.parameters[group, image]
                geometry, _, _ = trial_geometry(self.program, bundle, position, cell)
                bundle, ran_out = replay_capacity(
                    bundle,
                    _index_state(state.resources, int(group), int(image)),
                    geometry,
                )
                exhausted[group] |= ran_out
        runtime = PreparedAtomisticNudgedElasticBand(
            self.plan, self.program.with_neighborhood_bundle(bundle)
        )
        groups, interior = failed.shape
        rebased = jax.tree.map(
            lambda *leaves: jnp.stack(leaves).reshape(
                (groups, interior) + leaves[0].shape
            ),
            *[
                bundle.rebase(_index_state(state.resources, group, image))
                for group in range(groups)
                for image in range(interior)
            ],
        )
        reprime = jax.vmap(runtime._evaluate)(
            state.images[:, 1:-1],
            state.endpoints,
            state.endpoint_energies,
            rebased,
            state.climbing_active,
        )
        has_accepted = state.optimizer.has_accepted
        refreshed = has_accepted & ~reprime.capacity & ~reprime.scientific

        def refresh(new: Array, old: Array) -> Array:
            shape = refreshed.shape + (1,) * (new.ndim - 1)
            return jnp.where(jnp.reshape(refreshed, shape), new, old)

        new_status = jnp.where(
            jnp.asarray(exhausted),
            int(AtomisticRelaxationStatus.RESOURCE_EXHAUSTED),
            jnp.where(
                jnp.asarray(pending),
                int(AtomisticRelaxationStatus.ITERATING),
                state.status,
            ),
        ).astype(jnp.int32)
        return runtime, AtomisticReactionPathState(
            optimizer=state.optimizer,
            endpoints=state.endpoints,
            endpoint_energies=state.endpoint_energies,
            resources=rebased,
            candidate_resources=rebased,
            images=state.images,
            energies=refresh(reprime.energies, state.energies),
            forces=refresh(reprime.forces, state.forces),
            neb_forces=refresh(reprime.projection.forces, state.neb_forces),
            tangents=refresh(reprime.projection.tangents, state.tangents),
            image_residual=refresh(
                reprime.projection.image_residual, state.image_residual
            ),
            climbing_image=refresh(
                reprime.projection.climbing_image, state.climbing_image
            ),
            climbing_active=state.climbing_active,
            status=new_status,
            trial_outcome=state.trial_outcome,
            active=state.active,
            capacity_replays=state.capacity_replays,
            rejected_trials=state.rejected_trials,
            reprimes=state.reprimes + refreshed.astype(jnp.int32),
            evaluation_base=state.evaluation_base,
            iteration_base=state.iteration_base,
            plan_identity=state.plan_identity,
            binding_identity=runtime.prepared_identity,
            lineage_identities=state.lineage_identities,
        )

    def solve(
        self,
        state: AtomisticReactionPathState,
        /,
        *,
        maximum_rounds: int,
        maximum_growths: int = 8,
    ) -> tuple[PreparedAtomisticNudgedElasticBand, AtomisticReactionPathState]:
        """Host driver: compiled `run` segments separated by `grow` transactions.

        Every runtime of one layout shares the single compiled `run` entrypoint.
        """
        self._require_state(state)
        runtime = self
        for _ in range(int(maximum_growths) + 1):
            state = _RUN_BAND_ROUNDS(runtime, state, int(maximum_rounds))
            if not np.any(
                np.asarray(state.status)
                == int(AtomisticRelaxationStatus.CAPACITY_PENDING)
            ):
                break
            runtime, state = runtime.grow(state)
        return runtime, state

    def transition_hamiltonian(
        self,
        program: AbstractPreparedAtomisticHamiltonian,
        state: AtomisticReactionPathState,
        /,
    ) -> AtomisticPathHamiltonianTransitionResult:
        """Host transaction: continue every band under a new prepared Hamiltonian.

        Same contract as `PreparedAtomisticRelaxation.transition_hamiltonian`
        (scientific-model or numerical-realization ``kind`` by the canonical
        ``scientific_id``; resource-only rebinding refused),
        applied band-atomically: for every revisable (iterating, converged or
        capacity-pending, active) group the accepted band is kept, resources of
        all its images are initialized and grown under the new program, both
        endpoint energies and every interior image are re-evaluated without
        consuming budget, and the fresh optimizer (no L-BFGS secants, FIRE
        velocity or controller carried over) takes that evaluation first with
        the group's climbing flag.  Any image or endpoint failure of any
        revisable group refuses the whole transition and returns the previous
        runtime and state unchanged.  Terminal and inactive groups keep band,
        evidence and status.
        """
        self._require_state(state)
        kind = hamiltonian_transition_kind(self.program, program)
        previous_id = self.program.scientific_id
        scientific_id = program.scientific_id
        groups = self.plan.group_count
        count = self.plan.image_count
        status = np.asarray(state.status)
        revisable = np.asarray(state.active, dtype=np.bool_) & np.isin(
            status,
            (
                int(AtomisticRelaxationStatus.ITERATING),
                int(AtomisticRelaxationStatus.CONVERGED),
                int(AtomisticRelaxationStatus.CAPACITY_PENDING),
            ),
        )
        reprimed = revisable & np.asarray(state.optimizer.has_accepted, dtype=np.bool_)

        def refuse(reason: str) -> AtomisticPathHamiltonianTransitionResult:
            return AtomisticPathHamiltonianTransitionResult(
                self,
                state,
                AtomisticHamiltonianTransition(
                    np.zeros((groups,), dtype=np.bool_),
                    successful=False,
                    reason=reason,
                    kind=kind,
                    previous_scientific_id=previous_id,
                    scientific_id=scientific_id,
                    previous_binding_id=self.prepared_id,
                    binding_id=self.prepared_id,
                    previous_plan_id=self.plan.plan_id,
                    plan_id=self.plan.plan_id,
                    lineage=state.hamiltonian_lineage,
                ),
            )

        try:
            plan = AtomisticNudgedElasticBandPlan(
                AtomisticPotentialEnergySurface(self.plan.surface.system, program),
                image_count=count,
                spring_constant=float(self.plan.spring_constant),
                force_tolerance=float(self.plan.force_tolerance),
                maximum_evaluations=int(self.plan.maximum_evaluations),
                climbing_start=int(self.plan.climbing_start),
                group_count=groups,
                method=self.plan.method,
            )
        except ValueError as error:
            return refuse(f"The path plan refuses the new Hamiltonian: {error}")
        bound, resources, exhausted = _band_resources(
            program, state.images, revisable, self._cell()
        )
        if resources is None:
            return refuse(
                f"The declared capacities cannot hold the band of group {exhausted} "
                "under the new Hamiltonian."
            )
        runtime = PreparedAtomisticNudgedElasticBand(plan, bound)
        terminal = runtime._endpoint_trial(state.images, resources)
        ends_failed = revisable & np.any(
            np.asarray(terminal.capacity | terminal.scientific), axis=1
        )
        if np.any(ends_failed):
            return refuse(
                "Path endpoints of groups "
                f"{np.flatnonzero(ends_failed).tolist()} fail under the new Hamiltonian."
            )
        lanes = jnp.asarray(revisable)
        endpoint_energies = jnp.where(
            lanes[:, None], terminal.energy, state.endpoint_energies
        )
        interior_resources = _index_images(resources, tuple(range(1, count - 1)))
        reprime = jax.vmap(runtime._evaluate)(
            state.images[:, 1:-1],
            state.endpoints,
            endpoint_energies,
            interior_resources,
            state.climbing_active,
        )
        broken = reprimed & np.asarray(reprime.capacity | reprime.scientific)
        if np.any(broken):
            return refuse(
                "Accepted bands of groups "
                f"{np.flatnonzero(broken).tolist()} fail under the new Hamiltonian."
            )
        mask = jnp.asarray(reprimed)
        valid = ~reprime.capacity & ~reprime.scientific
        residual = jnp.max(reprime.projection.image_residual[:, 1:-1], axis=1)
        converged = valid & state.climbing_active & (residual <= plan.force_tolerance)
        nan = jnp.asarray(jnp.nan, dtype=reprime.band.dtype)
        interior = state.images[:, 1:-1]
        optimizer = plan.method.init(
            interior, layout=runtime.layout, value_dtype=interior.dtype
        )
        optimizer = _step_relaxation_method(
            plan.method,
            optimizer,
            jnp.where(valid, jnp.sum(reprime.energies[:, 1:-1], axis=1), nan),
            jnp.where(
                valid[:, None, None, None], -reprime.projection.forces[:, 1:-1], nan
            ),
            converged=mask & converged,
            active=mask,
        )
        # The reprime is not consumed work: totals continue unchanged.
        evaluation_base = state.evaluations - optimizer.evaluations
        budget = optimizer.evaluations + evaluation_base >= plan.maximum_evaluations
        new_status = jnp.where(
            lanes,
            jnp.where(
                mask & (optimizer.status == _SUCCESS),
                int(AtomisticRelaxationStatus.CONVERGED),
                jnp.where(
                    budget,
                    int(AtomisticRelaxationStatus.EVALUATION_LIMIT),
                    int(AtomisticRelaxationStatus.ITERATING),
                ),
            ),
            state.status,
        ).astype(jnp.int32)

        def keep(new: Array, old: Array) -> Array:
            shape = mask.shape + (1,) * (new.ndim - 1)
            return jnp.where(jnp.reshape(mask, shape), new, old)

        record = AtomisticHamiltonianTransition(
            reprimed,
            successful=True,
            reason="",
            kind=kind,
            previous_scientific_id=previous_id,
            scientific_id=scientific_id,
            previous_binding_id=self.prepared_id,
            binding_id=runtime.prepared_id,
            previous_plan_id=self.plan.plan_id,
            plan_id=plan.plan_id,
            lineage=state.hamiltonian_lineage,
        )
        continued = AtomisticReactionPathState(
            optimizer=optimizer,
            endpoints=state.endpoints,
            endpoint_energies=endpoint_energies,
            resources=interior_resources,
            candidate_resources=interior_resources,
            images=state.images,
            energies=keep(reprime.energies, state.energies),
            forces=keep(reprime.forces, state.forces),
            neb_forces=keep(reprime.projection.forces, state.neb_forces),
            tangents=keep(reprime.projection.tangents, state.tangents),
            image_residual=keep(reprime.projection.image_residual, state.image_residual),
            climbing_image=keep(reprime.projection.climbing_image, state.climbing_image),
            climbing_active=state.climbing_active,
            status=new_status,
            trial_outcome=jnp.where(
                lanes, int(AtomisticTrialOutcome.NONE), state.trial_outcome
            ).astype(jnp.int32),
            active=state.active,
            capacity_replays=state.capacity_replays,
            rejected_trials=state.rejected_trials,
            reprimes=state.reprimes + mask.astype(jnp.int32),
            evaluation_base=evaluation_base.astype(jnp.int32),
            iteration_base=state.iterations.astype(jnp.int32),
            plan_identity=plan.plan_identity,
            binding_identity=runtime.prepared_identity,
            lineage_identities=_identity_rows(
                (*state.hamiltonian_lineage, record.transition_id)
            ),
        )
        return AtomisticPathHamiltonianTransitionResult(runtime, continued, record)

    # -- observations ----------------------------------------------------------------

    def path_result(
        self, state: AtomisticReactionPathState, group: int, /
    ) -> ReactionPathResult:
        """Host `ReactionPathResult` of one group's accepted band."""
        self._require_state(state)
        index = int(group)
        host = jax.device_get(
            (
                state.images[index],
                state.energies[index],
                state.forces[index],
                state.neb_forces[index],
                state.tangents[index],
                state.image_residual[index],
                state.iterations[index],
                state.evaluations[index],
                state.capacity_replays[index],
                state.status[index],
                state.climbing_image[index],
                state.climbing_active[index],
            )
        )
        (
            images,
            energies,
            forces,
            neb,
            tangents,
            residual,
            iterations,
            evaluations,
            replays,
            status,
            climbing,
            climbing_active,
        ) = host
        interior = self.plan.image_count - 2
        source = canonical_fingerprint(
            {
                "kind": "native-reaction-path-evaluation",
                "binding": self.prepared_id,
                "arrays": array_tree_fingerprint(
                    {"images": images, "energies": energies, "forces": forces}
                ),
            }
        )
        return ReactionPathResult(
            images,
            energies,
            forces,
            neb,
            tangents,
            residual,
            int(iterations),
            (int(evaluations) + int(replays)) * interior + 2,
            int(status) == int(AtomisticRelaxationStatus.CONVERGED),
            status=int(status),
            climbing_image=int(climbing),
            climbing_active=bool(climbing_active),
            source_result_ids=(source,),
            plan_id=self.plan.plan_id,
        )

    def transition_state_inertia(
        self,
        state: AtomisticReactionPathState,
        group: int,
        /,
        *,
        gauge: AtomisticGaugePolicy | None = None,
        policy: AtomisticStationaryPolicy | None = None,
    ) -> AtomisticStationaryEvidence:
        """HVP/inertia evidence that the climbing image is a first-order saddle.

        The Hessian action is taken on the free (mobile) non-gauge space of the
        climbing image at its accepted resources; a converged climbing band is
        required before any saddle claim.
        """
        self._require_state(state)
        index = int(group)
        climbing = int(np.asarray(state.climbing_image)[index])
        resources = _index_state(state.resources, index, climbing - 1)
        positions = state.images[index, climbing]
        cell = self._cell()
        _, _, image_counts = trial_geometry(
            self.program, self.program.neighborhood_bundle, positions, cell
        )
        shape = positions.shape

        def objective(flat: Array) -> Array:
            energy, _ = program_energy(
                self.program, resources, flat.reshape(shape), cell, image_counts
            )
            return energy

        mask = np.asarray(self._path_mask())
        free = np.repeat(mask, 3).astype(np.asarray(positions).dtype)
        gauge_ = AtomisticGaugePolicy() if gauge is None else gauge
        if gauge_.rotations and self.system.cell is not None:
            raise ValueError("A rotation gauge exists only for finite systems.")
        columns = []
        weight = mask.astype(free.dtype)[:, None]
        if gauge_.translations:
            for axis in range(3):
                columns.append((np.eye(3)[axis][None] * weight).reshape((-1,)))
        if gauge_.rotations:
            point = np.asarray(positions)
            center = np.sum(weight * point, axis=0) / max(float(np.sum(weight)), 1.0)
            for axis in range(3):
                columns.append(
                    (np.cross(np.eye(3)[axis], (point - center)) * weight).reshape((-1,))
                )
        converged = int(np.asarray(state.status)[index]) == int(
            AtomisticRelaxationStatus.CONVERGED
        ) and bool(np.asarray(state.climbing_active)[index])
        return atomistic_stationary_inertia(
            objective,
            positions.reshape((-1,)),
            free=free,
            gauge_basis=None if not columns else np.stack(columns, axis=1),
            policy=policy,
            saddle=True,
            converged=converged,
        )


class AtomisticPathHamiltonianTransitionResult(NamedTuple):
    """Runtime, state and lineage record of a band Hamiltonian transition."""

    runtime: PreparedAtomisticNudgedElasticBand
    state: AtomisticReactionPathState
    record: AtomisticHamiltonianTransition


def _band_resources(
    program: AbstractPreparedAtomisticHamiltonian,
    images: Array,
    lanes: np.ndarray,
    cell: Array | None,
    /,
) -> tuple[
    AbstractPreparedAtomisticHamiltonian,
    AtomisticNeighborhoodBundleState | None,
    int | None,
]:
    """Host: initialize every image's resources, growing capacity for ``lanes``.

    Declared capacities grow until every image of every selected lane fits;
    other lanes are initialized under the final bundle as they are.  Returns
    the (possibly rebound) program, ``(G, M)`` resources, or ``None`` and the
    first group whose declared ladder is exhausted.
    """
    selected = np.asarray(lanes, dtype=np.bool_)
    while True:
        bundle = program.neighborhood_bundle
        resources = jax.vmap(
            jax.vmap(
                lambda position, bundle=bundle, program=program: bundle.initialize(
                    trial_geometry(program, bundle, position, cell)[0]
                )
            )
        )(images)
        failing = np.asarray(resources.capacity_only)[..., 0] & selected[:, None]
        if not np.any(failing):
            return program, resources, None
        for group, image in zip(*np.nonzero(failing), strict=True):
            geometry, _, _ = trial_geometry(program, bundle, images[group, image], cell)
            bundle, exhausted = initialize_capacity(bundle, geometry)
            if exhausted:
                return program, None, int(group)
        program = program.with_neighborhood_bundle(bundle)


def _index_images(
    resources: AtomisticNeighborhoodBundleState, images: tuple[int, ...], /
) -> AtomisticNeighborhoodBundleState:
    selected = jnp.asarray(images)
    return jax.tree.map(lambda leaf: leaf[:, selected], resources)


def _index_state(
    resources: AtomisticNeighborhoodBundleState, group: int, image: int, /
) -> AtomisticNeighborhoodBundleState:
    return jax.tree.map(lambda leaf: leaf[group, image], resources)


def _run_band_rounds(
    runtime: PreparedAtomisticNudgedElasticBand,
    state: AtomisticReactionPathState,
    rounds: int,
    /,
) -> AtomisticReactionPathState:
    return runtime.run(state, maximum_rounds=rounds)


_RUN_BAND_ROUNDS = eqx.filter_jit(_run_band_rounds)


class ReactionPathQualificationResult(StrictModule, NonTrainableState):
    tangent_overlap: Array
    successful: Array
    path_result_id: str = eqx.field(static=True)
    vibration_result_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: ReactionPathResult,
        vibration: VibrationalAnalysisResult,
        masses: ArrayLike,
        /,
        *,
        minimum_overlap: float = 0.5,
    ) -> None:
        threshold = float(minimum_overlap)
        mass = np.asarray(masses)
        if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_overlap must be finite and lie in [0, 1].")
        if (
            mass.shape != path.images.shape[1:2]
            or np.any(~np.isfinite(mass))
            or np.any(mass <= 0.0)
        ):
            raise ValueError("Reaction-path qualification masses are invalid.")
        if (
            vibration.stationary_point is not StationaryPointKind.FIRST_ORDER_SADDLE
            or not np.any(np.asarray(vibration.imaginary_mask))
        ):
            overlap = 0.0
            successful = False
        else:
            index = path.climbing_image
            tangent = np.asarray(path.images[index + 1] - path.images[index - 1])
            imaginary_index = int(np.flatnonzero(np.asarray(vibration.imaginary_mask))[0])
            mode = np.asarray(vibration.normal_modes)[..., imaginary_index]
            root_mass = np.sqrt(mass)[:, None]
            tangent_mass = root_mass * tangent
            mode_mass = root_mass * mode
            denominator = float(
                np.sqrt(
                    np.sum(tangent_mass * tangent_mass) * np.sum(mode_mass * mode_mass)
                )
            )
            overlap = (
                0.0
                if denominator == 0.0
                else abs(float(np.sum(tangent_mass * mode_mass))) / denominator
            )
            successful = (
                bool(path.successful)
                and bool(vibration.successful)
                and overlap >= threshold
            )
        self.tangent_overlap = jnp.asarray(overlap)
        self.successful = jnp.asarray(successful, dtype=jnp.bool_)
        self.path_result_id = path.result_id
        self.vibration_result_id = vibration.result_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "reaction-path-qualification",
                "path": path.result_id,
                "vibration": vibration.result_id,
                "tangent_overlap": overlap,
                "successful": successful,
            }
        )


class IntrinsicReactionCoordinateResult(StrictModule, NonTrainableState):
    forward: Array
    reverse: Array
    forward_energies: Array
    reverse_energies: Array
    successful: Array
    source_result_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward: ArrayLike,
        reverse: ArrayLike,
        forward_energies: ArrayLike,
        reverse_energies: ArrayLike,
        successful: ArrayLike,
        source_result_ids: tuple[str, ...],
        /,
    ) -> None:
        forward_ = jnp.asarray(forward)
        reverse_ = jnp.asarray(reverse, dtype=forward_.dtype)
        forward_energy = jnp.asarray(forward_energies, dtype=forward_.dtype)
        reverse_energy = jnp.asarray(reverse_energies, dtype=forward_.dtype)
        if forward_.ndim != 3 or reverse_.ndim != 3:
            raise ValueError("IRC branches must have shape (step, atom, 3).")
        if forward_energy.shape != (forward_.shape[0],) or reverse_energy.shape != (
            reverse_.shape[0],
        ):
            raise ValueError("IRC energies must align with branch steps.")
        self.forward = forward_
        self.reverse = reverse_
        self.forward_energies = forward_energy
        self.reverse_energies = reverse_energy
        self.successful = jnp.asarray(successful, dtype=jnp.bool_)
        self.source_result_ids = source_result_ids
        self.result_id = canonical_fingerprint(
            {
                "kind": "intrinsic-reaction-coordinate-result",
                "sources": list(source_result_ids),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "forward": np.asarray(forward_),
                        "reverse": np.asarray(reverse_),
                        "forward_energies": np.asarray(forward_energy),
                        "reverse_energies": np.asarray(reverse_energy),
                    }
                ),
            }
        )


class IntrinsicReactionCoordinatePlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    surface: AbstractPreparedPotentialEnergySurface
    step_size: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    force_tolerance: float = eqx.field(static=True)
    corrector_iterations: int = eqx.field(static=True)
    corrector_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        surface: AbstractPreparedPotentialEnergySurface,
        /,
        *,
        step_size: float = 0.02,
        maximum_steps: int = 200,
        force_tolerance: float = 1.0e-3,
        corrector_iterations: int = 8,
        corrector_tolerance: float = 1.0e-4,
    ) -> None:
        if surface.system_id != system.system_id or not surface.capabilities.forces:
            raise ValueError("IRC requires a force-capable surface for the same system.")
        if surface.units.unit_system_id != system.units.unit_system_id:
            raise ValueError("IRC surface and system unit identities differ.")
        step = float(step_size)
        tolerance = float(force_tolerance)
        steps = int(maximum_steps)
        corrector_steps = int(corrector_iterations)
        corrector = float(corrector_tolerance)
        if (
            not isfinite(step)
            or step <= 0.0
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or not isfinite(corrector)
            or corrector <= 0.0
        ):
            raise ValueError("IRC step and tolerances must be positive finite.")
        if steps <= 0 or corrector_steps < 0:
            raise ValueError(
                "IRC steps must be positive and corrector_iterations non-negative."
            )
        self.system = system
        self.surface = surface
        self.step_size = step
        self.maximum_steps = steps
        self.force_tolerance = tolerance
        self.corrector_iterations = corrector_steps
        self.corrector_tolerance = corrector
        self.plan_id = canonical_fingerprint(
            {
                "kind": "intrinsic-reaction-coordinate-plan",
                "system": system.system_id,
                "surface": surface.surface_id,
                "step_size": step,
                "maximum_steps": steps,
                "force_tolerance": tolerance,
                "corrector_iterations": corrector_steps,
                "corrector_tolerance": corrector,
            }
        )

    def run(
        self,
        transition_state: AtomicStructure,
        imaginary_mode: ArrayLike,
        /,
    ) -> IntrinsicReactionCoordinateResult:
        _require_structure_matches_system(transition_state, self.system)
        mode = np.array(
            imaginary_mode,
            dtype=np.dtype(self.system.coordinate_dtype),
            copy=True,
        )
        if mode.shape != transition_state.positions.shape:
            raise ValueError("imaginary_mode must align with transition-state positions.")
        masses = np.asarray(self.system.masses)[:, None]
        active = np.asarray(self.system.active_mask, dtype=np.bool_)
        movable = np.asarray(self.system.mobile_mask, dtype=np.bool_)
        mode[~movable] = 0.0
        norm = float(np.sqrt(np.sum(masses[active] * mode[active] ** 2)))
        if not isfinite(norm) or norm <= 0.0:
            raise ValueError("imaginary_mode has zero or nonfinite mass-weighted norm.")
        mode /= norm
        cell = (
            None if transition_state.cell is None else np.asarray(transition_state.cell)
        )
        source_ids: list[str] = []

        def branch(direction: float) -> tuple[np.ndarray, np.ndarray, bool]:
            position = (
                np.asarray(transition_state.positions).copy()
                + direction * self.step_size * mode
            )
            positions = [position.copy()]
            energies = []
            converged = False
            for iteration in range(self.maximum_steps):
                value = self.surface.evaluate(position, cell)
                source_ids.append(value.source_result_id)
                energies.append(float(value.energy))
                if not bool(value.successful):
                    break
                force = np.asarray(value.forces).copy()
                force[~movable] = 0.0
                maximum = float(np.max(np.abs(force[active])))
                if maximum <= self.force_tolerance:
                    converged = True
                    break
                mass_direction = force / masses
                mass_direction[~movable] = 0.0
                direction_norm = float(
                    np.sqrt(np.sum(masses[active] * mass_direction[active] ** 2))
                )
                if not isfinite(direction_norm) or direction_norm <= 0.0:
                    break
                origin = position.copy()
                candidate = origin + self.step_size * mass_direction / direction_norm
                for _ in range(self.corrector_iterations):
                    corrected = self.surface.evaluate(candidate, cell)
                    source_ids.append(corrected.source_result_id)
                    if not bool(corrected.successful):
                        break
                    radial = candidate - origin
                    radial[~movable] = 0.0
                    radial_norm = float(
                        np.sqrt(np.sum(masses[active] * radial[active] ** 2))
                    )
                    if not isfinite(radial_norm) or radial_norm <= 0.0:
                        break
                    tangent = radial / radial_norm
                    corrected_direction = np.asarray(corrected.forces) / masses
                    corrected_direction[~movable] = 0.0
                    projection = float(
                        np.sum(
                            masses[active] * corrected_direction[active] * tangent[active]
                        )
                    )
                    perpendicular = corrected_direction - projection * tangent
                    perpendicular_norm = float(
                        np.sqrt(np.sum(masses[active] * perpendicular[active] ** 2))
                    )
                    if perpendicular_norm <= self.corrector_tolerance:
                        break
                    candidate = candidate + 0.5 * self.step_size * perpendicular
                    radial = candidate - origin
                    radial_norm = float(
                        np.sqrt(np.sum(masses[active] * radial[active] ** 2))
                    )
                    candidate = origin + self.step_size * radial / radial_norm
                position = candidate
                if iteration + 1 < self.maximum_steps:
                    positions.append(position.copy())
            return np.asarray(positions), np.asarray(energies), converged

        forward, forward_energy, forward_success = branch(1.0)
        reverse, reverse_energy, reverse_success = branch(-1.0)
        return IntrinsicReactionCoordinateResult(
            forward,
            reverse,
            forward_energy,
            reverse_energy,
            forward_success and reverse_success,
            tuple(source_ids),
        )


__all__ = [
    "AtomisticNudgedElasticBandPlan",
    "AtomisticPathHamiltonianTransitionResult",
    "AtomisticReactionPathState",
    "IntrinsicReactionCoordinatePlan",
    "IntrinsicReactionCoordinateResult",
    "nudged_elastic_band_forces",
    "NudgedElasticBandPlan",
    "NudgedElasticBandProjection",
    "PreparedAtomisticNudgedElasticBand",
    "ReactionPathBranch",
    "ReactionPathBranchCertificate",
    "ReactionPathQualificationResult",
    "ReactionPathResult",
    "resolve_reaction_path_branch",
]
