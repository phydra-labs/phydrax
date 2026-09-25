#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared-face closures of conservative finite-volume fluxes.

A face closure corrects the one baseline normal flux density evaluated per
shared face, so conservation follows from single face evaluation. Closures act
on arbitrary oriented faces: every owner (Cartesian, mapped, triangular,
unstructured, moving, and overset faces) supplies a `FaceFluxContext` carrying
the unit normal, the positive face measure, the grid-normal velocity, an
optional Cartesian axis identity, and the geometry/frame identity.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, ClassVar, final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._differentiation import (
    _identifier,
    AbstractConstructionCertificate,
    branch_policy_contract,
    BranchDifferentiationPolicy,
    ComponentAuthority,
    DerivativeContract,
    DerivativeSurface,
)
from ..._fingerprint import canonical_fingerprint
from ..._model import AbstractComponentSlot
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


FaceClosureFrame: TypeAlias = Literal["global", "face-normal"]

# Unit-normal admission tolerance in units of the normal dtype epsilon; owners
# form unit normals as area vectors divided by their measures.
_UNIT_NORMAL_EPSILONS = 256.0


def _require_frame(frame: Any, /) -> FaceClosureFrame:
    match frame:
        case "global" | "face-normal":
            return frame
        case _:
            raise ValueError("frame must be 'global' or 'face-normal'.")


@final
class FaceFluxContext(StrictModule, NonTrainableState):
    """Oriented geometry of one batch of faces at which a closure is evaluated.

    `unit_normal` has shape `batch + (dimension,)` and points from the left
    (owner) trace to the right (neighbor) trace; orientation lives only in the
    normal. `face_measure` is the positive face measure, `grid_normal_velocity`
    the face velocity projected on the normal (zero on stationary faces), and
    `active` the support mask: inactive faces carry no geometry and receive no
    correction. `axis` identifies a Cartesian face family whose normal is
    `+-e_axis`; `geometry_id` identifies the prepared geometry and `frame` the
    frame in which the normal and vector components are expressed (`"global"`
    physical coordinates, or `"face-normal"` where the normal is `e_0`).

    Construction validates finite unit normals, positive finite measures, finite
    grid-normal velocities, and the Cartesian axis identity on active faces.
    """

    unit_normal: Array
    face_measure: Array
    grid_normal_velocity: Array
    active: Array
    axis: int | None = eqx.field(static=True)
    frame: FaceClosureFrame = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        unit_normal: ArrayLike,
        face_measure: ArrayLike,
        grid_normal_velocity: ArrayLike | None = None,
        /,
        *,
        geometry_id: str,
        axis: int | None = None,
        active: ArrayLike | None = None,
        frame: FaceClosureFrame = "global",
    ):
        normal = jnp.asarray(unit_normal)
        if normal.ndim < 1 or not jnp.issubdtype(normal.dtype, jnp.floating):
            raise TypeError("unit_normal must be a real floating array.")
        batch = normal.shape[:-1]
        dimension = normal.shape[-1]
        measure = jnp.asarray(face_measure)
        velocity = (
            jnp.zeros(batch, dtype=normal.dtype)
            if grid_normal_velocity is None
            else jnp.asarray(grid_normal_velocity)
        )
        support = (
            jnp.ones(batch, dtype=jnp.bool_)
            if active is None
            else jnp.asarray(active, dtype=jnp.bool_)
        )
        if measure.shape != batch or velocity.shape != batch or support.shape != batch:
            raise ValueError(
                "Face measure, grid-normal velocity, and support must match the "
                f"face batch shape {batch}; broadcasting is not permitted."
            )
        if not jnp.issubdtype(measure.dtype, jnp.floating) or not jnp.issubdtype(
            velocity.dtype, jnp.floating
        ):
            raise TypeError("Face measure and grid-normal velocity must be floating.")
        frame_ = _require_frame(frame)
        geometry = _identifier(geometry_id, "geometry_id")
        if axis is not None and (
            isinstance(axis, bool)
            or not isinstance(axis, int)
            or not 0 <= axis < dimension
        ):
            raise ValueError("axis must identify one spatial axis of the normal.")
        squared = ein.contract("...d,...d->...", normal, normal, backend="jax")
        tolerance = _UNIT_NORMAL_EPSILONS * jnp.finfo(normal.dtype).eps
        invalid = (
            ~jnp.isfinite(squared)
            | (jnp.abs(squared - 1.0) > tolerance)
            | ~jnp.isfinite(measure)
            | (measure <= 0.0)
            | ~jnp.isfinite(velocity)
        )
        if axis is not None and frame_ == "global":
            tangential = jnp.where(np.arange(dimension) == axis, 0.0, normal)
            invalid = invalid | jnp.any(tangential != 0.0, axis=-1)
        normal = eqx.error_if(
            normal,
            jnp.any(support & invalid),
            "Face closure context requires finite unit normals, positive finite "
            "face measures, finite grid-normal velocities, and exact Cartesian "
            "axis normals on active faces.",
        )
        self.unit_normal = normal
        self.face_measure = measure
        self.grid_normal_velocity = velocity
        self.active = support
        self.axis = axis
        self.frame = frame_
        self.geometry_id = geometry

    @classmethod
    def cartesian(
        cls,
        axis: int,
        dimension: int,
        face_measure: ArrayLike,
        /,
        *,
        geometry_id: str,
        active: ArrayLike | None = None,
    ) -> FaceFluxContext:
        """Stationary context of a Cartesian face family with normal `+e_axis`."""
        measure = jnp.asarray(face_measure)
        normal = jnp.broadcast_to(
            jnp.zeros((dimension,), dtype=measure.dtype).at[axis].set(1.0),
            measure.shape + (dimension,),
        )
        return cls(normal, measure, geometry_id=geometry_id, axis=axis, active=active)

    @property
    def dimension(self) -> int:
        return self.unit_normal.shape[-1]

    def reversed(self) -> FaceFluxContext:
        """The same faces seen from the opposite side (`n -> -n`, `w -> -w`)."""
        return FaceFluxContext(
            -self.unit_normal,
            self.face_measure,
            -self.grid_normal_velocity,
            geometry_id=self.geometry_id,
            axis=self.axis,
            active=self.active,
            frame=self.frame,
        )

    def in_normal_frame(self) -> FaceFluxContext:
        """The context expressed in the face-normal frame, where the normal is `e_0`."""
        normal = jnp.broadcast_to(
            jnp.zeros((self.dimension,), dtype=self.unit_normal.dtype).at[0].set(1.0),
            self.unit_normal.shape,
        )
        return FaceFluxContext(
            normal,
            self.face_measure,
            self.grid_normal_velocity,
            geometry_id=self.geometry_id,
            axis=self.axis,
            active=self.active,
            frame="face-normal",
        )


class AbstractFaceClosurePlan(AbstractComponentSlot):
    """Neutral slot correcting the shared baseline normal flux of each face.

    `apply(system, left, right, baseline_normal_flux, context, args)` returns the
    corrected normal flux density. An implementation is evaluated once per shared
    face, so any correction is conservative; `admit_system` refuses conservation
    systems the closure cannot correct before execution.
    """

    component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.DISCRETIZATION
    slot_semantic_id: ClassVar[str] = "discretization.face-closure"

    closure_id: eqx.AbstractVar[str]

    @property
    @abc.abstractmethod
    def derivative_contract(self) -> DerivativeContract:
        raise NotImplementedError

    @abc.abstractmethod
    def admit_system(self, system: Any, /) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def apply(
        self,
        system: Any,
        left: Array,
        right: Array,
        baseline_normal_flux: Array,
        context: FaceFluxContext,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError


@final
class SymmetrizedFaceClosureCertificate(AbstractConstructionCertificate):
    """Construction evidence that a correction is consistent and antisymmetric.

    The symmetrized construction satisfies `C(u, u, n) = 0` and
    `C(a, b, n) = -C(b, a, -n)` exactly for every generator.
    """

    capability_id: ClassVar[str] = "face-closure-consistency-antisymmetry"
    certificate_id: str = eqx.field(static=True)

    def __init__(self):
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "symmetrized-face-closure-certificate",
                "construction": "half-difference-of-midpoint-subtracted-generator",
            }
        )


@final
class SymmetrizedFaceClosure(StrictModule):
    """Correction built from a generator `g` to be consistent and antisymmetric.

    With `m = (a + b)/2` and `h(a, b, n) = g(a, b, n) - g(m, m, n)`, the
    correction is `C(a, b, n) = 1/2 [h(a, b, n) - h(b, a, -n)]`. The reversed
    evaluation receives the reversed baseline flux and context. Consistency and
    orientation antisymmetry hold by construction, which the attached
    `certificate` records; the generator is a dynamic child, so its model
    parameters keep their own roles.
    """

    generator: Callable

    def __init__(self, generator: Callable, /):
        if not callable(generator):
            raise TypeError("generator must be callable.")
        self.generator = generator

    @property
    def certificate(self) -> SymmetrizedFaceClosureCertificate:
        return SymmetrizedFaceClosureCertificate()

    def __call__(
        self,
        system: Any,
        left: Array,
        right: Array,
        baseline_normal_flux: Array,
        context: FaceFluxContext,
        args: Any = None,
        /,
    ) -> Array:
        midpoint = 0.5 * (left + right)
        reversed_context = context.reversed()
        reversed_baseline = -baseline_normal_flux
        forward = jnp.asarray(
            self.generator(system, left, right, baseline_normal_flux, context, args)
        ) - jnp.asarray(
            self.generator(
                system, midpoint, midpoint, baseline_normal_flux, context, args
            )
        )
        backward = jnp.asarray(
            self.generator(system, right, left, reversed_baseline, reversed_context, args)
        ) - jnp.asarray(
            self.generator(
                system, midpoint, midpoint, reversed_baseline, reversed_context, args
            )
        )
        return 0.5 * (forward - backward)


@final
class ArbitraryNormalFaceClosurePlan(AbstractFaceClosurePlan):
    """Arbitrary-normal shared-face correction of a baseline normal flux.

    `correction(system, left, right, baseline_normal_flux, context, args)`
    returns the additive correction of the baseline normal flux density. It is a
    dynamic child: a learned correction holds its model (PARAMETER lane), a
    frozen deployment holds an `ExplicitFreeze` provider, and a plain function
    carries no arrays. The closure identity is static metadata, so weights never
    change the numerical method identity.

    Contract, checked on active faces at every evaluation: the correction has
    the baseline shape and dtype, is finite, and vanishes at equal states within
    `consistency_tolerance`. Orientation antisymmetry
    `C(uL, uR, n) = -C(uR, uL, -n)` is established by construction when the
    correction is a `SymmetrizedFaceClosure`. `differentiability` declares the
    regularity of the correction through its canonical branch-policy contract.
    With `frame="face-normal"` the correction is evaluated in the face-normal
    frame of a system implementing `AbstractNormalFrameSystem`, which makes it
    rotation covariant; the capability is never inferred from a state layout.
    Systems with magnetic components are refused because this closure provides
    no compatible edge electromotive correction.
    """

    correction: Callable
    frame: FaceClosureFrame = eqx.field(static=True)
    consistency_tolerance: float = eqx.field(static=True)
    differentiability: BranchDifferentiationPolicy = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        correction: Callable,
        /,
        *,
        closure_id: str,
        frame: FaceClosureFrame = "global",
        consistency_tolerance: float = 1e-10,
        differentiability: BranchDifferentiationPolicy = (
            BranchDifferentiationPolicy.SMOOTH
        ),
    ):
        if not callable(correction):
            raise TypeError("correction must be callable.")
        if not isinstance(differentiability, BranchDifferentiationPolicy):
            raise TypeError("differentiability must be a BranchDifferentiationPolicy.")
        match differentiability:
            case (
                BranchDifferentiationPolicy.SMOOTH
                | BranchDifferentiationPolicy.BRANCHWISE
                | BranchDifferentiationPolicy.SMOOTH_SURROGATE
            ):
                pass
            case _:
                raise ValueError(
                    "ArbitraryNormalFaceClosurePlan supports SMOOTH, BRANCHWISE, or "
                    f"SMOOTH_SURROGATE; got {differentiability.name}."
                )
        identifier = _identifier(closure_id, "closure_id")
        frame_ = _require_frame(frame)
        tolerance = float(consistency_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("consistency_tolerance must be finite and non-negative.")
        certificates = (
            (correction.certificate,)
            if isinstance(correction, SymmetrizedFaceClosure)
            else ()
        )
        self.correction = correction
        self.frame = frame_
        self.consistency_tolerance = tolerance
        self.differentiability = differentiability
        self.closure_id = canonical_fingerprint(
            {
                "kind": "arbitrary-normal-face-closure",
                "declared_id": identifier,
                "frame": frame_,
                "consistency_tolerance": tolerance,
                "differentiability": differentiability.value,
                "certificates": [
                    [certificate.capability_id, certificate.certificate_id]
                    for certificate in certificates
                ],
            }
        )

    @property
    def certificates(self) -> tuple[AbstractConstructionCertificate, ...]:
        """Construction certificates of the correction (empty when declared only)."""
        if isinstance(self.correction, SymmetrizedFaceClosure):
            return (self.correction.certificate,)
        return ()

    @property
    def derivative_contract(self) -> DerivativeContract:
        return branch_policy_contract(
            self.differentiability,
            surfaces=(DerivativeSurface.PRIMAL_STATE, DerivativeSurface.MODEL_PARAMETER),
        )

    def admit_system(self, system: Any, /) -> None:
        # Imported here: the equations package imports the finite-volume owners.
        from ...equations._hyperbolic_systems import AbstractNormalFrameSystem

        if any(name.startswith("magnetic_") for name in system.component_names):
            raise ValueError(
                "Cell-face closures are unsupported for constrained MHD until they "
                "also provide compatible edge electromotive corrections."
            )
        if self.frame == "face-normal" and not isinstance(
            system, AbstractNormalFrameSystem
        ):
            raise ValueError(
                "A face-normal-frame closure requires a conservation system that "
                "implements AbstractNormalFrameSystem; "
                f"{type(system).__name__} declares no normal-frame transforms."
            )

    def apply(
        self,
        system: Any,
        left: Array,
        right: Array,
        baseline_normal_flux: Array,
        context: FaceFluxContext,
        args: Any = None,
        /,
    ) -> Array:
        self.admit_system(system)
        if not isinstance(context, FaceFluxContext):
            raise TypeError("context must be a FaceFluxContext.")
        if context.frame != "global":
            raise ValueError("Owners supply face closure contexts in the global frame.")
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        baseline = jnp.asarray(baseline_normal_flux)
        if left_.shape != baseline.shape or right_.shape != baseline.shape:
            raise ValueError("Face traces must match the baseline flux shape.")
        if context.unit_normal.shape != baseline.shape[:-1] + (system.dimension,):
            raise ValueError(
                "Face closure context must match the face batch and system dimension."
            )
        if self.frame == "face-normal":
            normal = context.unit_normal
            local = jnp.asarray(
                self.correction(
                    system,
                    system.rotate_state_to_normal_frame(left_, normal),
                    system.rotate_state_to_normal_frame(right_, normal),
                    system.rotate_flux_to_normal_frame(baseline, normal),
                    context.in_normal_frame(),
                    args,
                )
            )
            if local.shape != baseline.shape:
                raise ValueError(
                    "Face closure correction must match baseline flux shape."
                )
            correction = system.rotate_flux_from_normal_frame(local, normal)
        else:
            correction = jnp.asarray(
                self.correction(system, left_, right_, baseline, context, args)
            )
        if correction.shape != baseline.shape:
            raise ValueError("Face closure correction must match baseline flux shape.")
        if correction.dtype != baseline.dtype:
            raise TypeError(
                "Face closure correction dtype must match the baseline flux dtype "
                f"{baseline.dtype}; got {correction.dtype}."
            )
        active = context.active[..., None]
        correction = eqx.error_if(
            correction,
            jnp.any(active & ~jnp.isfinite(correction)),
            "Face closure produced a nonfinite correction.",
        )
        correction = jnp.where(active, correction, jnp.zeros((), correction.dtype))
        equal = context.active & (
            jnp.max(jnp.abs(left_ - right_), axis=-1) <= self.consistency_tolerance
        )
        consistency_defect = jnp.max(
            jnp.where(equal[..., None], jnp.abs(correction), 0.0), initial=0.0
        )
        correction = eqx.error_if(
            correction,
            consistency_defect > self.consistency_tolerance,
            "Face closure violates equal-state consistency.",
        )
        return baseline + correction


__all__ = [
    "AbstractFaceClosurePlan",
    "ArbitraryNormalFaceClosurePlan",
    "FaceClosureFrame",
    "FaceFluxContext",
    "SymmetrizedFaceClosure",
    "SymmetrizedFaceClosureCertificate",
]
