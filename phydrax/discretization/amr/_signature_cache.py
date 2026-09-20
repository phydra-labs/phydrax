#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-generated finite patch signatures and atomic JAX executable caching."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._canonical import BlockAMRResourcePlan, CanonicalPatchHierarchy
from ._patches import PatchShapeSignature


KernelFactory = Callable[["PatchExecutableSignature"], Callable[..., Any]]
SampleFactory = Callable[
    ["PatchExecutableSignature"], tuple[tuple[Any, ...], dict[str, Any]]
]


class PatchExecutableSignature(StrictModule, NonTrainableState):
    """Complete static shape and numerical identity for one patch executable."""

    patch_shape: PatchShapeSignature
    maximum_components_per_cell: int = eqx.field(static=True)
    physical_component_count: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_shape: PatchShapeSignature,
        /,
        *,
        maximum_components_per_cell: int,
        physical_component_count: int,
        method_id: str,
        dtype: str | np.dtype,
        backend: str,
    ):
        cut_components = int(maximum_components_per_cell)
        components = int(physical_component_count)
        method = str(method_id)
        dtype_ = np.dtype(dtype).name
        backend_ = str(backend)
        if (
            not isinstance(patch_shape, PatchShapeSignature)
            or cut_components <= 0
            or components <= 0
            or not method
            or not backend_
        ):
            raise ValueError("Patch executable signature fields must be complete.")
        self.patch_shape = patch_shape
        self.maximum_components_per_cell = cut_components
        self.physical_component_count = components
        self.method_id = method
        self.dtype = dtype_
        self.backend = backend_
        self.signature_id = canonical_fingerprint(
            {
                "kind": "block-amr-patch-executable-signature",
                "patch_shape": patch_shape.signature_id,
                "maximum_components_per_cell": cut_components,
                "physical_component_count": components,
                "method": method,
                "dtype": dtype_,
                "backend": backend_,
            }
        )

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (
            *self.patch_shape.envelope_shape,
            self.maximum_components_per_cell,
            self.physical_component_count,
        )


class PatchSignaturePolicy(StrictModule, NonTrainableState):
    """Generate aligned finite envelopes without a handwritten shape catalog."""

    alignment: tuple[int, ...] = eqx.field(static=True)
    halo_width: tuple[int, ...] = eqx.field(static=True)
    maximum_envelope: tuple[int, ...] = eqx.field(static=True)
    power_of_two_growth: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        alignment: Sequence[int],
        maximum_envelope: Sequence[int],
        /,
        *,
        halo_width: int | Sequence[int] = 1,
        power_of_two_growth: bool = True,
    ):
        alignment_ = tuple(alignment)
        maximum = tuple(maximum_envelope)
        halo = (
            (int(halo_width),) * len(alignment_)
            if isinstance(halo_width, int)
            else tuple(halo_width)
        )
        if (
            not alignment_
            or len(maximum) != len(alignment_)
            or len(halo) != len(alignment_)
            or any(value <= 0 for value in alignment_)
            or any(value <= 0 for value in maximum)
            or any(value < 0 for value in halo)
            or any(
                maximum_value % alignment_value
                for maximum_value, alignment_value in zip(
                    maximum, alignment_, strict=True
                )
            )
        ):
            raise ValueError("Patch signature policy alignment/envelope is invalid.")
        self.alignment = alignment_
        self.halo_width = halo
        self.maximum_envelope = maximum
        self.power_of_two_growth = bool(power_of_two_growth)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "patch-signature-policy",
                "alignment": alignment_,
                "halo": halo,
                "maximum": maximum,
                "power_of_two_growth": bool(power_of_two_growth),
            }
        )

    def shape_signature(self, extent: Sequence[int], /) -> PatchShapeSignature:
        values = tuple(extent)
        if len(values) != len(self.alignment) or any(value <= 0 for value in values):
            raise ValueError("Runtime patch extent has incompatible rank or size.")
        envelope = []
        for value, alignment, maximum in zip(
            values, self.alignment, self.maximum_envelope, strict=True
        ):
            aligned = (value + alignment - 1) // alignment * alignment
            if self.power_of_two_growth:
                grown = alignment
                while grown < aligned:
                    grown *= 2
                aligned = grown
            if aligned > maximum:
                raise ValueError(
                    "Runtime patch extent exceeds the declared executable envelope."
                )
            envelope.append(aligned)
        return PatchShapeSignature(
            tuple(envelope),
            halo_width=self.halo_width,
            alignment=self.alignment,
        )


class PreparedPatchExecutable(StrictModule, NonTrainableState):
    """One compiled JAX callable bound to an exact static signature."""

    signature: PatchExecutableSignature
    executable: Any = eqx.field(static=True)
    executable_id: str = eqx.field(static=True)

    def __init__(
        self,
        signature: PatchExecutableSignature,
        executable: Any,
        /,
    ):
        if not isinstance(signature, PatchExecutableSignature) or not callable(
            executable
        ):
            raise TypeError("Prepared patch executable requires signature and callable.")
        self.signature = signature
        self.executable = executable
        self.executable_id = canonical_fingerprint(
            {
                "kind": "prepared-patch-executable",
                "signature": signature.signature_id,
            }
        )

    def __call__(self, *args, **kwargs):
        return self.executable(*args, **kwargs)


class PatchExecutableCacheState(StrictModule, NonTrainableState):
    """Immutable installed executable set for one accepted runtime boundary."""

    executables: tuple[PreparedPatchExecutable, ...]
    generation: int = eqx.field(static=True)
    cache_id: str = eqx.field(static=True)

    def __init__(
        self,
        executables: Sequence[PreparedPatchExecutable] = (),
        /,
        *,
        generation: int = 0,
    ):
        values = tuple(
            sorted(executables, key=lambda value: value.signature.signature_id)
        )
        if (
            any(not isinstance(value, PreparedPatchExecutable) for value in values)
            or len({value.signature.signature_id for value in values}) != len(values)
            or int(generation) < 0
        ):
            raise ValueError("Patch executable cache entries or generation are invalid.")
        self.executables = values
        self.generation = int(generation)
        self.cache_id = canonical_fingerprint(
            {
                "kind": "patch-executable-cache-state",
                "executables": [value.executable_id for value in values],
                "generation": int(generation),
            }
        )

    def lookup(
        self, signature: PatchExecutableSignature, /
    ) -> PreparedPatchExecutable | None:
        for executable in self.executables:
            if executable.signature.signature_id == signature.signature_id:
                return executable
        return None


class PatchExecutableInstallResult(StrictModule, NonTrainableState):
    """Atomic cache successor and exact installed/missing identities."""

    state: PatchExecutableCacheState
    installed_signature_ids: tuple[str, ...] = eqx.field(static=True)
    reused_signature_ids: tuple[str, ...] = eqx.field(static=True)
    changed: bool = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: PatchExecutableCacheState,
        installed_signature_ids: Sequence[str],
        reused_signature_ids: Sequence[str],
        /,
    ):
        installed = tuple(installed_signature_ids)
        reused = tuple(reused_signature_ids)
        self.state = state
        self.installed_signature_ids = installed
        self.reused_signature_ids = reused
        self.changed = bool(installed)
        self.result_id = canonical_fingerprint(
            {
                "kind": "patch-executable-install-result",
                "state": state.cache_id,
                "installed": installed,
                "reused": reused,
            }
        )


class PatchExecutableCachePlan(StrictModule, NonTrainableState):
    """Compile missing signatures first, then atomically publish one cache state."""

    resources: BlockAMRResourcePlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, resources: BlockAMRResourcePlan, /):
        if not isinstance(resources, BlockAMRResourcePlan):
            raise TypeError("Patch executable cache requires BlockAMRResourcePlan.")
        self.resources = resources
        self.plan_id = canonical_fingerprint(
            {
                "kind": "patch-executable-cache-plan",
                "resources": resources.resource_id,
            }
        )

    def required_signatures(
        self,
        hierarchy: CanonicalPatchHierarchy,
        /,
        *,
        physical_component_count: int,
        method_id: str,
        dtype: str | np.dtype,
        backend: str | None = None,
    ) -> tuple[PatchExecutableSignature, ...]:
        if not isinstance(hierarchy, CanonicalPatchHierarchy):
            raise TypeError("Executable signatures require CanonicalPatchHierarchy.")
        backend_ = jax.default_backend() if backend is None else str(backend)
        values = {
            bucket.plan.signature.signature_id: PatchExecutableSignature(
                bucket.plan.signature,
                maximum_components_per_cell=self.resources.maximum_components_per_cell,
                physical_component_count=int(physical_component_count),
                method_id=method_id,
                dtype=dtype,
                backend=backend_,
            )
            for level in hierarchy.levels
            for bucket in level.buckets
        }
        return tuple(sorted(values.values(), key=lambda value: value.signature_id))

    def install(
        self,
        state: PatchExecutableCacheState,
        signatures: Sequence[PatchExecutableSignature],
        kernel_factory: KernelFactory,
        sample_factory: SampleFactory,
        /,
    ) -> PatchExecutableInstallResult:
        if not isinstance(state, PatchExecutableCacheState):
            raise TypeError("Executable installation requires cache state.")
        if not callable(kernel_factory) or not callable(sample_factory):
            raise TypeError("Executable installation requires kernel/sample factories.")
        requested = tuple(sorted(signatures, key=lambda value: value.signature_id))
        if any(
            not isinstance(value, PatchExecutableSignature) for value in requested
        ) or len({value.signature_id for value in requested}) != len(requested):
            raise ValueError("Requested executable signatures must be unique and valid.")
        prepared = list(state.executables)
        installed = []
        reused = []
        for signature in requested:
            existing = state.lookup(signature)
            if existing is not None:
                reused.append(signature.signature_id)
                continue
            kernel = kernel_factory(signature)
            if not callable(kernel):
                raise TypeError("Patch kernel factory must return a callable.")
            positional, keyword = sample_factory(signature)
            lowered = jax.jit(kernel).lower(*positional, **keyword)
            compiled = lowered.compile()
            prepared.append(PreparedPatchExecutable(signature, compiled))
            installed.append(signature.signature_id)
        successor = (
            state
            if not installed
            else PatchExecutableCacheState(
                prepared,
                generation=state.generation + 1,
            )
        )
        return PatchExecutableInstallResult(successor, installed, reused)


__all__ = [
    "KernelFactory",
    "PatchExecutableCachePlan",
    "PatchExecutableCacheState",
    "PatchExecutableInstallResult",
    "PatchExecutableSignature",
    "PatchSignaturePolicy",
    "PreparedPatchExecutable",
    "SampleFactory",
]
