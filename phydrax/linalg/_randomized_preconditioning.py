#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, PyTree

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._costs import PreconditionerCostEstimate
from ._hermitian_spectral import HermitianSpectrum
from ._materialization import MaterializationPolicy
from ._operators import AbstractLinearOperator
from ._pairings import EuclideanPairing
from ._preconditioner_properties import PreconditionerProperties
from ._preconditioners import AbstractPreconditioner
from ._preconditioning import AbstractPreconditionerBuilder
from ._spaces import _coordinate_dtype, ArraySpace, PyTreeSpace


ProbeRefresh: TypeAlias = Literal["reuse", "redraw"]


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _euclidean_space(operator: AbstractLinearOperator, /) -> ArraySpace | PyTreeSpace:
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Randomized Nyström setup requires an unbatched endomorphism.")
    space = operator.source
    if not isinstance(space, (ArraySpace, PyTreeSpace)) or not isinstance(
        space.pairing, EuclideanPairing
    ):
        raise TypeError(
            "Randomized Nyström preconditioning currently requires an ArraySpace or "
            "PyTreeSpace with EuclideanPairing."
        )
    if not operator.properties.certifies("self_adjoint"):
        raise ValueError("Randomized Nyström setup requires certified self-adjointness.")
    if not operator.properties.certifies("positive_semidefinite"):
        raise ValueError(
            "Randomized Nyström setup requires certified positive semidefiniteness."
        )
    return space


def _random_probes(
    key: Array,
    dimension: int,
    count: int,
    dtype: jnp.dtype,
    /,
) -> Array:
    real_dtype = jnp.empty((), dtype=dtype).real.dtype
    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_key, imag_key = jr.split(key)
        probes = (
            jr.normal(real_key, (dimension, count), dtype=real_dtype)
            + 1j * jr.normal(imag_key, (dimension, count), dtype=real_dtype)
        ) / jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype))
        return probes.astype(dtype)
    return jr.normal(key, (dimension, count), dtype=dtype)


def _operator_columns(
    operator: AbstractLinearOperator,
    coordinates: Array,
    /,
) -> Array:
    space = operator.source

    def apply(column):
        value = operator.mv(space.unflatten(column))
        return space.flatten(value)

    return jax.vmap(apply, in_axes=1, out_axes=1)(coordinates)


class RandomizedNystromDiagnostics(StrictModule):
    """Measured preparation evidence for one randomized Nyström action."""

    ritz_values: Array
    effective_rank: Array
    core_minimum_eigenvalue: Array
    core_condition_number: Array
    captured_sketch_energy_fraction: Array
    stabilization: Array
    valid: Array
    requested_rank: int = eqx.field(static=True)
    sketch_size: int = eqx.field(static=True)
    setup_matvec_count: int = eqx.field(static=True)
    refresh_count: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        ritz_values: Array,
        effective_rank: Array,
        core_minimum_eigenvalue: Array,
        core_condition_number: Array,
        captured_sketch_energy_fraction: Array,
        stabilization: Array,
        valid: Array,
        requested_rank: int,
        sketch_size: int,
        refresh_count: int,
    ):
        self.ritz_values = jnp.asarray(ritz_values)
        self.effective_rank = jnp.asarray(effective_rank, dtype=jnp.int32)
        self.core_minimum_eigenvalue = jnp.asarray(core_minimum_eigenvalue)
        self.core_condition_number = jnp.asarray(core_condition_number)
        self.captured_sketch_energy_fraction = jnp.asarray(
            captured_sketch_energy_fraction
        )
        self.stabilization = jnp.asarray(stabilization)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.requested_rank = int(requested_rank)
        self.sketch_size = int(sketch_size)
        self.setup_matvec_count = int(sketch_size)
        self.refresh_count = int(refresh_count)


class RandomizedNystromPreconditioner(AbstractPreconditioner):
    """Shifted inverse of a fixed-rank randomized positive-operator model."""

    basis: Array
    ritz_values: Array
    shift: Array
    diagnostics: RandomizedNystromDiagnostics

    def __init__(
        self,
        basis: Array,
        ritz_values: Array,
        shift: Array,
        diagnostics: RandomizedNystromDiagnostics,
        /,
        *,
        space: ArraySpace | PyTreeSpace,
        preconditioner_id: str,
    ):
        basis_ = jnp.asarray(basis)
        values = jnp.asarray(ritz_values)
        shift_ = jnp.asarray(shift, dtype=values.real.dtype).reshape(())
        if basis_.ndim != 2 or basis_.shape[0] != space.size:
            raise ValueError(
                "Randomized Nyström basis must have shape (space.size, rank)."
            )
        if values.shape != (basis_.shape[1],):
            raise ValueError("Randomized Nyström Ritz values must match basis rank.")
        if not isinstance(diagnostics, RandomizedNystromDiagnostics):
            raise TypeError("diagnostics must be RandomizedNystromDiagnostics.")
        self.space = space
        self.basis = jax.lax.stop_gradient(basis_)
        self.ritz_values = jax.lax.stop_gradient(values)
        self.shift = jax.lax.stop_gradient(shift_)
        self.diagnostics = diagnostics
        self.properties = PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "linear": "construction",
                "stationary": "construction",
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        )
        identifier = str(preconditioner_id)
        if not identifier:
            raise ValueError("preconditioner_id must be non-empty.")
        self.preconditioner_id = identifier

    def apply(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        del iteration
        coordinates = self.space.flatten(residual)
        coefficients = ein.contract("nr,n->r", jnp.conj(self.basis), coordinates)
        attenuation = self.ritz_values / (self.ritz_values + self.shift)
        correction = ein.contract("nr,r,r->n", self.basis, attenuation, coefficients)
        return self.space.unflatten((coordinates - correction) / self.shift)

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        del materialization
        space = _euclidean_space(setup_operator)
        if not space.compatible(self.space):
            raise ValueError("Prepared Nyström action and setup operator spaces differ.")
        itemsize = _coordinate_dtype(space).itemsize
        dimension, rank = self.basis.shape
        return PreconditionerCostEstimate(
            component=self.preconditioner_id,
            storage_bytes=(dimension * rank + rank) * itemsize,
            apply_workspace_bytes_per_rhs=(2 * dimension + 2 * rank) * itemsize,
            accepted=True,
            reason="prepared shifted randomized Nyström inverse action",
        )


class RandomizedNystromPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare a fixed-rank shifted inverse from matrix-free positive actions."""

    rank: int = eqx.field(static=True)
    oversampling: int = eqx.field(static=True)
    shift: float = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    probe_refresh: ProbeRefresh = eqx.field(static=True)
    stabilization: float | None = eqx.field(static=True)
    psd_tolerance: float = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        rank: int,
        /,
        *,
        oversampling: int = 8,
        shift: float = 1e-3,
        seed: int = 0,
        probe_refresh: ProbeRefresh = "reuse",
        stabilization: float | None = None,
        psd_tolerance: float = 1e-10,
    ):
        rank_ = int(rank)
        oversampling_ = int(oversampling)
        shift_ = float(shift)
        seed_ = int(seed)
        tolerance = float(psd_tolerance)
        if rank_ < 1:
            raise ValueError("rank must be positive.")
        if oversampling_ < 0:
            raise ValueError("oversampling must be non-negative.")
        if not isfinite(shift_) or shift_ <= 0.0:
            raise ValueError("shift must be finite and strictly positive.")
        if probe_refresh not in ("reuse", "redraw"):
            raise ValueError("probe_refresh must be 'reuse' or 'redraw'.")
        if stabilization is not None and (
            not isfinite(float(stabilization)) or float(stabilization) <= 0.0
        ):
            raise ValueError("stabilization must be finite and positive or None.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("psd_tolerance must be finite and non-negative.")
        self.rank = rank_
        self.oversampling = oversampling_
        self.shift = shift_
        self.seed = seed_
        self.probe_refresh = probe_refresh
        self.stabilization = None if stabilization is None else float(stabilization)
        self.psd_tolerance = tolerance
        self._builder_id = canonical_fingerprint(
            {
                "kind": "randomized-nystrom-preconditioner-builder",
                "rank": rank_,
                "oversampling": oversampling_,
                "shift": shift_,
                "seed": seed_,
                "probe_refresh": probe_refresh,
                "stabilization": self.stabilization,
                "psd_tolerance": tolerance,
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> Literal["numeric"]:
        return "numeric"

    def _validated_space(
        self, setup_operator: AbstractLinearOperator, /
    ) -> ArraySpace | PyTreeSpace:
        space = _euclidean_space(setup_operator)
        if self.rank + self.oversampling > space.size:
            raise ValueError(
                "rank + oversampling may not exceed the setup space dimension."
            )
        return space

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        self._validated_space(setup_operator)
        return PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "linear": "construction",
                "stationary": "construction",
                "self_adjoint": "transformed",
                "positive_definite": "construction",
            },
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        del materialization
        space = self._validated_space(setup_operator)
        itemsize = _coordinate_dtype(space).itemsize
        dimension = space.size
        sketch = self.rank + self.oversampling
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=(dimension * self.rank + self.rank) * itemsize,
            preparation_workspace_bytes=(2 * dimension * sketch + 3 * sketch * sketch)
            * itemsize,
            apply_workspace_bytes_per_rhs=(2 * dimension + 2 * self.rank) * itemsize,
            setup_matvec_count=sketch,
            accepted=True,
            reason="matrix-free randomized Nyström sketch and shifted low-rank inverse",
        )

    def _prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        refresh_count: int,
    ) -> RandomizedNystromPreconditioner:
        space = self._validated_space(setup_operator)
        dtype = _coordinate_dtype(space)
        sketch_size = self.rank + self.oversampling
        address = refresh_count if self.probe_refresh == "redraw" else 0
        key = jr.fold_in(jr.key(self.seed), address)
        probes = _random_probes(key, space.size, sketch_size, dtype)
        probes, _ = jnp.linalg.qr(probes, mode="reduced")
        images = _operator_columns(setup_operator, probes)
        core = 0.5 * (
            ein.contract("nk,nl->kl", jnp.conj(probes), images)
            + ein.contract("nk,nl->kl", jnp.conj(images), probes)
        )
        core_spectrum = HermitianSpectrum(core, tolerance=self.psd_tolerance)
        core_scale = jnp.maximum(jnp.max(jnp.abs(core_spectrum.eigenvalues)), 1.0)
        negative_limit = self.psd_tolerance * core_scale
        images = eqx.error_if(
            images,
            (~core_spectrum.valid)
            | (core_spectrum.minimum_eigenvalue < -negative_limit)
            | jnp.any(~jnp.isfinite(images)),
            "Randomized Nyström setup produced a nonfinite or materially indefinite core.",
        )
        real_dtype = jnp.empty((), dtype=dtype).real.dtype
        automatic_stabilization = jnp.finfo(real_dtype).eps * jnp.maximum(
            jnp.sqrt(jnp.sum(jnp.real(jnp.conj(images) * images))),
            jnp.asarray(1.0, dtype=real_dtype),
        )
        stabilization = (
            automatic_stabilization
            if self.stabilization is None
            else jnp.asarray(self.stabilization, dtype=real_dtype)
        )
        core_values = jnp.maximum(core_spectrum.eigenvalues, 0.0) + stabilization
        inverse_root = jnp.reciprocal(jnp.sqrt(core_values))
        factor = ein.contract(
            "nk,kl,l->nl", images, core_spectrum.eigenvectors, inverse_root
        )
        reduced = ein.contract("nk,nl->kl", jnp.conj(factor), factor)
        reduced_spectrum = HermitianSpectrum(reduced, tolerance=self.psd_tolerance)
        descending = jnp.arange(sketch_size - 1, -1, -1)
        selected = descending[: self.rank]
        sigma_square = jnp.maximum(reduced_spectrum.eigenvalues[selected], 0.0)
        selected_vectors = reduced_spectrum.eigenvectors[:, selected]
        spectral_floor = self.psd_tolerance * jnp.maximum(
            jnp.max(sigma_square), jnp.asarray(1.0, dtype=real_dtype)
        )
        safe_sigma = jnp.maximum(sigma_square, spectral_floor)
        basis = ein.contract(
            "nk,kr,r->nr", factor, selected_vectors, jnp.reciprocal(jnp.sqrt(safe_sigma))
        )
        ritz_values = jnp.maximum(sigma_square - stabilization, 0.0)
        effective_rank = jnp.sum(ritz_values > spectral_floor, dtype=jnp.int32)
        total_energy = jnp.sum(jnp.maximum(reduced_spectrum.eigenvalues, 0.0))
        captured = jnp.sum(ritz_values) / jnp.maximum(total_energy, spectral_floor)
        finite = (
            reduced_spectrum.valid
            & jnp.all(jnp.isfinite(basis))
            & jnp.all(jnp.isfinite(ritz_values))
            & jnp.isfinite(captured)
        )
        basis = eqx.error_if(
            basis,
            ~finite,
            "Randomized Nyström factorization produced nonfinite retained factors.",
        )
        basis = jax.lax.stop_gradient(basis)
        ritz_values = jax.lax.stop_gradient(ritz_values)
        diagnostics = RandomizedNystromDiagnostics(
            ritz_values=ritz_values,
            effective_rank=effective_rank,
            core_minimum_eigenvalue=core_spectrum.minimum_eigenvalue,
            core_condition_number=core_spectrum.condition_number,
            captured_sketch_energy_fraction=captured,
            stabilization=stabilization,
            valid=finite,
            requested_rank=self.rank,
            sketch_size=sketch_size,
            refresh_count=refresh_count,
        )
        return RandomizedNystromPreconditioner(
            basis,
            ritz_values,
            jnp.asarray(self.shift, dtype=real_dtype),
            diagnostics,
            space=space,
            preconditioner_id=canonical_fingerprint(
                {
                    "kind": "prepared-randomized-nystrom",
                    "builder": self.builder_id,
                    "setup_operator": setup_operator.operator_id,
                    "refresh_count": refresh_count,
                }
            ),
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> RandomizedNystromPreconditioner:
        if not isinstance(materialization, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        return self._prepare(setup_operator, refresh_count=0)

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> RandomizedNystromPreconditioner:
        if not isinstance(preconditioner, RandomizedNystromPreconditioner):
            raise TypeError(
                "Randomized Nyström refresh requires a RandomizedNystromPreconditioner."
            )
        if not isinstance(materialization, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        return self._prepare(
            setup_operator,
            refresh_count=preconditioner.diagnostics.refresh_count + 1,
        )


__all__ = [
    "ProbeRefresh",
    "RandomizedNystromDiagnostics",
    "RandomizedNystromPreconditioner",
    "RandomizedNystromPreconditionerBuilder",
]
