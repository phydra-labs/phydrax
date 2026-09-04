#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntFlag
from math import prod
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._fields import PlaneFieldSpace, ScalarPlaneField, TangentialPlaneField


Padding2D = tuple[tuple[int, int], tuple[int, int]]
PlaneField = ScalarPlaneField | TangentialPlaneField


class AngularSpectrumStatus(IntFlag):
    """Fail-closed status bits for same-grid angular-spectrum propagation."""

    SUCCESS = 0
    LEAKAGE_EXCEEDED = 1
    NONFINITE = 2


def _padding_width(value: object, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("padding widths must be integers.")
    width = int(value)
    if width < 0:
        raise ValueError("padding widths must be nonnegative.")
    return width


def _normalize_padding(
    padding: int | Sequence[Sequence[int]] | None,
    /,
) -> Padding2D | None:
    if padding is None:
        return None
    if isinstance(padding, bool):
        raise TypeError("padding widths must be integers.")
    if isinstance(padding, Integral):
        value = _padding_width(padding)
        return ((value, value), (value, value))
    axes = tuple(tuple(_padding_width(width) for width in pair) for pair in padding)
    if len(axes) != 2 or any(len(pair) != 2 for pair in axes):
        raise ValueError("padding must contain (before, after) widths for two axes.")
    return axes  # type: ignore[return-value]


def _uniform_spacing(space: PlaneFieldSpace, axis_index: int, /) -> float:
    axis = space.grid.axes[axis_index]
    if axis.basis not in ("uniform", "fourier"):
        raise ValueError(
            "Angular-spectrum propagation requires uniform or Fourier grid axes."
        )
    nodes = np.asarray(axis.nodes)
    if nodes.size < 2:
        raise ValueError("Angular-spectrum axes require at least two samples.")
    differences = np.diff(nodes)
    spacing = float(np.mean(differences, dtype=np.float64))
    relative_tolerance = 64.0 * float(np.finfo(nodes.dtype).eps)
    absolute_tolerance = relative_tolerance * max(1.0, abs(spacing))
    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.allclose(
            differences,
            spacing,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    ):
        raise ValueError("Angular-spectrum axes must be finite and uniformly increasing.")
    return spacing


def _field_intensity(values: Array, tangential: bool, /) -> Array:
    density = jnp.real(values * jnp.conj(values))
    return jnp.sum(density, axis=-1) if tangential else density


def _outgoing_longitudinal_wavenumber(
    medium_wavenumber: Array,
    transverse_wavenumber_squared: Array,
    /,
) -> Array:
    candidate = jnp.sqrt(
        medium_wavenumber * medium_wavenumber
        - transverse_wavenumber_squared.astype(medium_wavenumber.dtype)
    )
    reverse = (jnp.imag(candidate) < 0.0) | (
        (jnp.imag(candidate) == 0.0) & (jnp.real(candidate) < 0.0)
    )
    return jnp.where(reverse, -candidate, candidate)


class AngularSpectrumEvidence(StrictModule):
    """Energy and validity evidence from one fixed-shape propagation."""

    input_energy: Array
    working_energy: Array
    retained_energy: Array
    cropped_energy: Array
    leakage_fraction: Array
    finite: Array
    accepted: Array
    status: Array
    evidence_id: str = eqx.field(static=True)


class AngularSpectrumResult(StrictModule):
    """Same-grid propagated field plus fail-closed propagation evidence."""

    field: PlaneField
    evidence: AngularSpectrumEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.accepted

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def leakage_fraction(self) -> Array:
        return self.evidence.leakage_fraction

    @property
    def cropped_energy(self) -> Array:
        return self.evidence.cropped_energy


class AngularSpectrumPlan(StrictModule, NonTrainableState):
    """Static padding and leakage policy for same-grid angular propagation.

    With the ``exp(-i ω t)`` phasor convention, each unshifted Fourier component
    advances by ``exp(+i k_z distance)``. Periodic cells require ``padding=None``
    and therefore perform a no-pad FFT. Finite windows require explicit, positive
    padding on both sides of both axes; execution returns the exact original grid
    crop and reports all energy removed by that crop.
    """

    padding: Padding2D | None = eqx.field(static=True)
    maximum_leakage_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        padding: int | Sequence[Sequence[int]] | None = None,
        /,
        *,
        maximum_leakage_fraction: float = 1.0e-6,
        plan_id: str | None = None,
    ):
        padding_ = _normalize_padding(padding)
        tolerance = float(maximum_leakage_fraction)
        if not np.isfinite(tolerance) or not 0.0 <= tolerance <= 1.0:
            raise ValueError("maximum_leakage_fraction must lie in [0, 1].")
        generated = canonical_fingerprint(
            {
                "kind": "angular-spectrum-plan",
                "padding": padding_,
                "maximum_leakage_fraction": tolerance.hex(),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.padding = padding_
        self.maximum_leakage_fraction = tolerance
        self.plan_id = identifier

    def prepare(self, space: PlaneFieldSpace, /) -> PreparedAngularSpectrum:
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        if space.topology == "periodic-cell":
            if self.padding is not None:
                raise ValueError("Periodic-cell propagation requires padding=None.")
            padding: Padding2D = ((0, 0), (0, 0))
        else:
            if self.padding is None or any(
                width <= 0 for pair in self.padding for width in pair
            ):
                raise ValueError(
                    "Finite-window propagation requires explicit positive padding "
                    "before and after both axes."
                )
            padding = self.padding
        spacings = tuple(_uniform_spacing(space, index) for index in range(2))
        working_shape = tuple(
            size + before + after
            for size, (before, after) in zip(space.shape, padding, strict=True)
        )
        frequencies = tuple(
            2.0 * jnp.pi * jnp.fft.fftfreq(size, d=spacing)
            for size, spacing in zip(working_shape, spacings, strict=True)
        )
        transverse_wavenumber_squared = (
            frequencies[0][:, None] ** 2 + frequencies[1][None, :] ** 2
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-angular-spectrum",
                "plan": self.plan_id,
                "space": space.space_id,
                "padding": padding,
                "working_shape": working_shape,
                "sample_spacings": [spacing.hex() for spacing in spacings],
            }
        )
        return PreparedAngularSpectrum(
            plan=self,
            space=space,
            padding=padding,
            sample_spacings=spacings,
            working_shape=working_shape,  # type: ignore[arg-type]
            transverse_wavenumber_squared=transverse_wavenumber_squared,
            prepared_id=prepared_id,
        )


class PreparedAngularSpectrum(StrictModule, NonTrainableState):
    """Prepared fixed-shape FFT geometry for repeated dynamic propagation."""

    plan: AngularSpectrumPlan
    space: PlaneFieldSpace
    transverse_wavenumber_squared: Array
    padding: Padding2D = eqx.field(static=True)
    sample_spacings: tuple[float, float] = eqx.field(static=True)
    working_shape: tuple[int, int] = eqx.field(static=True)
    workspace_complex_elements_per_component: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan: AngularSpectrumPlan,
        space: PlaneFieldSpace,
        padding: Padding2D,
        sample_spacings: tuple[float, float],
        working_shape: tuple[int, int],
        transverse_wavenumber_squared: Array,
        prepared_id: str,
    ):
        if transverse_wavenumber_squared.shape != working_shape:
            raise ValueError("Prepared transverse spectrum has the wrong fixed shape.")
        if not prepared_id:
            raise ValueError("prepared_id must be non-empty.")
        self.plan = plan
        self.space = space
        self.transverse_wavenumber_squared = transverse_wavenumber_squared
        self.padding = padding
        self.sample_spacings = sample_spacings
        self.working_shape = working_shape
        self.workspace_complex_elements_per_component = 3 * prod(working_shape)
        self.prepared_id = prepared_id

    def execute(
        self,
        field: PlaneField,
        distance: ArrayLike,
        medium_wavenumber: ArrayLike,
        /,
    ) -> AngularSpectrumResult:
        """Propagate by a dynamic nonnegative distance in an explicit medium."""
        if not isinstance(field, (ScalarPlaneField, TangentialPlaneField)):
            raise TypeError("field must be a ScalarPlaneField or TangentialPlaneField.")
        if field.space.space_id != self.space.space_id:
            raise ValueError("field does not belong to the prepared plane space.")
        supplied_distance = jnp.asarray(distance)
        if supplied_distance.shape != ():
            raise ValueError("distance must be a scalar.")
        if (
            jnp.iscomplexobj(supplied_distance)
            or not jnp.issubdtype(supplied_distance.dtype, jnp.number)
            or jnp.issubdtype(supplied_distance.dtype, jnp.bool_)
        ):
            raise TypeError("distance must be real numeric data.")
        distance_ = supplied_distance.astype(
            jnp.result_type(supplied_distance.dtype, jnp.float32)
        )
        distance_ = eqx.error_if(
            distance_,
            (~jnp.isfinite(distance_)) | (distance_ < 0.0),
            "distance must be finite and nonnegative.",
        )
        supplied_wavenumber = jnp.asarray(medium_wavenumber)
        if supplied_wavenumber.shape != ():
            raise ValueError("medium_wavenumber must be a scalar.")
        if not jnp.issubdtype(supplied_wavenumber.dtype, jnp.number) or jnp.issubdtype(
            supplied_wavenumber.dtype, jnp.bool_
        ):
            raise TypeError("medium_wavenumber must be numeric.")
        complex_dtype = jnp.result_type(
            field.values.dtype, supplied_wavenumber.dtype, jnp.complex64
        )
        wavenumber = supplied_wavenumber.astype(complex_dtype)
        wavenumber = eqx.error_if(
            wavenumber,
            (~jnp.isfinite(jnp.real(wavenumber)))
            | (~jnp.isfinite(jnp.imag(wavenumber)))
            | (jnp.real(wavenumber) <= 0.0)
            | (jnp.imag(wavenumber) < 0.0),
            "medium_wavenumber must be finite with positive real part and "
            "nonnegative imaginary part.",
        )

        tangential = isinstance(field, TangentialPlaneField)
        pad_width = self.padding + (((0, 0),) if tangential else ())
        working_values = jnp.pad(field.values.astype(complex_dtype), pad_width)
        longitudinal_wavenumber = _outgoing_longitudinal_wavenumber(
            wavenumber,
            self.transverse_wavenumber_squared,
        )
        transfer = jnp.exp(1j * longitudinal_wavenumber * distance_)
        spectrum = jnp.fft.fftn(working_values, axes=(0, 1))
        multiplier = transfer[..., None] if tangential else transfer
        propagated = jnp.fft.ifftn(spectrum * multiplier, axes=(0, 1))

        (axis0_before, axis0_after), (axis1_before, axis1_after) = self.padding
        axis0_stop = self.working_shape[0] - axis0_after
        axis1_stop = self.working_shape[1] - axis1_after
        cropped = propagated[
            axis0_before:axis0_stop,
            axis1_before:axis1_stop,
            ...,
        ]
        if cropped.shape != field.values.shape:
            raise RuntimeError("Prepared crop did not recover the input field shape.")

        sample_area = self.sample_spacings[0] * self.sample_spacings[1]
        input_energy = sample_area * jnp.sum(_field_intensity(field.values, tangential))
        working_energy = sample_area * jnp.sum(_field_intensity(propagated, tangential))
        retained_energy = sample_area * jnp.sum(_field_intensity(cropped, tangential))
        cropped_energy = jnp.maximum(working_energy - retained_energy, 0.0)
        leakage_fraction = jnp.where(
            working_energy > 0.0,
            cropped_energy / working_energy,
            jnp.asarray(0.0, dtype=working_energy.dtype),
        )
        finite = (
            jnp.all(jnp.isfinite(jnp.real(cropped)))
            & jnp.all(jnp.isfinite(jnp.imag(cropped)))
            & jnp.isfinite(input_energy)
            & jnp.isfinite(working_energy)
            & jnp.isfinite(retained_energy)
            & jnp.isfinite(leakage_fraction)
        )
        leakage_exceeded = leakage_fraction > self.plan.maximum_leakage_fraction
        status = jnp.where(
            leakage_exceeded,
            int(AngularSpectrumStatus.LEAKAGE_EXCEEDED),
            int(AngularSpectrumStatus.SUCCESS),
        ).astype(jnp.int32) | jnp.where(
            finite,
            int(AngularSpectrumStatus.SUCCESS),
            int(AngularSpectrumStatus.NONFINITE),
        ).astype(jnp.int32)
        accepted = status == int(AngularSpectrumStatus.SUCCESS)
        output_coordinate = field.longitudinal_coordinate + distance_
        if tangential:
            output: PlaneField = TangentialPlaneField(
                self.space,
                cropped,
                field.angular_frequency,
                output_coordinate,
            )
        else:
            output = ScalarPlaneField(
                self.space,
                cropped,
                field.angular_frequency,
                output_coordinate,
            )
        evidence = AngularSpectrumEvidence(
            input_energy,
            working_energy,
            retained_energy,
            cropped_energy,
            leakage_fraction,
            finite,
            accepted,
            status,
            self.prepared_id,
        )
        return AngularSpectrumResult(output, evidence, self.prepared_id)


def propagate_angular_spectrum(
    prepared: PreparedAngularSpectrum,
    field: PlaneField,
    distance: ArrayLike,
    medium_wavenumber: ArrayLike,
    /,
) -> AngularSpectrumResult:
    """Execute a prepared same-grid angular-spectrum propagation."""
    if not isinstance(prepared, PreparedAngularSpectrum):
        raise TypeError("prepared must be a PreparedAngularSpectrum.")
    return prepared.execute(field, distance, medium_wavenumber)


__all__ = [
    "AngularSpectrumEvidence",
    "AngularSpectrumPlan",
    "AngularSpectrumResult",
    "AngularSpectrumStatus",
    "PreparedAngularSpectrum",
    "propagate_angular_spectrum",
]
