#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reduced 1-D/2-D (dD3V) compatible Maxwell as a PIC field solver."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._dtype_names import RealPrecisionDType
from .._fingerprint import canonical_fingerprint
from .._trainable import NonTrainableState
from ..discretization import AxisEntityKind
from ..discretization.pic import PICSpeciesPlan, ReducedPICTransferPlan
from ._maxwell_reduced import (
    _charge_divergence,
    CompatibleMaxwell1DPlan,
    CompatibleMaxwell1DState,
    CompatibleMaxwell2DPlan,
    CompatibleMaxwell2DState,
)
from ._pic_field_solver import (
    AbstractPreparedPICFieldSolver,
    PICFieldAdvance,
    PICFieldDeposit,
    PICGaussProjectionResult,
    PICRestartComponent,
    PICTensorKind,
    PICTensorMap,
    restart_component,
    restore_component,
)


type ReducedMaxwellPlan = CompatibleMaxwell1DPlan | CompatibleMaxwell2DPlan
type ReducedMaxwellState = CompatibleMaxwell1DState | CompatibleMaxwell2DState
type _Triple = tuple[Array, Array, Array]


def _shift_without_wrap(value: Array, axis: int, cells: int, /) -> Array:
    shifted = jnp.roll(value, -cells, axis=axis)
    index: list[slice] = [slice(None)] * value.ndim
    index[axis] = slice(value.shape[axis] - cells, value.shape[axis])
    return shifted.at[tuple(index)].set(0.0)


class ReducedMaxwellPICFieldSolver(AbstractPreparedPICFieldSolver, NonTrainableState):
    """dD3V Yee field with CIC gather and continuity-projected current.

    Species-agnostic: every species shares the grid transfer, so deposits of
    all species fuse into one pass (`PICMultiDeposit`). Positions carry the
    ``d`` resolved coordinates; the remaining directions are invariant.
    """

    field: ReducedMaxwellPlan
    transfer: ReducedPICTransferPlan
    solver_id: str = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    field_dtype: RealPrecisionDType = eqx.field(static=True)

    def __init__(
        self, field: ReducedMaxwellPlan, transfer: ReducedPICTransferPlan, /
    ) -> None:
        if not isinstance(field, (CompatibleMaxwell1DPlan, CompatibleMaxwell2DPlan)):
            raise TypeError("field must be a compatible reduced Maxwell plan.")
        if not isinstance(transfer, ReducedPICTransferPlan):
            raise TypeError("transfer must be ReducedPICTransferPlan.")
        if transfer.grid.prepared_id != field.grid.prepared_id:
            raise ValueError("Reduced PIC field and transfer grids differ.")
        self.field = field
        self.transfer = transfer
        self.spatial_dimension = transfer.dimension
        self.field_dtype = "float64"
        self.solver_id = canonical_fingerprint(
            {
                "kind": "reduced-maxwell-pic-field-solver",
                "field": field.plan_id,
                "transfer": transfer.plan_id,
            }
        )

    @property
    def stable_step(self) -> Array:
        return jnp.asarray(self.field.stable_dt)

    @property
    def displacement_widths(self) -> Array:
        return jnp.asarray(self.transfer.spacing)

    def validate_species(self, species: tuple[PICSpeciesPlan, ...], /) -> None:
        if any(
            value.population.particles.ambient_dimension != self.spatial_dimension
            for value in species
        ):
            raise ValueError("Reduced PIC species must match the field dimension.")

    def pairing_probe(self, species: int, capacity: int, /) -> tuple[Array, Array]:
        del species
        slot = np.arange(capacity)
        start = np.stack(
            tuple(
                lower + (slot % count + 0.7) * spacing
                for lower, count, spacing in zip(
                    self.transfer.lower,
                    self.transfer.shape,
                    self.transfer.spacing,
                    strict=True,
                )
            ),
            axis=-1,
        )
        step = np.asarray(
            tuple(
                fraction * spacing
                for fraction, spacing in zip((0.5, 0.2), self.transfer.spacing)
            )
        )
        return jnp.asarray(start), jnp.asarray(start + step)

    def _state(
        self, electric: _Triple, magnetic: _Triple, charge: Array, /
    ) -> ReducedMaxwellState:
        return self.field.initialize(electric=electric, magnetic=magnetic, charge=charge)

    def field_with_charge(self, charge: Array, /) -> ReducedMaxwellState:
        zero = jnp.zeros_like(charge)
        return self._state((zero, zero, zero), (zero, zero, zero), charge)

    def _electrostatic(self, charge: Array, /) -> tuple[_Triple, Array]:
        epsilon = self.field.permittivity
        zero = jnp.zeros_like(charge)
        neutral = jnp.abs(jnp.sum(charge)) <= 1.0e-12 * jnp.maximum(
            1.0, jnp.sum(jnp.abs(charge))
        )
        if self.spatial_dimension == 1:
            spacing = self.transfer.spacing[0]
            electric = jnp.cumsum(charge) * spacing / epsilon
            if self.transfer.periodic[0]:
                # Periodic Gauss fixes E_x up to a constant; zero mean removes it.
                return (electric - jnp.mean(electric), zero, zero), neutral
            return (electric, zero, zero), jnp.asarray(True)
        if not all(self.transfer.periodic):
            raise ValueError(
                "Gauss-consistent initialization of reduced 2-D fields requires "
                "periodic axes."
            )
        # E = -forward(phi) with backward(forward(phi)) having symbol -lambda.
        eigenvalue = jnp.zeros(self.transfer.shape, dtype=charge.dtype)
        for axis, (count, spacing) in enumerate(
            zip(self.transfer.shape, self.transfer.spacing, strict=True)
        ):
            frequency = 2.0 * jnp.pi * jnp.fft.fftfreq(count)
            shape = [1, 1]
            shape[axis] = count
            eigenvalue = eigenvalue + (2.0 - 2.0 * jnp.cos(frequency)).reshape(shape) / (
                spacing**2
            )
        transformed = jnp.fft.fftn(charge)
        safe = jnp.where(eigenvalue > 0.0, eigenvalue, 1.0)
        potential = jnp.real(
            jnp.fft.ifftn(
                jnp.where(eigenvalue > 0.0, transformed / (epsilon * safe), 0.0)
            )
        )
        components = tuple(
            -(jnp.roll(potential, -1, axis=axis) - potential) / spacing
            for axis, spacing in enumerate(self.transfer.spacing)
        )
        return (components[0], components[1], zero), neutral

    def initialize_field(
        self, charge: Array, /, *, magnetic: Any = None
    ) -> tuple[ReducedMaxwellState, Array]:
        electric, successful = self._electrostatic(charge)
        zero = jnp.zeros_like(charge)
        flux = (
            (zero, zero, zero)
            if magnetic is None
            else tuple(jnp.asarray(value, dtype=charge.dtype) for value in magnetic)
        )
        if len(flux) != 3:
            raise ValueError("Reduced magnetic field must have three components.")
        return self._state(electric, (flux[0], flux[1], flux[2]), charge), successful

    def _gauss_charge(self, field: ReducedMaxwellState, /) -> Array:
        """Discrete ``ε ∇·E`` in the charge layout of the field update."""
        plan = self.field
        if isinstance(plan, CompatibleMaxwell1DPlan):
            return plan.permittivity * _charge_divergence(
                field.electric[0], plan.spacing, plan.periodic[0]
            )
        if isinstance(plan, CompatibleMaxwell2DPlan) and isinstance(
            field, CompatibleMaxwell2DState
        ):
            return plan.divergence_electric(field)
        raise TypeError("Reduced field state does not match its Maxwell plan.")

    def project_gauss(
        self, field: ReducedMaxwellState, charge: Array, /
    ) -> PICGaussProjectionResult:
        """Poisson projection onto ``charge``: exact 1-D cochain inverse or 2-D FFT.

        The residual ``ρ - ε∇·E`` is inverted by the same electrostatic solve
        that initializes the field (cumulative sum in 1-D, the Yee Laplacian
        symbol on periodic 2-D grids); ``B`` and absorber memory are unchanged.
        """
        residual = charge - self._gauss_charge(field)
        correction, successful = self._electrostatic(residual)
        projected = type(field)(
            (
                field.electric[0] + correction[0],
                field.electric[1] + correction[1],
                field.electric[2] + correction[2],
            ),
            field.magnetic,
            charge,
            field.pml_memory,
        )
        after = self._gauss_charge(projected) - charge
        return PICGaussProjectionResult(
            projected,
            jnp.max(jnp.abs(residual), initial=0.0),
            jnp.max(jnp.abs(after), initial=0.0),
            self.field_energy(projected) - self.field_energy(field),
            successful & jnp.all(jnp.isfinite(after)),
            "spectral-poisson" if self.spatial_dimension == 2 else "cochain-poisson",
        )

    def field_charge(self, field: ReducedMaxwellState, /) -> Array:
        return field.charge

    def field_energy(self, field: ReducedMaxwellState, /) -> Array:
        plan = self.field
        if isinstance(plan, CompatibleMaxwell1DPlan) and isinstance(
            field, CompatibleMaxwell1DState
        ):
            return plan.energy(field)
        if isinstance(plan, CompatibleMaxwell2DPlan) and isinstance(
            field, CompatibleMaxwell2DState
        ):
            return plan.energy(field)
        raise TypeError("Reduced field state does not match its Maxwell plan.")

    def deposit_charge(
        self,
        species: int,
        position: Array,
        macrocharge: Array,
        active: Array,
        /,
    ) -> tuple[Array, Array]:
        del species
        charge = self.transfer.deposit(position, macrocharge, active)
        return charge, jnp.all(jnp.isfinite(charge))

    def deposit(
        self,
        species: int,
        start: Array,
        end: Array,
        velocity: Array,
        macrocharge: Array,
        active: Array,
        step_size: Array,
        /,
    ) -> PICFieldDeposit:
        del species
        result = self.transfer.current(
            start, end, macrocharge, velocity, active, step_size
        )
        return PICFieldDeposit(
            result.current,
            result.start_charge,
            result.end_charge,
            result.maximum_continuity_defect,
            result.successful,
        )

    def deposit_all(
        self,
        starts: tuple[Array, ...],
        ends: tuple[Array, ...],
        velocities: tuple[Array, ...],
        macrocharges: tuple[Array, ...],
        actives: tuple[Array, ...],
        step_size: Array,
        /,
    ) -> PICFieldDeposit:
        return self.deposit(
            0,
            jnp.concatenate(starts),
            jnp.concatenate(ends),
            jnp.concatenate(velocities),
            jnp.concatenate(macrocharges),
            jnp.concatenate(actives),
            step_size,
        )

    def advance(
        self,
        time: Array,
        field: ReducedMaxwellState,
        current: _Triple,
        step_size: Array,
        /,
    ) -> PICFieldAdvance:
        del time
        plan = self.field
        if isinstance(plan, CompatibleMaxwell1DPlan) and isinstance(
            field, CompatibleMaxwell1DState
        ):
            advanced, diagnostics = plan.step(field, current, step_size)
        elif isinstance(plan, CompatibleMaxwell2DPlan) and isinstance(
            field, CompatibleMaxwell2DState
        ):
            advanced, diagnostics = plan.step(field, current, step_size)
        else:
            raise TypeError("Reduced field state does not match its Maxwell plan.")
        return PICFieldAdvance(
            advanced,
            advanced.charge,
            diagnostics.electric_constraint_linf,
            diagnostics.magnetic_constraint_linf,
            diagnostics.energy,
            diagnostics,
            diagnostics.successful,
        )

    def gather_fields(
        self,
        species: int,
        position: Array,
        active: Array,
        field: ReducedMaxwellState,
        /,
    ) -> tuple[Array, Array, Array]:
        del species
        electric = self.transfer.gather(position, field.electric, active)
        magnetic = self.transfer.gather(position, field.magnetic, active)
        # Nonperiodic CIC merges exterior half-stencils into boundary cells, so
        # every in-grid particle has full support.
        return electric, magnetic, jnp.ones(active.shape, dtype=jnp.bool_)

    @property
    def tensor_periodic(self) -> tuple[bool, ...]:
        return self.transfer.periodic

    def _map_component(
        self,
        value: Array,
        location: tuple[AxisEntityKind, ...],
        parity: tuple[int, ...],
        function: PICTensorMap,
        /,
    ) -> Array:
        # Cell arrays are intervals; face arrays store the upper face of each
        # cell. On a wall-bounded axis the lower wall face is implicit zero (the
        # Yee backward difference), so it is materialized for the map and dropped.
        padded = value
        for axis, (kind, periodic) in enumerate(
            zip(location, self.transfer.periodic, strict=True)
        ):
            if kind == "point" and not periodic:
                shape = list(padded.shape)
                shape[axis] = 1
                padded = jnp.concatenate(
                    (jnp.zeros(tuple(shape), dtype=padded.dtype), padded), axis=axis
                )
        mapped = function(padded, location, parity)
        for axis, (kind, periodic) in enumerate(
            zip(location, self.transfer.periodic, strict=True)
        ):
            if kind == "point" and not periodic:
                mapped = jnp.take(mapped, jnp.arange(1, mapped.shape[axis]), axis=axis)
        return mapped

    def _map_triple(
        self, value: _Triple, axial: bool, function: PICTensorMap, /
    ) -> _Triple:
        # Polar components (E, J) sit on faces along their own resolved axis,
        # axial components (B) on faces along the other resolved axes; faces
        # carry the odd mirror parity.
        dimension = self.spatial_dimension
        mapped = []
        for component, array in enumerate(value):
            faces = tuple(
                (axis != component) if axial else (axis == component)
                for axis in range(dimension)
            )
            location: tuple[AxisEntityKind, ...] = tuple(
                "point" if face else "interval" for face in faces
            )
            parity = tuple(-1 if face else 1 for face in faces)
            mapped.append(self._map_component(array, location, parity, function))
        return mapped[0], mapped[1], mapped[2]

    def tensor_template(self, kind: PICTensorKind, /) -> Any:
        zero = jnp.zeros(self.transfer.shape, dtype=jnp.float64)
        match kind:
            case "charge":
                return zero
            case "current":
                return zero, zero, zero
            case "field":
                return self.field_with_charge(zero)
            case _:
                raise ValueError("PIC tensor kind is invalid.")

    def map_tensors(
        self, kind: PICTensorKind, value: Any, function: PICTensorMap, /
    ) -> Any:
        """Map cell charge, the current triple, or the gathered ``E``/``B`` triples."""
        match kind:
            case "charge":
                dimension = self.spatial_dimension
                return self._map_component(
                    value, ("interval",) * dimension, (1,) * dimension, function
                )
            case "current":
                return self._map_triple(value, False, function)
            case "field":
                return type(value)(
                    self._map_triple(value.electric, False, function),
                    self._map_triple(value.magnetic, True, function),
                    value.charge,
                    value.pml_memory,
                )
            case _:
                raise ValueError("PIC tensor kind is invalid.")

    def dispersion_frequency(
        self, wavevector: ArrayLike, step_size: ArrayLike, /
    ) -> Array:
        """Yee vacuum dispersion over the ``d`` resolved wavevector components."""
        dt = float(jnp.asarray(step_size))
        if not 0.0 < dt <= self.field.stable_dt:
            raise ValueError("step_size must be positive and within the stable step.")
        k = jnp.asarray(wavevector, dtype=jnp.float64)
        if k.ndim != 2 or k.shape[1] != self.spatial_dimension:
            raise ValueError("wavevector must have shape (K, spatial_dimension).")
        spacing = jnp.asarray(self.transfer.spacing)
        speed = 1.0 / np.sqrt(self.field.permittivity * self.field.permeability)
        spatial = jnp.sqrt(jnp.sum((jnp.sin(0.5 * k * spacing) / spacing) ** 2, axis=-1))
        return 2.0 / dt * jnp.arcsin(speed * dt * spatial)

    def window_interval(self, axis: int, /) -> float:
        return self.transfer.spacing[axis]

    def window_bounds(self, axis: int, /) -> tuple[float, float]:
        return self.transfer.lower[axis], self.transfer.upper[axis]

    def shift_window(
        self, field: ReducedMaxwellState, axis: int, cells: int, /
    ) -> ReducedMaxwellState:
        """Translate E, B, and charge; CPML memory belongs to the fixed absorber."""
        electric = tuple(
            _shift_without_wrap(value, axis, cells) for value in field.electric
        )
        magnetic = tuple(
            _shift_without_wrap(value, axis, cells) for value in field.magnetic
        )
        return type(field)(
            (electric[0], electric[1], electric[2]),
            (magnetic[0], magnetic[1], magnetic[2]),
            _shift_without_wrap(field.charge, axis, cells),
            field.pml_memory,
        )

    def restart_component(self, field: ReducedMaxwellState, /) -> PICRestartComponent:
        return restart_component("field", self.solver_id, field)

    def restore_component(self, component: PICRestartComponent, /) -> ReducedMaxwellState:
        template = self.field_with_charge(jnp.zeros(self.transfer.shape))
        return restore_component(component, "field", self.solver_id, template)


__all__ = ["ReducedMaxwellPICFieldSolver"]
