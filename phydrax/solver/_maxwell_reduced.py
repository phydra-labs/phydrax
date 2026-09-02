#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PreparedTensorGrid
from ._maxwell_boundaries import MaxwellBoundaryPlan
from ._maxwell_pml import MaxwellCPMLPlan, MaxwellCPMLState


def _forward(value: Array, axis: int, spacing: float, periodic: bool) -> Array:
    if periodic:
        shifted = jnp.roll(value, -1, axis=axis)
    else:
        indices = jnp.minimum(jnp.arange(value.shape[axis]) + 1, value.shape[axis] - 1)
        shifted = jnp.take(value, indices, axis=axis)
    return (shifted - value) / spacing


def _backward(value: Array, axis: int, spacing: float, periodic: bool) -> Array:
    if periodic:
        previous = jnp.roll(value, 1, axis=axis)
    else:
        indices = jnp.maximum(jnp.arange(value.shape[axis]) - 1, 0)
        previous = jnp.take(value, indices, axis=axis)
    return (value - previous) / spacing


class PreparedReducedMaxwellCPMLTerm(StrictModule, NonTrainableState):
    """One boundary-packed directional derivative memory."""

    indices: Array
    sigma: Array
    kappa: Array
    alpha: Array
    axis: int = eqx.field(static=True)
    component: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)


class PreparedReducedMaxwellCPML(StrictModule, NonTrainableState):
    """Prepared CPML profiles for fixed-shape reduced Maxwell derivatives."""

    electric_terms: tuple[PreparedReducedMaxwellCPMLTerm, ...]
    magnetic_terms: tuple[PreparedReducedMaxwellCPMLTerm, ...]
    electric_slots: tuple[int, ...] = eqx.field(static=True)
    magnetic_slots: tuple[int, ...] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MaxwellCPMLPlan,
        shape: tuple[int, ...],
        periodic: tuple[bool, ...],
        /,
    ):
        if not isinstance(plan, MaxwellCPMLPlan):
            raise TypeError("plan must be MaxwellCPMLPlan.")
        shape = tuple(int(value) for value in shape)
        periodic = tuple(bool(value) for value in periodic)
        dimension = len(shape)
        if (
            dimension not in (1, 2)
            or len(periodic) != dimension
            or any(value < 1 for value in shape)
        ):
            raise ValueError("Reduced Maxwell CPML requires a valid 1-D or 2-D shape.")
        widths = plan.widths * dimension if len(plan.widths) == 1 else plan.widths
        if len(widths) != dimension:
            raise ValueError("Reduced Maxwell CPML requires one width per axis.")
        for count, width, is_periodic in zip(shape, widths, periodic, strict=True):
            if width and is_periodic:
                raise ValueError("Periodic reduced Maxwell axes cannot carry CPML.")
            if 2 * width >= count:
                raise ValueError("Reduced Maxwell CPML leaves no undamped interior.")

        def make_term(axis: int, component: int, kind: str):
            width = widths[axis]
            if width == 0:
                return None
            grid = np.indices(shape, dtype=np.int64)
            coordinate = grid[axis]
            count = shape[axis]
            low = coordinate < width
            high = coordinate >= count - width
            mask = low | high
            low_depth = (width - coordinate - 0.5) / width
            high_depth = (coordinate - (count - width) + 0.5) / width
            depth = np.clip(np.maximum(low_depth, high_depth)[mask], 0.0, 1.0)
            indices = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)[mask]
            powered = depth**plan.sigma_order
            sigma_max = -(plan.sigma_order + 1.0) * np.log(plan.target_reflection) / width
            term_id = canonical_fingerprint(
                {
                    "kind": "prepared-reduced-maxwell-cpml-term",
                    "plan": plan.plan_id,
                    "shape": shape,
                    "field": kind,
                    "axis": axis,
                    "component": component,
                }
            )
            return PreparedReducedMaxwellCPMLTerm(
                jnp.asarray(indices),
                jnp.asarray(sigma_max * powered),
                jnp.asarray(1.0 + (plan.kappa_max - 1.0) * powered),
                jnp.asarray(plan.alpha_max * (1.0 - depth)),
                axis,
                component,
                term_id,
            )

        electric_terms = []
        magnetic_terms = []
        electric_slots = [-1] * (3 * dimension)
        magnetic_slots = [-1] * (3 * dimension)
        active_pairs = (
            ((0, 1), (0, 2)) if dimension == 1 else ((1, 0), (0, 1), (0, 2), (1, 2))
        )
        for kind, terms, slots in (
            ("electric", electric_terms, electric_slots),
            ("magnetic", magnetic_terms, magnetic_slots),
        ):
            for axis, component in active_pairs:
                term = make_term(axis, component, kind)
                if term is not None:
                    slots[3 * axis + component] = len(terms)
                    terms.append(term)
        self.electric_terms = tuple(electric_terms)
        self.magnetic_terms = tuple(magnetic_terms)
        self.electric_slots = tuple(electric_slots)
        self.magnetic_slots = tuple(magnetic_slots)
        self.shape = shape
        self.dimension = dimension
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-reduced-maxwell-cpml",
                "plan": plan.plan_id,
                "shape": shape,
                "electric_terms": tuple(term.term_id for term in electric_terms),
                "magnetic_terms": tuple(term.term_id for term in magnetic_terms),
            }
        )

    def initialize(self, /, *, dtype=float) -> MaxwellCPMLState:
        return MaxwellCPMLState(
            tuple(
                jnp.zeros(term.indices.shape, dtype=dtype) for term in self.electric_terms
            ),
            tuple(
                jnp.zeros(term.indices.shape, dtype=dtype) for term in self.magnetic_terms
            ),
        )

    def validate_state(self, state: MaxwellCPMLState, /) -> None:
        if not isinstance(state, MaxwellCPMLState):
            raise TypeError("Reduced Maxwell CPML state must be MaxwellCPMLState.")
        electric_shapes = tuple(term.indices.shape for term in self.electric_terms)
        magnetic_shapes = tuple(term.indices.shape for term in self.magnetic_terms)
        if (
            tuple(value.shape for value in state.electric_memory) != electric_shapes
            or tuple(value.shape for value in state.magnetic_memory) != magnetic_shapes
        ):
            raise ValueError("Reduced Maxwell CPML memory shapes are incompatible.")

    def reset(self, state: MaxwellCPMLState, /) -> MaxwellCPMLState:
        """Reset only CPML history while preserving its dtype and packed layout."""

        self.validate_state(state)
        return MaxwellCPMLState(
            tuple(jnp.zeros_like(value) for value in state.electric_memory),
            tuple(jnp.zeros_like(value) for value in state.magnetic_memory),
        )

    def apply(
        self,
        derivative: Array,
        state: MaxwellCPMLState,
        step_size: Array,
        /,
        *,
        electric: bool,
        axis: int,
        component: int,
    ) -> tuple[Array, MaxwellCPMLState]:
        """Advance one directional memory and return the CPML-modified derivative."""

        self.validate_state(state)
        terms = self.electric_terms if electric else self.magnetic_terms
        slots = self.electric_slots if electric else self.magnetic_slots
        memory = state.electric_memory if electric else state.magnetic_memory
        slot = slots[3 * axis + component]
        if slot < 0:
            return derivative, state
        term = terms[slot]
        flat = jnp.asarray(derivative).reshape((-1,))
        sample = flat[term.indices]
        old = memory[slot]
        dt = jnp.asarray(step_size)
        decay = jnp.exp(-(term.sigma / term.kappa + term.alpha) * dt)
        denominator = term.sigma * term.kappa + term.alpha * term.kappa**2
        coefficient = jnp.where(
            denominator > 0.0,
            term.sigma * (decay - 1.0) / denominator,
            0.0,
        )
        new = decay * old + coefficient * sample
        corrected = flat.at[term.indices].add((1.0 / term.kappa - 1.0) * sample + new)
        next_memory = memory[:slot] + (new,) + memory[slot + 1 :]
        next_state = (
            MaxwellCPMLState(next_memory, state.magnetic_memory)
            if electric
            else MaxwellCPMLState(state.electric_memory, next_memory)
        )
        return corrected.reshape(self.shape), next_state


def _select_cpml_state(
    predicate: Array,
    candidate: MaxwellCPMLState | None,
    old: MaxwellCPMLState | None,
    /,
) -> MaxwellCPMLState | None:
    if candidate is None or old is None:
        return old
    return MaxwellCPMLState(
        tuple(
            jnp.where(predicate, new, prior)
            for new, prior in zip(
                candidate.electric_memory, old.electric_memory, strict=True
            )
        ),
        tuple(
            jnp.where(predicate, new, prior)
            for new, prior in zip(
                candidate.magnetic_memory, old.magnetic_memory, strict=True
            )
        ),
    )


def _cpml_state_finite(state: MaxwellCPMLState | None, /) -> Array:
    if state is None:
        return jnp.asarray(True)
    memories = state.electric_memory + state.magnetic_memory
    if not memories:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in memories)))


def _apply_boundary_traces(
    fields: tuple[Array, Array, Array],
    boundaries: tuple[tuple[MaxwellBoundaryPlan | None, MaxwellBoundaryPlan | None], ...],
    step_size: Array,
    /,
    *,
    electric: bool,
) -> tuple[Array, Array, Array]:
    result = list(fields)
    for axis, pair in enumerate(boundaries):
        for side_index, boundary in enumerate(pair):
            if boundary is None:
                continue
            boundary_index = 0 if side_index == 0 else result[0].shape[axis] - 1
            for component in range(3):
                if component == axis:
                    continue
                value = result[component]
                trace = jnp.take(value, boundary_index, axis=axis)
                constrained = (electric and boundary.kind == "pec") or (
                    (not electric) and boundary.kind == "pmc"
                )
                if constrained:
                    trace = jnp.zeros_like(trace)
                elif boundary.kind == "impedance":
                    admittance = jnp.asarray(boundary.admittance, dtype=value.dtype)
                    trace = trace / (1.0 + step_size * admittance)
                indices = [slice(None)] * value.ndim
                indices[axis] = boundary_index
                result[component] = value.at[tuple(indices)].set(trace)
    return tuple(result)


class ReducedMaxwellDiagnostics(StrictModule):
    energy: Array
    electric_constraint_linf: Array
    magnetic_constraint_linf: Array
    source_power: Array
    power_balance_residual: Array
    step_fraction: Array
    finite: Array
    stable: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class CompatibleMaxwell2DState(StrictModule):
    electric: tuple[Array, Array, Array]
    magnetic: tuple[Array, Array, Array]
    charge: Array
    pml_memory: MaxwellCPMLState | None


class CompatibleMaxwell2DPlan(StrictModule, NonTrainableState):
    """Periodic 2D3V Yee/de-Rham Maxwell block with explicit staggering."""

    grid: PreparedTensorGrid
    permittivity: float = eqx.field(static=True)
    permeability: float = eqx.field(static=True)
    courant_factor: float = eqx.field(static=True)
    shape: tuple[int, int] = eqx.field(static=True)
    spacing: tuple[float, float] = eqx.field(static=True)
    periodic: tuple[bool, bool] = eqx.field(static=True)
    boundaries: tuple[tuple[MaxwellBoundaryPlan | None, MaxwellBoundaryPlan | None], ...]
    pml: PreparedReducedMaxwellCPML | None
    stable_dt: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        permittivity: float = 1.0,
        permeability: float = 1.0,
        courant_factor: float = 0.95,
        boundaries: tuple[
            tuple[MaxwellBoundaryPlan | None, MaxwellBoundaryPlan | None], ...
        ]
        | None = None,
        pml: MaxwellCPMLPlan | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid) or len(grid.shape) != 2:
            raise TypeError(
                "CompatibleMaxwell2DPlan requires a prepared 2-D tensor grid."
            )
        periodic = tuple(bool(axis.periodic) for axis in grid.structured_axes)
        boundary_pairs = (
            tuple((None, None) for _ in periodic)
            if boundaries is None
            else tuple(boundaries)
        )
        if len(boundary_pairs) != 2 or any(len(value) != 2 for value in boundary_pairs):
            raise ValueError("Reduced Maxwell requires two side plans per axis.")
        for axis, pair in enumerate(boundary_pairs):
            if periodic[axis] and any(value is not None for value in pair):
                raise ValueError("Periodic reduced Maxwell axes accept no boundary plan.")
            if not periodic[axis] and any(
                not isinstance(value, MaxwellBoundaryPlan) for value in pair
            ):
                raise ValueError("Nonperiodic reduced Maxwell axes require boundaries.")
        if pml is not None and not isinstance(pml, MaxwellCPMLPlan):
            raise TypeError("pml must be MaxwellCPMLPlan or None.")
        epsilon, mu, courant = (
            float(permittivity),
            float(permeability),
            float(courant_factor),
        )
        if epsilon <= 0.0 or mu <= 0.0 or not 0.0 < courant <= 1.0:
            raise ValueError("Reduced Maxwell material/Courant values are invalid.")
        widths = tuple(np.asarray(axis.interval_widths) for axis in grid.structured_axes)
        if any(not np.allclose(value, value[0]) for value in widths):
            raise ValueError("Reduced Maxwell currently requires uniform axes.")
        spacing = (float(widths[0][0]), float(widths[1][0]))
        shape = (
            int(grid.structured_axes[0].interval_centers.size),
            int(grid.structured_axes[1].interval_centers.size),
        )
        wave_speed = 1.0 / np.sqrt(epsilon * mu)
        stable = courant / (
            wave_speed * np.sqrt(sum(1.0 / value**2 for value in spacing))
        )
        prepared_pml = (
            None if pml is None else PreparedReducedMaxwellCPML(pml, shape, periodic)
        )
        self.grid = grid
        self.permittivity = epsilon
        self.permeability = mu
        self.courant_factor = courant
        self.shape = shape
        self.spacing = spacing
        self.periodic = periodic
        self.boundaries = boundary_pairs
        self.pml = prepared_pml
        self.stable_dt = stable
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compatible-maxwell-2d3v",
                "grid": grid.prepared_id,
                "epsilon": epsilon,
                "mu": mu,
                "courant": courant,
                "boundaries": tuple(
                    tuple(None if side is None else side.plan_id for side in pair)
                    for pair in boundary_pairs
                ),
                "pml": None if pml is None else pml.plan_id,
            }
        )

    def initialize(
        self,
        *,
        electric: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        magnetic: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        charge: ArrayLike | None = None,
    ) -> CompatibleMaxwell2DState:
        zero = jnp.zeros(self.shape)
        e = (
            (zero, zero, zero)
            if electric is None
            else tuple(jnp.asarray(v) for v in electric)
        )
        b = (
            (zero, zero, zero)
            if magnetic is None
            else tuple(jnp.asarray(v) for v in magnetic)
        )
        rho = zero if charge is None else jnp.asarray(charge)
        if any(value.shape != self.shape for value in e + b) or rho.shape != self.shape:
            raise ValueError(
                "Reduced 2-D Maxwell fields must share the cell-count shape."
            )
        dtype = jnp.result_type(*(e + b + (rho,)))
        memory = None if self.pml is None else self.pml.initialize(dtype=dtype)
        return CompatibleMaxwell2DState(e, b, rho, memory)

    def reset_pml(self, state: CompatibleMaxwell2DState, /) -> CompatibleMaxwell2DState:
        """Clear CPML history without changing the primary Maxwell checkpoint."""

        if self.pml is None:
            if state.pml_memory is not None:
                raise ValueError("Reduced Maxwell state has CPML memory without CPML.")
            return state
        if state.pml_memory is None:
            raise ValueError("Reduced Maxwell CPML state is missing its memory.")
        return CompatibleMaxwell2DState(
            state.electric,
            state.magnetic,
            state.charge,
            self.pml.reset(state.pml_memory),
        )

    def divergence_electric(self, state: CompatibleMaxwell2DState, /) -> Array:
        ex, ey, _ = state.electric
        dx, dy = self.spacing
        return self.permittivity * (
            _backward(ex, 0, dx, self.periodic[0])
            + _backward(ey, 1, dy, self.periodic[1])
        )

    def divergence_magnetic(self, state: CompatibleMaxwell2DState, /) -> Array:
        bx, by, _ = state.magnetic
        dx, dy = self.spacing
        return _backward(bx, 0, dx, self.periodic[0]) + _backward(
            by, 1, dy, self.periodic[1]
        )

    def energy(self, state: CompatibleMaxwell2DState, /) -> Array:
        volume = self.spacing[0] * self.spacing[1]
        e2 = jnp.sum(jnp.stack(tuple(jnp.sum(value * value) for value in state.electric)))
        b2 = jnp.sum(jnp.stack(tuple(jnp.sum(value * value) for value in state.magnetic)))
        return 0.5 * volume * (self.permittivity * e2 + b2 / self.permeability)

    def step(
        self,
        state: CompatibleMaxwell2DState,
        electric_current: tuple[ArrayLike, ArrayLike, ArrayLike],
        step_size: ArrayLike,
        /,
    ) -> tuple[CompatibleMaxwell2DState, ReducedMaxwellDiagnostics]:
        dt = jnp.asarray(step_size).reshape(())
        current = tuple(jnp.asarray(value) for value in electric_current)
        if any(value.shape != self.shape for value in current):
            raise ValueError("Reduced 2-D current components must match the field shape.")
        if self.pml is None:
            if state.pml_memory is not None:
                raise ValueError("Reduced Maxwell state has CPML memory without CPML.")
        elif state.pml_memory is None:
            raise ValueError("Reduced Maxwell CPML state is missing its memory.")
        else:
            self.pml.validate_state(state.pml_memory)
        ex, ey, ez = state.electric
        bx, by, bz = state.magnetic
        dx, dy = self.spacing
        pml_memory = state.pml_memory
        d_y_ez = _forward(ez, 1, dy, self.periodic[1])
        d_x_ez = _forward(ez, 0, dx, self.periodic[0])
        d_x_ey = _forward(ey, 0, dx, self.periodic[0])
        d_y_ex = _forward(ex, 1, dy, self.periodic[1])
        if self.pml is not None:
            d_y_ez, pml_memory = self.pml.apply(
                d_y_ez, pml_memory, 0.5 * dt, electric=False, axis=1, component=0
            )
            d_x_ez, pml_memory = self.pml.apply(
                d_x_ez, pml_memory, 0.5 * dt, electric=False, axis=0, component=1
            )
            d_x_ey, pml_memory = self.pml.apply(
                d_x_ey, pml_memory, 0.5 * dt, electric=False, axis=0, component=2
            )
            d_y_ex, pml_memory = self.pml.apply(
                d_y_ex, pml_memory, 0.5 * dt, electric=False, axis=1, component=2
            )
        half_bx = bx - 0.5 * dt * d_y_ez
        half_by = by + 0.5 * dt * d_x_ez
        half_bz = bz - 0.5 * dt * (d_x_ey - d_y_ex)
        jx, jy, jz = current
        d_y_bz = _backward(half_bz / self.permeability, 1, dy, self.periodic[1])
        d_x_bz = _backward(half_bz / self.permeability, 0, dx, self.periodic[0])
        d_x_by = _backward(half_by / self.permeability, 0, dx, self.periodic[0])
        d_y_bx = _backward(half_bx / self.permeability, 1, dy, self.periodic[1])
        if self.pml is not None:
            d_y_bz, pml_memory = self.pml.apply(
                d_y_bz, pml_memory, dt, electric=True, axis=1, component=0
            )
            d_x_bz, pml_memory = self.pml.apply(
                d_x_bz, pml_memory, dt, electric=True, axis=0, component=1
            )
            d_x_by, pml_memory = self.pml.apply(
                d_x_by, pml_memory, dt, electric=True, axis=0, component=2
            )
            d_y_bx, pml_memory = self.pml.apply(
                d_y_bx, pml_memory, dt, electric=True, axis=1, component=2
            )
        next_ex = ex + dt / self.permittivity * (d_y_bz - jx)
        next_ey = ey + dt / self.permittivity * (-d_x_bz - jy)
        next_ez = ez + dt / self.permittivity * (d_x_by - d_y_bx - jz)
        d_y_next_ez = _forward(next_ez, 1, dy, self.periodic[1])
        d_x_next_ez = _forward(next_ez, 0, dx, self.periodic[0])
        d_x_next_ey = _forward(next_ey, 0, dx, self.periodic[0])
        d_y_next_ex = _forward(next_ex, 1, dy, self.periodic[1])
        if self.pml is not None:
            d_y_next_ez, pml_memory = self.pml.apply(
                d_y_next_ez, pml_memory, 0.5 * dt, electric=False, axis=1, component=0
            )
            d_x_next_ez, pml_memory = self.pml.apply(
                d_x_next_ez, pml_memory, 0.5 * dt, electric=False, axis=0, component=1
            )
            d_x_next_ey, pml_memory = self.pml.apply(
                d_x_next_ey, pml_memory, 0.5 * dt, electric=False, axis=0, component=2
            )
            d_y_next_ex, pml_memory = self.pml.apply(
                d_y_next_ex, pml_memory, 0.5 * dt, electric=False, axis=1, component=2
            )
        next_bx = half_bx - 0.5 * dt * d_y_next_ez
        next_by = half_by + 0.5 * dt * d_x_next_ez
        next_bz = half_bz - 0.5 * dt * (d_x_next_ey - d_y_next_ex)
        next_charge = state.charge - dt * (
            _backward(jx, 0, dx, self.periodic[0])
            + _backward(jy, 1, dy, self.periodic[1])
        )
        next_electric = _apply_boundary_traces(
            (next_ex, next_ey, next_ez), self.boundaries, dt, electric=True
        )
        next_magnetic = _apply_boundary_traces(
            (next_bx, next_by, next_bz), self.boundaries, dt, electric=False
        )
        candidate = CompatibleMaxwell2DState(
            next_electric,
            next_magnetic,
            next_charge,
            pml_memory,
        )
        old_energy = self.energy(state)
        new_energy = self.energy(candidate)
        source_power = (
            self.spacing[0]
            * self.spacing[1]
            * sum(
                jnp.sum(e * current_component)
                for e, current_component in zip(candidate.electric, current, strict=True)
            )
        )
        gauss = jnp.max(jnp.abs(self.divergence_electric(candidate) - next_charge))
        magnetic = jnp.max(jnp.abs(self.divergence_magnetic(candidate)))
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in candidate.electric + candidate.magnetic
                )
            )
        ) & jnp.all(jnp.isfinite(next_charge))
        finite = finite & _cpml_state_finite(candidate.pml_memory)
        stable = jnp.isfinite(dt) & (dt > 0.0) & (dt <= self.stable_dt)
        successful = finite & stable
        accepted = CompatibleMaxwell2DState(
            tuple(
                jnp.where(successful, new, old)
                for new, old in zip(candidate.electric, state.electric, strict=True)
            ),
            tuple(
                jnp.where(successful, new, old)
                for new, old in zip(candidate.magnetic, state.magnetic, strict=True)
            ),
            jnp.where(successful, candidate.charge, state.charge),
            _select_cpml_state(successful, candidate.pml_memory, state.pml_memory),
        )
        diagnostics = ReducedMaxwellDiagnostics(
            new_energy,
            gauss,
            magnetic,
            source_power,
            new_energy - old_energy + dt * source_power,
            dt / self.stable_dt,
            finite,
            stable,
            successful,
            self.plan_id,
        )
        return accepted, diagnostics


class CompatibleMaxwell1DState(StrictModule):
    electric: tuple[Array, Array, Array]
    magnetic: tuple[Array, Array, Array]
    charge: Array
    pml_memory: MaxwellCPMLState | None


class CompatibleMaxwell1DPlan(StrictModule, NonTrainableState):
    """Periodic 1D3V compatible longitudinal/transverse Maxwell blocks."""

    grid: PreparedTensorGrid
    permittivity: float = eqx.field(static=True)
    permeability: float = eqx.field(static=True)
    courant_factor: float = eqx.field(static=True)
    count: int = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    stable_dt: float = eqx.field(static=True)
    periodic: tuple[bool] = eqx.field(static=True)
    boundaries: tuple[tuple[MaxwellBoundaryPlan | None, MaxwellBoundaryPlan | None], ...]
    pml: PreparedReducedMaxwellCPML | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        permittivity: float = 1.0,
        permeability: float = 1.0,
        courant_factor: float = 0.95,
        boundaries: tuple[
            tuple[MaxwellBoundaryPlan | None, MaxwellBoundaryPlan | None], ...
        ]
        | None = None,
        pml: MaxwellCPMLPlan | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid) or len(grid.shape) != 1:
            raise TypeError(
                "CompatibleMaxwell1DPlan requires a prepared 1-D tensor grid."
            )
        axis = grid.structured_axes[0]
        periodic = (bool(axis.periodic),)
        boundary_pairs = ((None, None),) if boundaries is None else tuple(boundaries)
        if len(boundary_pairs) != 1 or len(boundary_pairs[0]) != 2:
            raise ValueError("Reduced 1-D Maxwell requires two side boundary plans.")
        if periodic[0] and any(value is not None for value in boundary_pairs[0]):
            raise ValueError("Periodic reduced Maxwell accepts no boundary plan.")
        if not periodic[0] and any(
            not isinstance(value, MaxwellBoundaryPlan) for value in boundary_pairs[0]
        ):
            raise ValueError("Nonperiodic reduced Maxwell requires boundary plans.")
        if pml is not None and not isinstance(pml, MaxwellCPMLPlan):
            raise TypeError("pml must be MaxwellCPMLPlan or None.")
        epsilon, mu, courant = (
            float(permittivity),
            float(permeability),
            float(courant_factor),
        )
        widths = np.asarray(axis.interval_widths)
        if (
            epsilon <= 0.0
            or mu <= 0.0
            or not 0.0 < courant <= 1.0
            or not np.allclose(widths, widths[0])
        ):
            raise ValueError("Reduced 1-D Maxwell parameters/grid are invalid.")
        spacing = float(widths[0])
        stable = courant * spacing * np.sqrt(epsilon * mu)
        count = int(axis.interval_centers.size)
        prepared_pml = (
            None if pml is None else PreparedReducedMaxwellCPML(pml, (count,), periodic)
        )
        self.grid = grid
        self.permittivity = epsilon
        self.permeability = mu
        self.courant_factor = courant
        self.count = count
        self.spacing = spacing
        self.stable_dt = stable
        self.periodic = periodic
        self.boundaries = boundary_pairs
        self.pml = prepared_pml
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compatible-maxwell-1d3v",
                "grid": grid.prepared_id,
                "epsilon": epsilon,
                "mu": mu,
                "courant": courant,
                "boundaries": tuple(
                    None if value is None else value.plan_id
                    for value in boundary_pairs[0]
                ),
                "pml": None if pml is None else pml.plan_id,
            }
        )

    def initialize(
        self,
        *,
        electric: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        magnetic: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        charge: ArrayLike | None = None,
    ) -> CompatibleMaxwell1DState:
        zero = jnp.zeros((self.count,))
        electric_ = (
            (zero, zero, zero)
            if electric is None
            else tuple(jnp.asarray(value) for value in electric)
        )
        magnetic_ = (
            (zero, zero, zero)
            if magnetic is None
            else tuple(jnp.asarray(value) for value in magnetic)
        )
        charge_ = zero if charge is None else jnp.asarray(charge)
        expected = (self.count,)
        if (
            any(value.shape != expected for value in electric_ + magnetic_)
            or charge_.shape != expected
        ):
            raise ValueError("Reduced 1-D Maxwell fields must match the grid count.")
        dtype = jnp.result_type(*(electric_ + magnetic_ + (charge_,)))
        memory = None if self.pml is None else self.pml.initialize(dtype=dtype)
        return CompatibleMaxwell1DState(electric_, magnetic_, charge_, memory)

    def reset_pml(self, state: CompatibleMaxwell1DState, /) -> CompatibleMaxwell1DState:
        """Clear CPML history without changing the primary Maxwell checkpoint."""

        if self.pml is None:
            if state.pml_memory is not None:
                raise ValueError("Reduced Maxwell state has CPML memory without CPML.")
            return state
        if state.pml_memory is None:
            raise ValueError("Reduced Maxwell CPML state is missing its memory.")
        return CompatibleMaxwell1DState(
            state.electric,
            state.magnetic,
            state.charge,
            self.pml.reset(state.pml_memory),
        )

    def energy(self, state: CompatibleMaxwell1DState, /) -> Array:
        electric_energy = jnp.sum(
            jnp.stack(tuple(jnp.sum(value**2) for value in state.electric))
        )
        magnetic_energy = jnp.sum(
            jnp.stack(tuple(jnp.sum(value**2) for value in state.magnetic))
        )
        return (
            0.5
            * self.spacing
            * (self.permittivity * electric_energy + magnetic_energy / self.permeability)
        )

    def step(
        self,
        state: CompatibleMaxwell1DState,
        electric_current: tuple[ArrayLike, ArrayLike, ArrayLike],
        step_size: ArrayLike,
        /,
    ) -> tuple[CompatibleMaxwell1DState, ReducedMaxwellDiagnostics]:
        dt = jnp.asarray(step_size).reshape(())
        current = tuple(jnp.asarray(value) for value in electric_current)
        if any(value.shape != (self.count,) for value in current):
            raise ValueError("Reduced 1-D currents must match the grid count.")
        if self.pml is None:
            if state.pml_memory is not None:
                raise ValueError("Reduced Maxwell state has CPML memory without CPML.")
        elif state.pml_memory is None:
            raise ValueError("Reduced Maxwell CPML state is missing its memory.")
        else:
            self.pml.validate_state(state.pml_memory)
        ex, ey, ez = state.electric
        bx, by, bz = state.magnetic
        pml_memory = state.pml_memory
        d_x_ez = _forward(ez, 0, self.spacing, self.periodic[0])
        d_x_ey = _forward(ey, 0, self.spacing, self.periodic[0])
        if self.pml is not None:
            d_x_ez, pml_memory = self.pml.apply(
                d_x_ez, pml_memory, 0.5 * dt, electric=False, axis=0, component=1
            )
            d_x_ey, pml_memory = self.pml.apply(
                d_x_ey, pml_memory, 0.5 * dt, electric=False, axis=0, component=2
            )
        half_by = by + 0.5 * dt * d_x_ez
        half_bz = bz - 0.5 * dt * d_x_ey
        jx, jy, jz = current
        next_ex = ex - dt * jx / self.permittivity
        d_x_bz = _backward(half_bz / self.permeability, 0, self.spacing, self.periodic[0])
        d_x_by = _backward(half_by / self.permeability, 0, self.spacing, self.periodic[0])
        if self.pml is not None:
            d_x_bz, pml_memory = self.pml.apply(
                d_x_bz, pml_memory, dt, electric=True, axis=0, component=1
            )
            d_x_by, pml_memory = self.pml.apply(
                d_x_by, pml_memory, dt, electric=True, axis=0, component=2
            )
        next_ey = ey + dt / self.permittivity * (-d_x_bz - jy)
        next_ez = ez + dt / self.permittivity * (d_x_by - jz)
        d_x_next_ez = _forward(next_ez, 0, self.spacing, self.periodic[0])
        d_x_next_ey = _forward(next_ey, 0, self.spacing, self.periodic[0])
        if self.pml is not None:
            d_x_next_ez, pml_memory = self.pml.apply(
                d_x_next_ez, pml_memory, 0.5 * dt, electric=False, axis=0, component=1
            )
            d_x_next_ey, pml_memory = self.pml.apply(
                d_x_next_ey, pml_memory, 0.5 * dt, electric=False, axis=0, component=2
            )
        next_by = half_by + 0.5 * dt * d_x_next_ez
        next_bz = half_bz - 0.5 * dt * d_x_next_ey
        charge = state.charge - dt * _backward(jx, 0, self.spacing, self.periodic[0])
        next_electric = _apply_boundary_traces(
            (next_ex, next_ey, next_ez), self.boundaries, dt, electric=True
        )
        next_magnetic = _apply_boundary_traces(
            (bx, next_by, next_bz), self.boundaries, dt, electric=False
        )
        candidate = CompatibleMaxwell1DState(
            next_electric, next_magnetic, charge, pml_memory
        )
        old_energy, new_energy = self.energy(state), self.energy(candidate)
        source_power = self.spacing * sum(
            jnp.sum(e * value)
            for e, value in zip(candidate.electric, current, strict=True)
        )
        gauss = jnp.max(
            jnp.abs(
                self.permittivity
                * _backward(candidate.electric[0], 0, self.spacing, self.periodic[0])
                - charge
            )
        )
        finite = (
            jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(value))
                        for value in candidate.electric + candidate.magnetic
                    )
                )
            )
            & jnp.all(jnp.isfinite(charge))
            & _cpml_state_finite(candidate.pml_memory)
        )
        stable = jnp.isfinite(dt) & (dt > 0.0) & (dt <= self.stable_dt)
        successful = finite & stable
        accepted = CompatibleMaxwell1DState(
            tuple(
                jnp.where(successful, new, old)
                for new, old in zip(candidate.electric, state.electric, strict=True)
            ),
            tuple(
                jnp.where(successful, new, old)
                for new, old in zip(candidate.magnetic, state.magnetic, strict=True)
            ),
            jnp.where(successful, candidate.charge, state.charge),
            _select_cpml_state(successful, candidate.pml_memory, state.pml_memory),
        )
        return accepted, ReducedMaxwellDiagnostics(
            new_energy,
            gauss,
            jnp.asarray(0.0, dtype=charge.dtype),
            source_power,
            new_energy - old_energy + dt * source_power,
            dt / self.stable_dt,
            finite,
            stable,
            successful,
            self.plan_id,
        )


__all__ = [
    "CompatibleMaxwell1DPlan",
    "CompatibleMaxwell1DState",
    "CompatibleMaxwell2DPlan",
    "CompatibleMaxwell2DState",
    "PreparedReducedMaxwellCPML",
    "PreparedReducedMaxwellCPMLTerm",
    "ReducedMaxwellDiagnostics",
]
