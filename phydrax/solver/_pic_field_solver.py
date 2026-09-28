#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Minimal field-solver protocol for explicit electromagnetic PIC.

A prepared PIC field solver owns the discrete field, its Gauss charge, and the
particle↔grid transfers bound to its discretization. The PIC runtime
(`ElectromagneticPICPlan`) owns species, pusher, processes, boundaries,
recorders, filters, radiation ownership, and precision; it talks to the field
only through the core operations

- ``deposit``: charge-conserving current of one species over one step, with
  the start/end charge in the solver's own Gauss-charge representation;
- ``advance``: one field step driven by the summed current;
- ``gather(derivative_order)``: physical ``E``/``B`` (and for order one their
  spatial gradients) at particle positions.

The deposit↔Gauss pairing — advancing the field with a deposited current moves
the solver's current-driven Gauss charge exactly to the deposited end charge — is verified
numerically when the PIC run is prepared (`deposit_gauss_pairing_defect`).
Optional capabilities are structural protocols: `PICSpectralSymbol`,
`PICHuygensSampling`, `PICMultiDeposit`, `PICWindowShift`, `PICGalileanGrid`,
`PICEnergyAccounting`, `PICOpenDomain`, `PICRestartState`, and
`PICRelativisticSelfFields`. Prescribed external fields use the core
`phydrax.discretization.pic.ExternalFieldSource`.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, runtime_checkable, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._dtype_names import real_precision_dtype_name, RealPrecisionDType
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import AxisEntityKind
from ..discretization.pic import PICSpeciesPlan
from ..typing import parse
from ._maxwell_far_field import HuygensSurfacePhasors


PICGatherDerivativeOrder: TypeAlias = Literal[0, 1]
PICGaussProjectionRoute: TypeAlias = Literal["cochain-poisson", "spectral-poisson"]
PICSelfFieldInitialization: TypeAlias = Literal[
    "electrostatic", "relativistic-per-species"
]


class PICPrecisionPolicy(StrictModule, NonTrainableState):
    """Explicit field, particle, and accumulation dtypes of one PIC run."""

    field_dtype: RealPrecisionDType = eqx.field(static=True)
    particle_dtype: RealPrecisionDType = eqx.field(static=True)
    accumulation_dtype: RealPrecisionDType = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        field_dtype: Any = "float64",
        particle_dtype: Any = "float64",
        accumulation_dtype: Any = "float64",
    ) -> None:
        field = real_precision_dtype_name(field_dtype)
        particle = real_precision_dtype_name(particle_dtype)
        accumulation = real_precision_dtype_name(accumulation_dtype)
        if np.finfo(accumulation).bits < max(
            np.finfo(field).bits, np.finfo(particle).bits
        ):
            raise ValueError(
                "PIC accumulation precision must be at least the field and particle "
                "precision."
            )
        self.field_dtype = field
        self.particle_dtype = particle
        self.accumulation_dtype = accumulation
        self.policy_id = canonical_fingerprint(
            {
                "kind": "pic-precision-policy",
                "field": field,
                "particle": particle,
                "accumulation": accumulation,
            }
        )


class PICFieldDeposit(StrictModule):
    """Charge-conserving source in the solver's own current/charge layout.

    ``continuity_defect`` is the largest ``|Δρ/Δt + ∇·J|`` of the deposit and
    ``continuity_scale`` the unsigned charge-rate magnitude it is certified
    against (``max(|Δρ| + 2ρ_|q|)/Δt``, the size of the charges whose
    difference the residual is): its roundoff floor is relative to that scale.
    """

    current: Any
    start_charge: Array
    end_charge: Array
    continuity_defect: Array
    successful: Array
    continuity_scale: Array


class PICFieldSample(StrictModule):
    """Physical fields at particle positions.

    ``electric_gradient``/``magnetic_gradient`` are ``[N, 3, 3]`` arrays
    ``∂F_i/∂x_j`` for derivative order one (``None`` for order zero); spatial
    directions a reduced geometry is invariant along carry exact zeros.
    """

    electric: Array
    magnetic: Array
    electric_gradient: Array | None
    magnetic_gradient: Array | None
    successful: Array


class PICFieldAdvance(StrictModule):
    """One field step and its constraint/energy evidence.

    ``charge`` is the start Gauss charge moved by the step's current alone
    (the solver's discrete continuity update), in the solver's charge layout.
    Charge the field acquires by itself — induced wall charge of conducting
    boundaries, conduction or plasma charge of the medium, and absorber
    bookkeeping divergence — stays in the field state (`field_charge`) and is
    not part of ``charge``.
    """

    field: Any
    charge: Array
    electric_constraint: Array
    magnetic_constraint: Array
    energy: Array
    diagnostics: Any
    successful: Array


class PICFieldEnergy(StrictModule):
    """Electromagnetic and medium energy of one field state at its integer time.

    ``electric`` is ``½⟨E, ⋆ε_∞E⟩`` with the instantaneous electric response,
    ``magnetic`` is ``½⟨H, ⋆μ_∞H⟩`` minus the leapfrog half-kick term (so the
    lossless vacuum update exchanges it exactly with ``−∫E·J``), and
    ``material`` is the energy stored by the medium itself (polarization and
    magnetization oscillators, cold-plasma current).
    """

    electric: Array
    magnetic: Array
    material: Array


class PICGaussProjectionResult(StrictModule):
    """Field Gauss-projected onto a prescribed charge, with divergence evidence.

    ``divergence_before``/``divergence_after`` are max-norm Gauss residuals
    ``|∇·D − ρ|`` of the input and the projected field against the prescribed
    charge, in the solver's own charge layout. ``energy_change`` is the field
    energy added by the curl-free correction. ``route`` names the Poisson
    discretization the solver used.
    """

    field: Any
    divergence_before: Array
    divergence_after: Array
    energy_change: Array
    successful: Array
    route: PICGaussProjectionRoute = eqx.field(static=True)


class PICRestartComponent(StrictModule):
    """One independently admitted restart component.

    ``owner_id`` is the plan identity of the component's owner; a component is
    admitted only by an owner with the same identity.
    """

    name: str = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    leaves: tuple[Array, ...]


def restart_component(name: str, owner_id: str, tree: Any, /) -> PICRestartComponent:
    return PICRestartComponent(
        name, owner_id, tuple(jnp.asarray(value) for value in jax.tree.leaves(tree))
    )


def restore_component(
    component: PICRestartComponent,
    name: str,
    owner_id: str,
    template: Any,
    /,
) -> Any:
    """Admit one component against its owner identity and runtime template."""
    if not isinstance(component, PICRestartComponent):
        raise TypeError("Restart components must be PICRestartComponent.")
    if component.name != name:
        raise ValueError(f"Restart component {component.name!r} is not {name!r}.")
    if component.owner_id != owner_id:
        raise ValueError(
            f"Restart component {name!r} belongs to another plan and is not admitted."
        )
    leaves, treedef = jax.tree.flatten(template)
    if len(leaves) != len(component.leaves) or any(
        jnp.shape(value) != jnp.shape(expected)
        or jnp.result_type(value) != jnp.result_type(expected)
        for value, expected in zip(component.leaves, leaves, strict=True)
    ):
        raise ValueError(f"Restart component {name!r} has an incompatible layout.")
    return jax.tree.unflatten(treedef, list(component.leaves))


class AbstractPreparedPICFieldSolver(StrictModule):
    """Minimal prepared field solver consumed by `ElectromagneticPICPlan`."""

    solver_id: eqx.AbstractVar[str]
    spatial_dimension: eqx.AbstractVar[int]
    field_dtype: eqx.AbstractVar[RealPrecisionDType]

    @property
    @abc.abstractmethod
    def stable_step(self) -> Array:
        """Largest stable field step."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def displacement_widths(self) -> Array:
        """Per-axis length scale bounding one-step particle displacement."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate_species(self, species: tuple[PICSpeciesPlan, ...], /) -> None:
        """Refuse species this solver has no prepared transfer for."""
        raise NotImplementedError

    @abc.abstractmethod
    def pairing_probe(self, species: int, capacity: int, /) -> tuple[Array, Array]:
        """Deterministic interior start/end positions for the pairing check."""
        raise NotImplementedError

    @abc.abstractmethod
    def field_with_charge(self, charge: Array, /) -> Any:
        """Zero ``E``/``B`` field state carrying ``charge`` as its Gauss charge."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize_field(
        self, charge: Array, /, *, magnetic: Any = None
    ) -> tuple[Any, Array]:
        """Gauss-consistent electrostatic field for ``charge`` and its success flag.

        ``magnetic`` is an optional initial magnetic field in the solver's own
        layout (a degree-two cochain, or a reduced component triple).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def field_charge(self, field: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def field_energy(self, field: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def deposit_charge(
        self,
        species: int,
        position: Array,
        macrocharge: Array,
        active: Array,
        /,
    ) -> tuple[Array, Array]:
        """Instantaneous charge of one species and its success flag."""
        raise NotImplementedError

    @abc.abstractmethod
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
        """Charge-conserving current of one species along ``start → end``.

        ``velocity[N, 3]`` is the step-mean velocity; solvers that cannot infer
        currents along invariant directions from the path use it for those
        components.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def advance(
        self, time: Array, field: Any, current: Any, step_size: Array, /
    ) -> PICFieldAdvance:
        raise NotImplementedError

    @abc.abstractmethod
    def gather_fields(
        self,
        species: int,
        position: Array,
        active: Array,
        field: Any,
        /,
    ) -> tuple[Array, Array, Array]:
        """Physical ``E[N, 3]``, ``B[N, 3]``, and per-particle support."""
        raise NotImplementedError

    def gather(
        self,
        species: int,
        position: Array,
        active: Array,
        field: Any,
        /,
        *,
        derivative_order: PICGatherDerivativeOrder = 0,
    ) -> PICFieldSample:
        """Fields at particle positions and, for order one, their gradients.

        Gradients are exact forward-mode derivatives of the solver's own
        interpolant with respect to each particle's position; the interpolant
        of each particle depends only on its own position, so one tangent per
        spatial axis yields every particle's gradient column.
        """
        order = parse(derivative_order, PICGatherDerivativeOrder, "derivative_order")
        electric, magnetic, support = self.gather_fields(species, position, active, field)
        match order:
            case 0:
                electric_gradient = magnetic_gradient = None
            case 1:

                def fields(value: Array) -> tuple[Array, Array]:
                    e, b, _ = self.gather_fields(species, value, active, field)
                    return e, b

                columns = tuple(
                    jax.jvp(
                        fields,
                        (position,),
                        (jnp.zeros_like(position).at[:, axis].set(1.0),),
                    )[1]
                    for axis in range(position.shape[1])
                )
                padding = ((0, 0), (0, 0), (0, 3 - position.shape[1]))
                electric_gradient = jnp.pad(
                    jnp.stack(tuple(value[0] for value in columns), axis=-1), padding
                )
                magnetic_gradient = jnp.pad(
                    jnp.stack(tuple(value[1] for value in columns), axis=-1), padding
                )
                electric_gradient = jnp.where(
                    active[:, None, None], electric_gradient, 0.0
                )
                magnetic_gradient = jnp.where(
                    active[:, None, None], magnetic_gradient, 0.0
                )
            case _:
                raise ValueError("derivative_order is invalid.")
        finite = jnp.all(
            jnp.where(
                active[:, None],
                jnp.isfinite(electric) & jnp.isfinite(magnetic),
                True,
            )
        )
        return PICFieldSample(
            electric,
            magnetic,
            electric_gradient,
            magnetic_gradient,
            finite & jnp.all(support | ~active),
        )


class PICFilterContinuityReport(StrictModule, NonTrainableState):
    """Host evidence of one field filter bound to one prepared solver.

    ``interior_commutation_defect`` is ``max|F(Δρ(J)) − Δρ(F J)| / max|Δρ(J)|``
    for a deterministic probe current without wall-normal components, where
    ``Δρ`` is the Gauss-charge change of one solver step; a charge-conserving
    filter makes it roundoff. ``boundary_commutation_defect`` is the same
    measure for a probe carrying wall-normal current (identically the interior
    value when every axis is periodic). ``gauss_initialization_defect`` is the
    relative Gauss residual of the field initialized from filtered neutral
    charge.
    """

    filter_id: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    interior_commutation_defect: float = eqx.field(static=True)
    boundary_commutation_defect: float = eqx.field(static=True)
    gauss_initialization_defect: float = eqx.field(static=True)


class AbstractPICFieldFilter(StrictModule):
    """Linear smoothing applied identically to charge, current, and gathered fields.

    A charge-conserving filter commutes with the solver's discrete divergence,
    so the Gauss charge advanced by filtered current equals the filtered
    deposited charge. Filters are solver-agnostic plans; every operation
    receives the prepared solver that owns the value's layout.
    """

    filter_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def continuity_report(
        self, solver: AbstractPreparedPICFieldSolver, /
    ) -> PICFilterContinuityReport:
        """Refuse an unsupported solver, else measure filtered continuity."""
        raise NotImplementedError

    @abc.abstractmethod
    def filter_charge(
        self, solver: AbstractPreparedPICFieldSolver, charge: Array, /
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def filter_current(
        self, solver: AbstractPreparedPICFieldSolver, current: Any, /
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def filter_field(self, solver: AbstractPreparedPICFieldSolver, field: Any, /) -> Any:
        raise NotImplementedError


PICTensorKind: TypeAlias = Literal["charge", "current", "field"]
type PICTensorMap = Callable[[Array, tuple[AxisEntityKind, ...], tuple[int, ...]], Array]


@runtime_checkable
class PICTensorLayout(Protocol):
    """Uniform structured tensor components of charge, current, and gathered fields.

    ``map_tensors(kind, value, function)`` applies
    ``function(array, location, parity)`` to every component array of a
    ``"charge"``, ``"current"``, or ``"field"`` value and reassembles the
    solver value. ``location[axis]`` is ``"point"`` or ``"interval"``; along a
    nonperiodic axis walls sit at points and point arrays include both wall
    points. ``parity[axis]`` is the component's ±1 mirror parity (polar
    normal components and axial tangential components are odd). For
    ``"field"`` the components are the physical gathered ``E`` and ``B`` and the
    result is a field state whose gathered fields are the mapped ones.
    ``tensor_template(kind)`` is the zero value of a kind; charge templates
    and currents share the solver's own charge/current layouts.
    """

    @property
    def tensor_periodic(self) -> tuple[bool, ...]: ...

    def tensor_template(self, kind: PICTensorKind, /) -> Any: ...

    def map_tensors(
        self, kind: PICTensorKind, value: Any, function: PICTensorMap, /
    ) -> Any: ...


@runtime_checkable
class PICSpectralSymbol(Protocol):
    """Vacuum numerical dispersion ``ω(k)`` of the prepared field update."""

    def dispersion_frequency(
        self, wavevector: ArrayLike, step_size: ArrayLike, /
    ) -> Array: ...


@runtime_checkable
class PICHuygensSampling(Protocol):
    """Huygens-surface phasors acquired by the field solver's observers."""

    def huygens_phasors(self, field: Any, /) -> tuple[HuygensSurfacePhasors, ...]: ...


@runtime_checkable
class PICMultiDeposit(Protocol):
    """Fused current deposition of every species in one pass."""

    def deposit_all(
        self,
        starts: tuple[Array, ...],
        ends: tuple[Array, ...],
        velocities: tuple[Array, ...],
        macrocharges: tuple[Array, ...],
        actives: tuple[Array, ...],
        step_size: Array,
        /,
    ) -> PICFieldDeposit: ...


@runtime_checkable
class PICWindowShift(Protocol):
    """Integer-cell field translation along one uniform axis without wrap."""

    def window_interval(self, axis: int, /) -> float: ...

    def window_bounds(self, axis: int, /) -> tuple[float, float]: ...

    def shift_window(self, field: Any, axis: int, cells: int, /) -> Any: ...


@runtime_checkable
class PICGalileanGrid(Protocol):
    """Field grid translating uniformly at ``grid_velocity`` (Galilean coordinates).

    Particle positions are grid coordinates ``x − v_grid t``: the runtime drifts
    them by ``(v − v_grid)Δt`` and samples external fields at the lab position
    ``x + v_grid t``.
    """

    @property
    def grid_velocity(self) -> tuple[float, ...]: ...


@runtime_checkable
class PICEnergyAccounting(Protocol):
    """Field energy split and the power the field loses by itself.

    ``loss_power(field)`` is the source-free ``−dW/dt`` of the field update:
    conduction, material damping, impedance boundaries, and absorbing layers.
    The PIC ledger integrates it with the trapezoidal rule over each step.
    """

    def energy_components(self, field: Any, step_size: Array, /) -> PICFieldEnergy: ...

    def loss_power(self, field: Any, /) -> Array: ...


@runtime_checkable
class PICOpenDomain(Protocol):
    """Axis-aligned field box whose axes are periodic or wall-bounded.

    Particles must stay ``boundary_inset(species)[axis]`` inside every wall of
    a bounded axis so the species' deposit and gather stencils stay on the grid.
    """

    @property
    def domain_periodic(self) -> tuple[bool, ...]: ...

    @property
    def domain_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]: ...

    def boundary_inset(self, species: int, /) -> tuple[float, ...]: ...


@runtime_checkable
class PICRestartState(Protocol):
    """Field restart component admitted only by the same prepared solver."""

    def restart_component(self, field: Any, /) -> PICRestartComponent: ...

    def restore_component(self, component: PICRestartComponent, /) -> Any: ...


@runtime_checkable
class PICGaussProjection(Protocol):
    """Curl-free Poisson projection of the electric field onto a Gauss charge.

    ``project_gauss(field, charge)`` corrects ``D ← D − ε∇φ`` with
    ``∇·(ε∇φ) = ∇·D − ρ`` in the solver's own discrete divergence, so the
    returned field carries ``charge`` as its Gauss charge; magnetic,
    constitutive-memory, absorber, and observer state are unchanged. Required
    by population processes that redistribute charge (particle resampling).
    """

    def project_gauss(self, field: Any, charge: Array, /) -> PICGaussProjectionResult: ...


class PICRelativisticFieldResult(StrictModule):
    """Superposed lab-frame self-fields of uniformly drifting species.

    ``gauss_residual`` is the max-norm Gauss defect ``|−∇·D − ρ|`` of the
    superposed field against the summed charge on free (not grounded) Gauss
    entities; ``magnetic_divergence`` is the max-norm ``|∇·B|``. ``successful``
    requires every species' certified Poisson solve and finite fields.
    """

    field: Any
    gauss_residual: Array
    magnetic_divergence: Array
    successful: Array


@runtime_checkable
class PICRelativisticSelfFields(Protocol):
    """Boosted-Coulomb self-fields of species drifting along one grid axis.

    For species ``s`` with drift ``β_s`` (a velocity over ``speed_of_light``)
    along axis ``∥`` and ``γ_s = (1 − β_s²)^{-1/2}``, the solver solves
    ``−∇·(ε(∇⊥ + γ_s⁻² ê∥∂∥)φ_s) = ρ_s`` and superposes
    ``E_s = −(∇⊥φ_s + γ_s⁻² ê∥∂∥φ_s)`` and ``B_s = ∇ × (β_s φ_s / c) = β_s × E_s / c``,
    the lab-frame field of the species' rest-frame Coulomb field. A zero drift
    is the electrostatic field. ``drifts`` are host values.
    """

    def initialize_relativistic_field(
        self,
        charges: tuple[Array, ...],
        drifts: tuple[tuple[float, ...], ...],
        speed_of_light: float,
        /,
    ) -> PICRelativisticFieldResult: ...


def add_deposits(values: Sequence[PICFieldDeposit], /) -> PICFieldDeposit:
    """Superpose species deposits; continuity is linear in charge and current."""
    first, *rest = values
    total = first
    for value in rest:
        total = PICFieldDeposit(
            jax.tree.map(jnp.add, total.current, value.current),
            total.start_charge + value.start_charge,
            total.end_charge + value.end_charge,
            jnp.maximum(total.continuity_defect, value.continuity_defect),
            total.successful & value.successful,
            jnp.maximum(total.continuity_scale, value.continuity_scale),
        )
    return total


def chain_deposits(first: PICFieldDeposit, second: PICFieldDeposit, /) -> PICFieldDeposit:
    """Join two consecutive path segments deposited over the same step."""
    return PICFieldDeposit(
        jax.tree.map(jnp.add, first.current, second.current),
        first.start_charge,
        second.end_charge,
        jnp.maximum(first.continuity_defect, second.continuity_defect),
        first.successful & second.successful,
        jnp.maximum(first.continuity_scale, second.continuity_scale),
    )


@eqx.filter_jit
def _pairing_probe(
    solver: AbstractPreparedPICFieldSolver,
    species: int,
    start: Array,
    end: Array,
    active: Array,
    /,
) -> tuple[Array, Array]:
    """One species' probe deposit, field advance, and relative Gauss defect.

    The residual is certified above a charge-scaled floating-point floor. This
    prevents a uniformly populated periodic probe, whose exact charge change is
    zero, from dividing one rounding error by another.

    Module-level and compiled once per solver structure and probe shape: plans
    over the same prepared solver reuse the executable instead of dispatching
    the deposit and field update op by op at every preparation.
    """
    step = 0.5 * jnp.asarray(solver.stable_step, dtype=start.dtype)
    charge = jnp.where(active, 1.0, 0.0).astype(start.dtype)
    velocity = jnp.pad((end - start) / step, ((0, 0), (0, 3 - start.shape[1])))
    deposited = solver.deposit(species, start, end, velocity, charge, active, step)
    advanced = solver.advance(
        jnp.zeros((), dtype=start.dtype),
        solver.field_with_charge(deposited.start_charge),
        deposited.current,
        step,
    )
    change = jnp.max(jnp.abs(deposited.end_charge - deposited.start_charge))
    error = jnp.max(jnp.abs(advanced.charge - deposited.end_charge))
    charge_scale = (
        deposited.continuity_scale * step
        + jnp.max(jnp.abs(deposited.start_charge), initial=0.0)
        + jnp.max(jnp.abs(deposited.end_charge), initial=0.0)
    )
    roundoff = 64.0 * jnp.finfo(start.dtype).eps * charge_scale
    defect = jnp.maximum(error - roundoff, 0.0) / jnp.maximum(
        jnp.maximum(change, roundoff), jnp.finfo(start.dtype).tiny
    )
    return defect, deposited.successful & jnp.isfinite(defect)


def deposit_gauss_pairing_defect(
    solver: AbstractPreparedPICFieldSolver,
    species: tuple[PICSpeciesPlan, ...],
    /,
) -> float:
    """Relative defect between the advanced Gauss charge and deposited end charge.

    Each species deposits a deterministic probe path through its prepared
    transfer; the field solver then advances a zero field carrying the start
    charge with that current. A paired solver's current-driven Gauss charge
    (`PICFieldAdvance.charge`, the field's own discrete divergence applied to
    the current) lands on the deposited end charge to the charge-scaled
    floating-point floor; charge the field induces at conducting walls or in
    the medium is not a pairing defect. The probes of all species are evaluated
    before one host read of the evidence.
    """
    probes = []
    for index, value in enumerate(species):
        start, end = solver.pairing_probe(index, value.capacity)
        probes.append(
            _pairing_probe(
                solver, index, start, end, value.population.particles.active_mask
            )
        )
    defects, successes = jax.device_get(tuple(zip(*probes, strict=True)))
    if not all(bool(value) for value in successes):
        raise ValueError("PIC deposit↔Gauss pairing probe failed to deposit.")
    return max(float(value) for value in defects)


__all__ = [
    "AbstractPICFieldFilter",
    "AbstractPreparedPICFieldSolver",
    "PICFieldAdvance",
    "PICEnergyAccounting",
    "PICFieldDeposit",
    "PICFieldEnergy",
    "PICFieldSample",
    "PICFilterContinuityReport",
    "PICGatherDerivativeOrder",
    "PICGaussProjection",
    "PICGaussProjectionResult",
    "PICGaussProjectionRoute",
    "PICHuygensSampling",
    "PICGalileanGrid",
    "PICMultiDeposit",
    "PICOpenDomain",
    "PICPrecisionPolicy",
    "PICRelativisticFieldResult",
    "PICRelativisticSelfFields",
    "PICRestartComponent",
    "PICRestartState",
    "PICSelfFieldInitialization",
    "PICSpectralSymbol",
    "PICTensorKind",
    "PICTensorLayout",
    "PICTensorMap",
    "PICWindowShift",
]
