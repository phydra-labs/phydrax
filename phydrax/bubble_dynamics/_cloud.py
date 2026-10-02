#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bubble clouds: species groups, implicit coupled radial dynamics and Bjerknes forces.

Bubbles with one static law structure form a `BubbleSpeciesGroup` whose dynamic
coefficients are batched along the member axis; heterogeneous clouds are a
static tuple of groups, so no runtime selector exists per bubble. Every bubble
is evaluated by its own composed `RadialBubbleModel` (all hidden `R̈` terms of
the four radial equations are in its per-unit-density inertia `a_i`), and the
incompressible neighbour near field couples the accelerations implicitly
(`_cloud_coupling`); no neighbour `R̈` is lagged. The `"dense"` route solves the
coupled system with a native dense Cholesky factorization under an explicit
element/flop bound; the `"fmm"` route is matrix-free conjugate gradients whose
pair action is the signed Laplace monopole FMM. The `"retarded"` coupling
replaces the instantaneous near field by retarded neighbour sources
(`_cloud_retarded`). Bubbles may translate with declared added mass under
primary and secondary Bjerknes forces. Overlap is a terminal `OVERLAP` status;
no merged or resolved bubble is synthesized.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import fixed_field, NonTrainableState, parameter_field
from .._validation import positive_finite_float, positive_integer
from ..discretization.spatial import MortonAddressPlan, MortonRadiusRelationPlan
from ..solver import CartesianExpansionSpace, CartesianFMMResourceEvidence, UniformFMMPlan
from ..typing import checked, parse
from ._cloud_coupling import (
    AbstractCloudCoupling,
    CloudField,
    coupling_spectrum,
    dense_inverse_distance,
    DenseCloudCoupling,
    FMMCloudCoupling,
)
from ._contracts import (
    AbstractBubblePressureDrive,
    BubbleScales,
    PressureDriveEvaluation,
    scalar_parameter,
)
from ._emission import FarFieldEmissionPlan, FarFieldEmissionResult
from ._events import BubbleEventPolicy
from ._gas import sphere_volume
from ._radial import BubbleEquilibrium, BubbleState, RadialBubbleModel, RadialBubbleRates
from ._single import BubbleDifferentiation, SingleBubbleIntegrator
from ._status import bubble_status_successful, BubbleDynamicsStatus
from ._validity import BubbleValidityEvidence, BubbleValidityPolicy


BubbleCloudRoute: TypeAlias = Literal["auto", "dense", "fmm"]
ResolvedBubbleCloudRoute: TypeAlias = Literal["dense", "fmm"]
BubbleCloudCoupling: TypeAlias = Literal["incompressible", "retarded"]
BubbleCloudPressureFieldKind: TypeAlias = Literal["uniform", "standing_wave"]
CloudIntegrator: TypeAlias = Literal["explicit", "stiff"]


def _member_vector(
    value: ArrayLike, count: int, name: str, /, *, positive: bool
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array of shape ({count},).")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive.")
    return array


def _member_vectors(value: ArrayLike, count: int, name: str, /) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array of shape ({count}, 3).")
    return array


def _bubble_ids(values: Sequence[int], count: int, /) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("bubble_ids must be a sequence of integers.")
    ids = tuple(values)
    if len(ids) != count:
        raise ValueError(f"bubble_ids must contain {count} identifiers.")
    checked = []
    for value in ids:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("bubble_ids must be integers.")
        if value < 0:
            raise ValueError("bubble_ids must be nonnegative.")
        checked.append(int(value))
    if len(set(checked)) != count:
        raise ValueError("bubble_ids must be unique.")
    return tuple(checked)


def _stacked_model(
    model: RadialBubbleModel | Sequence[RadialBubbleModel], count: int, /
) -> RadialBubbleModel:
    """One static law structure whose array leaves carry the member axis."""
    if isinstance(model, RadialBubbleModel):
        reference = model
        dynamic, static = eqx.partition(reference, eqx.is_array)
        stacked = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (count,) + leaf.shape), dynamic
        )
    else:
        models = tuple(model)
        if len(models) != count or not all(
            isinstance(item, RadialBubbleModel) for item in models
        ):
            raise TypeError(
                f"model must be a RadialBubbleModel or a sequence of {count} RadialBubbleModels."
            )
        reference = models[0]
        dynamic, static = eqx.partition(reference, eqx.is_array)
        structure = jax.tree.structure(dynamic)
        shapes = [leaf.shape for leaf in jax.tree.leaves(dynamic)]
        dynamics = []
        for item in models:
            item_dynamic, item_static = eqx.partition(item, eqx.is_array)
            if (
                item.model_id != reference.model_id
                or jax.tree.structure(item_dynamic) != structure
                or [leaf.shape for leaf in jax.tree.leaves(item_dynamic)] != shapes
                or not eqx.tree_equal(item_static, static)
            ):
                raise ValueError(
                    "Every member model of a species group must share one law structure."
                )
            dynamics.append(item_dynamic)
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *dynamics)
    if reference.interface.guard_count != 0:
        raise ValueError(
            "Bubble clouds require smooth interface laws (guard_count == 0); piecewise "
            "shells such as MarmottantShell are not supported in clouds "
            "(GompertzMarmottantShell is the smooth alternative)."
        )
    return eqx.combine(stacked, static)


class BubbleSpeciesGroup(StrictModule):
    """Bubbles sharing one static law structure with batched dynamic coefficients.

    `model` is one `RadialBubbleModel` shared by every member or a sequence of
    member models with an identical static structure (equal `model_id`, tree
    structure and leaf shapes); their dynamic coefficients are stacked along the
    member axis. Positions are absolute (m). Members start at their equilibrium
    radius at rest unless `initial_radii`/`initial_wall_velocities` are given;
    `initial_velocities` are translational velocities (m/s). `bubble_ids` are
    stable, cloud-unique identifiers. Only smooth interface laws are accepted.
    """

    model: RadialBubbleModel
    equilibrium_radius: Array = parameter_field()
    position: Array = parameter_field()
    initial_radius: Array = fixed_field()
    initial_wall_velocity: Array = fixed_field()
    initial_velocity: Array = fixed_field()
    bubble_ids: tuple[int, ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)
    group_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: RadialBubbleModel | Sequence[RadialBubbleModel],
        equilibrium_radii: ArrayLike,
        positions: ArrayLike,
        /,
        *,
        bubble_ids: Sequence[int],
        initial_radii: ArrayLike | None = None,
        initial_wall_velocities: ArrayLike | None = None,
        initial_velocities: ArrayLike | None = None,
    ) -> None:
        radii = np.asarray(equilibrium_radii, dtype=np.float64)
        if radii.ndim != 1 or radii.shape[0] == 0:
            raise ValueError("equilibrium_radii must be a non-empty rank-1 array.")
        count = radii.shape[0]
        radii = _member_vector(radii, count, "equilibrium_radii", positive=True)
        points = _member_vectors(positions, count, "positions")
        ids = _bubble_ids(bubble_ids, count)
        initial = (
            radii
            if initial_radii is None
            else _member_vector(initial_radii, count, "initial_radii", positive=True)
        )
        wall = (
            np.zeros((count,))
            if initial_wall_velocities is None
            else _member_vector(
                initial_wall_velocities, count, "initial_wall_velocities", positive=False
            )
        )
        velocity = (
            np.zeros((count, 3))
            if initial_velocities is None
            else _member_vectors(initial_velocities, count, "initial_velocities")
        )
        stacked = _stacked_model(model, count)
        self.model = stacked
        self.equilibrium_radius = jnp.asarray(radii, dtype=jnp.float64)
        self.position = jnp.asarray(points, dtype=jnp.float64)
        self.initial_radius = jnp.asarray(initial, dtype=jnp.float64)
        self.initial_wall_velocity = jnp.asarray(wall, dtype=jnp.float64)
        self.initial_velocity = jnp.asarray(velocity, dtype=jnp.float64)
        self.bubble_ids = ids
        self.size = count
        self.group_id = canonical_fingerprint(
            {
                "kind": "bubble-species-group",
                "model": stacked.model_id,
                "bubble_ids": list(ids),
            }
        )


class BubbleTranslation(StrictModule):
    """Declared added-mass translation under Bjerknes forces and linear drag.

    `d(m_a v)/dt = F₁ + F₂ + F_d` with added mass `m_a = C_a ρ V` (gas mass
    neglected), primary Bjerknes force `F₁ = −V ∇p_ac`, secondary Bjerknes force
    `F₂ = −V ∇p_nb` from the neighbour monopole pressure, and drag
    `F_d = −C_d π μ R v` (`C_d = 12`: Levich high-Reynolds clean bubble;
    `C_d = 4`: Hadamard–Rybczynski creeping flow). The liquid velocity induced by
    neighbours, dipole (translational) near fields and the history force are not
    represented.
    """

    liquid_viscosity: Array = parameter_field()
    drag_coefficient: Array = parameter_field()
    added_mass_coefficient: Array = parameter_field()
    translation_id: str = eqx.field(static=True)

    def __init__(
        self,
        liquid_viscosity: ArrayLike,
        /,
        *,
        drag_coefficient: ArrayLike = 12.0,
        added_mass_coefficient: ArrayLike = 0.5,
    ) -> None:
        viscosity = scalar_parameter(
            liquid_viscosity, "liquid_viscosity", lower=0.0, inclusive=True
        )
        drag = scalar_parameter(
            drag_coefficient, "drag_coefficient", lower=0.0, inclusive=True
        )
        added = scalar_parameter(
            added_mass_coefficient, "added_mass_coefficient", lower=0.0
        )
        self.liquid_viscosity = viscosity
        self.drag_coefficient = drag
        self.added_mass_coefficient = added
        self.translation_id = canonical_fingerprint(
            {"kind": "bubble-translation-added-mass"}
        )


class BubbleCloudPressureField(StrictModule):
    """Spatial shape `s(x)` of the drive: the local excess pressure is `s(x) p_d(t)`.

    `"uniform"`: `s = 1` (no primary Bjerknes force). `"standing_wave"`:
    `s = cos(k·x + φ)` with wave vector `k` (rad/m); pressure antinodes sit at
    `k·x + φ = nπ`. The convective rate `(∇s·v) p_d` seen by a translating
    bubble is neglected in the compressible radiation terms (`|v| ≪ c`).
    """

    kind: BubbleCloudPressureFieldKind = eqx.field(static=True)
    wave_vector: Array = parameter_field()
    phase: Array = parameter_field()
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: BubbleCloudPressureFieldKind = "uniform",
        /,
        *,
        wave_vector: ArrayLike | None = None,
        phase: ArrayLike = 0.0,
    ) -> None:
        selected = parse(kind, BubbleCloudPressureFieldKind, "kind")
        vector = (
            np.zeros((3,)) if wave_vector is None else np.asarray(wave_vector, np.float64)
        )
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("wave_vector must be a finite array of shape (3,).")
        match selected:
            case "uniform":
                if np.any(vector != 0.0):
                    raise ValueError("A uniform pressure field has no wave vector.")
            case "standing_wave":
                if not np.any(vector != 0.0):
                    raise ValueError("A standing wave requires a nonzero wave vector.")
            case _:
                assert_never(selected)
        self.kind = selected
        self.wave_vector = jnp.asarray(vector, dtype=jnp.float64)
        self.phase = scalar_parameter(phase, "phase")
        self.field_id = canonical_fingerprint(
            {"kind": "bubble-cloud-pressure-field", "shape": selected}
        )

    def shape(self, position: Array, /) -> tuple[Array, Array]:
        """Shape `s(x)` and its gradient `∇s(x)` at every bubble position."""
        match self.kind:
            case "uniform":
                return jnp.ones(position.shape[:1]), jnp.zeros_like(position)
            case "standing_wave":
                argument = position @ self.wave_vector + self.phase
                gradient = -jnp.sin(argument)[:, None] * self.wave_vector[None, :]
                return jnp.cos(argument), gradient
            case _:
                assert_never(self.kind)


class BubbleCloudResourcePolicy(StrictModule, NonTrainableState):
    """Resource bounds, admissibility thresholds and matrix-free controls of a cloud.

    Dense pair geometry is permitted only when `N² ≤ maximum_dense_entries` and
    the Cholesky work `N³/3 ≤ maximum_dense_flops`; `route="auto"` selects the
    dense route inside that bound and the FMM route outside it. A pair overlaps
    when `d_ij ≤ minimum_contact_ratio (R_i + R_j)`. The FMM route certifies
    overlap only over candidate pairs within
    `minimum_contact_ratio · overlap_growth_bound · 2 R_max`
    (`maximum_candidate_pairs` capacity) and stops with `VALIDITY_EXCEEDED` once
    a bubble outgrows that certificate. `maximum_condition_number` bounds the
    2-norm condition number of the scaled coupling `I + W` (dense route).
    `maximum_delay_pairs` bounds the retarded route (its dense delay history is
    bounded by the plan's `maximum_steps`).
    """

    maximum_dense_entries: int = eqx.field(static=True)
    maximum_dense_flops: int = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    minimum_contact_ratio: float = eqx.field(static=True)
    overlap_growth_bound: float = eqx.field(static=True)
    maximum_candidate_pairs: int = eqx.field(static=True)
    coupling_tolerance: float = eqx.field(static=True)
    maximum_coupling_iterations: int = eqx.field(static=True)
    fmm_order: int = eqx.field(static=True)
    fmm_opening_angle: float = eqx.field(static=True)
    fmm_depth: int = eqx.field(static=True)
    fmm_leaf_occupancy: int = eqx.field(static=True)
    maximum_delay_pairs: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_dense_entries: int = 4096,
        maximum_dense_flops: int = 1 << 24,
        maximum_condition_number: float = 1.0e8,
        minimum_contact_ratio: float = 1.0,
        overlap_growth_bound: float = 4.0,
        maximum_candidate_pairs: int = 1 << 18,
        coupling_tolerance: float = 1.0e-11,
        maximum_coupling_iterations: int = 200,
        fmm_order: int = 5,
        fmm_opening_angle: float = 0.5,
        fmm_depth: int = 6,
        fmm_leaf_occupancy: int = 16,
        maximum_delay_pairs: int = 496,
    ) -> None:
        entries = positive_integer(maximum_dense_entries, "maximum_dense_entries")
        flops = positive_integer(maximum_dense_flops, "maximum_dense_flops")
        condition = positive_finite_float(
            maximum_condition_number, "maximum_condition_number"
        )
        if condition <= 1.0:
            raise ValueError("maximum_condition_number must exceed 1.")
        contact = positive_finite_float(minimum_contact_ratio, "minimum_contact_ratio")
        growth = positive_finite_float(overlap_growth_bound, "overlap_growth_bound")
        if growth <= 1.0:
            raise ValueError("overlap_growth_bound must exceed 1.")
        candidates = positive_integer(maximum_candidate_pairs, "maximum_candidate_pairs")
        tolerance = positive_finite_float(coupling_tolerance, "coupling_tolerance")
        iterations = positive_integer(
            maximum_coupling_iterations, "maximum_coupling_iterations"
        )
        order = positive_integer(fmm_order, "fmm_order")
        angle = positive_finite_float(fmm_opening_angle, "fmm_opening_angle")
        if not angle < 1.0:
            raise ValueError("fmm_opening_angle must lie in (0, 1).")
        depth = positive_integer(fmm_depth, "fmm_depth")
        occupancy = positive_integer(fmm_leaf_occupancy, "fmm_leaf_occupancy")
        delays = positive_integer(maximum_delay_pairs, "maximum_delay_pairs")
        self.maximum_dense_entries = entries
        self.maximum_dense_flops = flops
        self.maximum_condition_number = condition
        self.minimum_contact_ratio = contact
        self.overlap_growth_bound = growth
        self.maximum_candidate_pairs = candidates
        self.coupling_tolerance = tolerance
        self.maximum_coupling_iterations = iterations
        self.fmm_order = order
        self.fmm_opening_angle = angle
        self.fmm_depth = depth
        self.fmm_leaf_occupancy = occupancy
        self.maximum_delay_pairs = delays
        self.policy_id = canonical_fingerprint(
            {
                "kind": "bubble-cloud-resource-policy",
                "maximum_dense_entries": entries,
                "maximum_dense_flops": flops,
                "maximum_condition_number": condition,
                "minimum_contact_ratio": contact,
                "overlap_growth_bound": growth,
                "maximum_candidate_pairs": candidates,
                "coupling_tolerance": tolerance,
                "maximum_coupling_iterations": iterations,
                "fmm_order": order,
                "fmm_opening_angle": angle,
                "fmm_depth": depth,
                "fmm_leaf_occupancy": occupancy,
                "maximum_delay_pairs": delays,
            }
        )

    def dense_admissible(self, count: int, /) -> bool:
        """Whether dense pair geometry for `count` bubbles lies inside the bound."""
        return count * count <= self.maximum_dense_entries and (
            count**3 // 3 <= self.maximum_dense_flops
        )


class BubbleCloudState(StrictModule):
    """Per-group batched radial states and translational position/velocity."""

    groups: tuple[BubbleState, ...]
    position: Array
    velocity: Array


class BubbleCloudRates(StrictModule):
    """Coupled time derivative and forces of a cloud at one state.

    Per-bubble arrays follow the cloud order (groups in plan order, members in
    group order). `neighbor_pressure` is the incompressible (or retarded)
    neighbour monopole pressure `ρ Σ_j Q̇_j/d_ij` at each bubble;
    `uncoupled_acceleration` is the isolated-bubble `R̈` under the local drive.
    """

    derivative: BubbleCloudState
    acceleration: Array
    uncoupled_acceleration: Array
    inertia: Array
    neighbor_pressure: Array
    neighbor_pressure_gradient: Array
    primary_bjerknes: Array
    secondary_bjerknes: Array
    group_rates: tuple[RadialBubbleRates, ...]
    coupling_successful: Array
    coupling_iterations: Array
    coupling_residual: Array
    wall_work_rate: Array
    dissipation_rate: Array
    gas_heat_rate: Array


class _LocalPressureDrive(AbstractBubblePressureDrive):
    """Drive seen at one bubble: `s(x_b) p_d(t)` for the prepared field shape."""

    base: AbstractBubblePressureDrive
    scale: Array
    drive_id: str = eqx.field(static=True)

    def __init__(self, base: AbstractBubblePressureDrive, scale: Array, /) -> None:
        self.base = base
        self.scale = scale
        self.drive_id = base.drive_id

    def evaluate(self, time: Array, /) -> PressureDriveEvaluation:
        value = self.base.evaluate(time)
        return PressureDriveEvaluation(
            self.scale * value.pressure,
            self.scale * value.pressure_rate,
            value.in_support,
        )

    def characteristic_pressure(self) -> Array:
        return jnp.abs(self.scale) * self.base.characteristic_pressure()


def _validated_save_times(save_times: ArrayLike, /) -> np.ndarray:
    times = np.asarray(save_times, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] == 0:
        raise ValueError("save_times must be a non-empty rank-1 array.")
    if not np.all(np.isfinite(times)) or times[0] < 0.0 or times[-1] <= 0.0:
        raise ValueError("save_times must be finite, nonnegative and end after t = 0.")
    if times.shape[0] > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("save_times must be strictly increasing.")
    return times


def _medium(groups: tuple[BubbleSpeciesGroup, ...], /) -> None:
    """Refuse clouds whose species disagree on the shared liquid density or sound speed."""
    density = np.concatenate(
        [
            np.asarray(eqx.filter_vmap(lambda m: m.far_field_density())(g.model))
            for g in groups
        ]
    )
    speed = np.concatenate(
        [
            np.asarray(eqx.filter_vmap(lambda m: m.far_field_sound_speed())(g.model))
            for g in groups
        ]
    )
    if not (
        np.allclose(density, density[0], rtol=1.0e-12)
        and np.allclose(speed, speed[0], rtol=1.0e-12)
    ):
        raise ValueError(
            "Every bubble of a cloud must share one liquid density and sound speed."
        )


def _resolve_route(
    route: BubbleCloudRoute, count: int, resources: BubbleCloudResourcePolicy, /
) -> ResolvedBubbleCloudRoute:
    admissible = resources.dense_admissible(count)
    match route:
        case "auto":
            return "dense" if admissible else "fmm"
        case "dense":
            if not admissible:
                raise ValueError(
                    f"Dense pair geometry for {count} bubbles exceeds the resource policy "
                    f"(N² = {count * count} > {resources.maximum_dense_entries} entries or "
                    f"N³/3 = {count**3 // 3} > {resources.maximum_dense_flops} flops)."
                )
            return "dense"
        case "fmm":
            return "fmm"
        case _:
            assert_never(route)


def _resolve_integrator(
    integrator: SingleBubbleIntegrator, groups: tuple[BubbleSpeciesGroup, ...], /
) -> CloudIntegrator:
    match integrator:
        case "auto":
            stiff = any(group.model.requires_stiff_integration for group in groups)
            return "stiff" if stiff else "explicit"
        case "explicit":
            return "explicit"
        case "stiff":
            return "stiff"
        case _:
            assert_never(integrator)


class BubbleCloudPlan(StrictModule):
    """Static structure, route, physics options and tolerances of one cloud solve.

    `route` selects the coupled-acceleration solve (`"auto"` follows the dense
    resource bound). `coupling="retarded"` uses retarded neighbour sources and
    requires fixed positions and the dense route. `translation=None` keeps the
    bubbles at fixed positions; translation requires the dense route. `emission`
    evaluates the far-field monopole pressure (fixed positions only). Every
    bubble shares one liquid (density and sound speed).
    """

    groups: tuple[BubbleSpeciesGroup, ...]
    drive: AbstractBubblePressureDrive
    pressure_field: BubbleCloudPressureField
    translation: BubbleTranslation | None
    emission: FarFieldEmissionPlan | None
    resources: BubbleCloudResourcePolicy
    events: BubbleEventPolicy
    validity: BubbleValidityPolicy
    save_times: Array = fixed_field()
    route: ResolvedBubbleCloudRoute = eqx.field(static=True)
    coupling: BubbleCloudCoupling = eqx.field(static=True)
    integrator: CloudIntegrator = eqx.field(static=True)
    differentiation: BubbleDifferentiation = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    bubble_count: int = eqx.field(static=True)
    bubble_ids: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        groups: Sequence[BubbleSpeciesGroup],
        drive: AbstractBubblePressureDrive,
        save_times: ArrayLike,
        /,
        *,
        route: BubbleCloudRoute = "auto",
        coupling: BubbleCloudCoupling = "incompressible",
        pressure_field: BubbleCloudPressureField | None = None,
        translation: BubbleTranslation | None = None,
        emission: FarFieldEmissionPlan | None = None,
        resources: BubbleCloudResourcePolicy | None = None,
        events: BubbleEventPolicy | None = None,
        validity: BubbleValidityPolicy | None = None,
        integrator: SingleBubbleIntegrator = "auto",
        differentiation: BubbleDifferentiation = "reverse",
        relative_tolerance: float = 1.0e-8,
        absolute_tolerance: float = 1.0e-10,
        maximum_steps: int = 16384,
    ) -> None:
        members = tuple(groups)
        if not members or not all(
            isinstance(group, BubbleSpeciesGroup) for group in members
        ):
            raise TypeError("groups must be a non-empty sequence of BubbleSpeciesGroup.")
        times = _validated_save_times(save_times)
        field = BubbleCloudPressureField() if pressure_field is None else pressure_field
        if not isinstance(field, BubbleCloudPressureField):
            raise TypeError("pressure_field must be a BubbleCloudPressureField or None.")
        if translation is not None and not isinstance(translation, BubbleTranslation):
            raise TypeError("translation must be a BubbleTranslation or None.")
        if emission is not None and not isinstance(emission, FarFieldEmissionPlan):
            raise TypeError("emission must be a FarFieldEmissionPlan or None.")
        policy = BubbleCloudResourcePolicy() if resources is None else resources
        if not isinstance(policy, BubbleCloudResourcePolicy):
            raise TypeError("resources must be a BubbleCloudResourcePolicy or None.")
        event_policy = BubbleEventPolicy() if events is None else events
        if not isinstance(event_policy, BubbleEventPolicy):
            raise TypeError("events must be a BubbleEventPolicy or None.")
        support = BubbleValidityPolicy() if validity is None else validity
        if not isinstance(support, BubbleValidityPolicy):
            raise TypeError("validity must be a BubbleValidityPolicy or None.")
        ids = tuple(bubble for group in members for bubble in group.bubble_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("bubble_ids must be unique across the cloud.")
        count = len(ids)
        selected_route = _resolve_route(
            parse(route, BubbleCloudRoute, "route"), count, policy
        )
        selected_coupling = parse(coupling, BubbleCloudCoupling, "coupling")
        _validate_options(
            selected_route, selected_coupling, translation, emission, count, policy
        )
        if selected_coupling == "retarded":
            _retarded_start(members, drive)
        _medium(members)
        rtol = positive_finite_float(relative_tolerance, "relative_tolerance")
        atol = positive_finite_float(absolute_tolerance, "absolute_tolerance")
        steps = positive_integer(maximum_steps, "maximum_steps")
        resolved = _resolve_integrator(
            parse(integrator, SingleBubbleIntegrator, "integrator"), members
        )
        mode = parse(differentiation, BubbleDifferentiation, "differentiation")
        self.groups = members
        self.drive = drive
        self.pressure_field = field
        self.translation = translation
        self.emission = emission
        self.resources = policy
        self.events = event_policy
        self.validity = support
        self.save_times = jnp.asarray(times, dtype=jnp.float64)
        self.route = selected_route
        self.coupling = selected_coupling
        self.integrator = resolved
        self.differentiation = mode
        self.relative_tolerance = rtol
        self.absolute_tolerance = atol
        self.maximum_steps = steps
        self.bubble_count = count
        self.bubble_ids = ids
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bubble-cloud-plan",
                "groups": [group.group_id for group in members],
                "drive": drive.drive_id,
                "pressure_field": field.field_id,
                "translation": None
                if translation is None
                else translation.translation_id,
                "emission": None if emission is None else emission.plan_id,
                "resources": policy.policy_id,
                "events": event_policy.policy_id,
                "validity": support.policy_id,
                "save_times": times,
                "route": selected_route,
                "coupling": selected_coupling,
                "integrator": resolved,
                "differentiation": mode,
                "relative_tolerance": rtol,
                "absolute_tolerance": atol,
                "maximum_steps": steps,
            }
        )

    @property
    def translating(self) -> bool:
        """Whether bubble positions are part of the dynamic state."""
        return self.translation is not None

    def liquid_density(self) -> Array:
        """Density of the liquid shared by every bubble."""
        return eqx.filter_vmap(lambda m: m.far_field_density())(self.groups[0].model)[0]

    def sound_speed(self) -> Array:
        """Sound speed of the liquid shared by every bubble."""
        return eqx.filter_vmap(lambda m: m.far_field_sound_speed())(self.groups[0].model)[
            0
        ]

    def prepare(self) -> PreparedBubbleCloud:
        """Equilibria, scales, pair structure and coupling route of the cloud.

        Host boundary: the FMM box, softening and candidate pair capacity are
        derived from the concrete initial positions and radii.
        """
        equilibria, states, regimes, scales, state_scales = _group_preparation(self)
        position = jnp.concatenate([group.position for group in self.groups])
        radius = jnp.concatenate([group.equilibrium_radius for group in self.groups])
        coupling, pairs, growth_limit, candidates = _coupling_preparation(
            self, position, radius
        )
        velocity = jnp.concatenate([group.initial_velocity for group in self.groups])
        initial = BubbleCloudState(states, position, velocity)
        time_scale = jnp.min(jnp.concatenate([scale.time for scale in scales]))
        velocity_scale = jnp.max(jnp.concatenate([scale.velocity for scale in scales]))
        energy_scale = jnp.max(jnp.concatenate([scale.energy for scale in scales]))
        return PreparedBubbleCloud(
            self,
            equilibria,
            initial,
            regimes,
            scales,
            state_scales,
            radius,
            time_scale,
            jnp.max(radius),
            velocity_scale,
            energy_scale,
            coupling,
            pairs,
            growth_limit,
            candidates,
        )


def _validate_options(
    route: ResolvedBubbleCloudRoute,
    coupling: BubbleCloudCoupling,
    translation: BubbleTranslation | None,
    emission: FarFieldEmissionPlan | None,
    count: int,
    resources: BubbleCloudResourcePolicy,
    /,
) -> None:
    if translation is not None and route != "dense":
        raise ValueError("Translating bubbles require the dense route.")
    if emission is not None and translation is not None:
        raise ValueError("Far-field emission requires fixed bubble positions.")
    match coupling:
        case "incompressible":
            return
        case "retarded":
            if route != "dense" or translation is not None:
                raise ValueError(
                    "Retarded coupling requires the dense route and fixed positions."
                )
            pairs = count * (count - 1) // 2
            if pairs == 0:
                raise ValueError("Retarded coupling requires at least two bubbles.")
            if pairs > resources.maximum_delay_pairs:
                raise ValueError(
                    f"Retarded coupling of {count} bubbles needs {pairs} delay pairs "
                    f"(> maximum_delay_pairs = {resources.maximum_delay_pairs})."
                )
        case _:
            assert_never(coupling)


def _retarded_start(
    groups: tuple[BubbleSpeciesGroup, ...], drive: AbstractBubblePressureDrive, /
) -> None:
    """Require the derivative-compatible start of the retarded (neutral) route.

    With bubbles at rest in equilibrium and `p_d(0) = 0` the constant prehistory
    has the same derivative as the solution at `t = 0⁺`, so no neutral
    derivative discontinuity has to be propagated along pair-lag sums.
    """
    at_rest = all(
        np.array_equal(
            np.asarray(group.initial_radius), np.asarray(group.equilibrium_radius)
        )
        and not np.any(np.asarray(group.initial_wall_velocity))
        for group in groups
    )
    quiet = float(drive.evaluate(jnp.zeros(())).pressure) == 0.0
    if not (at_rest and quiet):
        raise ValueError(
            "Retarded coupling requires bubbles at rest in equilibrium and p_d(0) = 0."
        )


def _group_preparation(
    plan: BubbleCloudPlan, /
) -> tuple[
    tuple[BubbleEquilibrium, ...],
    tuple[BubbleState, ...],
    tuple[Array, ...],
    tuple[BubbleScales, ...],
    tuple[BubbleState, ...],
]:
    characteristic = plan.drive.characteristic_pressure()
    equilibria = []
    states = []
    regimes = []
    scales = []
    state_scales = []
    for group in plan.groups:
        equilibrium = eqx.filter_vmap(lambda m, r: m.equilibrium(r))(
            group.model, group.equilibrium_radius
        )
        state = BubbleState(
            group.initial_radius,
            group.initial_wall_velocity,
            equilibrium.state.gas,
            equilibrium.state.liquid,
            equilibrium.state.interface,
        )
        regime = eqx.filter_vmap(lambda m, r, z: m.interface.initial_regime(r, z))(
            group.model, group.initial_radius, equilibrium.state.interface
        ).astype(jnp.int32)
        scale = eqx.filter_vmap(
            lambda m, e, v: m.characteristic_scales(e, characteristic, v)
        )(group.model, equilibrium, group.initial_wall_velocity)
        state_scale = eqx.filter_vmap(lambda m, g, s: m.state_scale(g, s))(
            group.model, equilibrium.state.gas, scale
        )
        equilibria.append(equilibrium)
        states.append(state)
        regimes.append(regime)
        scales.append(scale)
        state_scales.append(state_scale)
    return (
        tuple(equilibria),
        tuple(states),
        tuple(regimes),
        tuple(scales),
        tuple(state_scales),
    )


class _CandidatePairs(StrictModule):
    """FMM overlap-certificate candidate relation evidence."""

    required: Array
    capacity: int = eqx.field(static=True)
    successful: Array


def _coupling_preparation(
    plan: BubbleCloudPlan, position: Array, radius: Array, /
) -> tuple[
    AbstractCloudCoupling,
    tuple[Array, Array, Array],
    Array | None,
    _CandidatePairs | None,
]:
    count = plan.bubble_count
    match plan.route:
        case "dense":
            first, second = np.triu_indices(count, k=1)
            pairs = (
                jnp.asarray(first, dtype=jnp.int32),
                jnp.asarray(second, dtype=jnp.int32),
                jnp.ones((first.shape[0],), dtype=jnp.bool_),
            )
            kernel = None if plan.translating else dense_inverse_distance(position)
            return DenseCloudCoupling(kernel), pairs, None, None
        case "fmm":
            return _fmm_preparation(plan, position, radius)
        case _:
            assert_never(plan.route)


def _fmm_preparation(
    plan: BubbleCloudPlan, position: Array, radius: Array, /
) -> tuple[FMMCloudCoupling, tuple[Array, Array, Array], Array, _CandidatePairs]:
    resources = plan.resources
    points = np.asarray(position)
    largest = float(np.max(np.asarray(radius)))
    certificate = (
        resources.minimum_contact_ratio
        * resources.overlap_growth_bound
        * 2.0
        * largest
    )
    extent = float(np.max(np.ptp(points, axis=0)))
    margin = max(extent, largest) * 0.03125 + 2.0 * largest
    lower = np.min(points, axis=0) - margin
    side = extent + 2.0 * margin
    box = (side, side, side)
    fmm = UniformFMMPlan(
        1.0,
        CartesianExpansionSpace(resources.fmm_order),
        softening=1.0e-9 * side,
        opening_angle=resources.fmm_opening_angle,
        maximum_leaf_occupancy=resources.fmm_leaf_occupancy,
    )
    local = position - jnp.asarray(lower, dtype=position.dtype)
    structure = fmm.prepare_structure(local, box_size=box, depth=resources.fmm_depth)
    coupling = FMMCloudCoupling(
        fmm,
        structure,
        tolerance=resources.coupling_tolerance,
        maximum_iterations=resources.maximum_coupling_iterations,
    )
    address = MortonAddressPlan(tuple(lower), tuple(lower + side), resources.fmm_depth)
    relation_plan = MortonRadiusRelationPlan(
        address, plan.bubble_count, plan.bubble_count, resources.maximum_candidate_pairs
    )
    query = relation_plan.query(
        position, position, certificate, exclude_self=True, pair_once=True
    )
    relation = query.relation
    pairs = (
        relation.source_indices.astype(jnp.int32),
        relation.target_indices.astype(jnp.int32),
        relation.valid,
    )
    candidates = _CandidatePairs(
        query.evidence.required_pairs,
        capacity=resources.maximum_candidate_pairs,
        successful=query.evidence.successful,
    )
    # Below this guard, every possible contact distance is at most `certificate`.
    growth_limit = resources.overlap_growth_bound * jnp.max(radius)
    return coupling, pairs, growth_limit, candidates


class PreparedBubbleCloud(StrictModule):
    """Equilibria, initial state, scales and the prepared coupling route.

    `pair_first`/`pair_second`/`pair_valid` are the overlap-certificate pairs
    (every pair on the dense route, the candidate relation on the FMM route).
    """

    plan: BubbleCloudPlan
    equilibria: tuple[BubbleEquilibrium, ...]
    initial_state: BubbleCloudState
    regimes: tuple[Array, ...]
    scales: tuple[BubbleScales, ...]
    state_scales: tuple[BubbleState, ...]
    equilibrium_radius: Array
    time_scale: Array
    length_scale: Array
    velocity_scale: Array
    energy_scale: Array
    coupling: AbstractCloudCoupling
    pair_first: Array
    pair_second: Array
    pair_valid: Array
    growth_limit: Array | None
    candidates: _CandidatePairs | None

    def __init__(
        self,
        plan: BubbleCloudPlan,
        equilibria: tuple[BubbleEquilibrium, ...],
        initial_state: BubbleCloudState,
        regimes: tuple[Array, ...],
        scales: tuple[BubbleScales, ...],
        state_scales: tuple[BubbleState, ...],
        equilibrium_radius: Array,
        time_scale: Array,
        length_scale: Array,
        velocity_scale: Array,
        energy_scale: Array,
        coupling: AbstractCloudCoupling,
        pairs: tuple[Array, Array, Array],
        growth_limit: Array | None,
        candidates: _CandidatePairs | None,
        /,
    ) -> None:
        self.plan = plan
        self.equilibria = equilibria
        self.initial_state = initial_state
        self.regimes = regimes
        self.scales = scales
        self.state_scales = state_scales
        self.equilibrium_radius = equilibrium_radius
        self.time_scale = time_scale
        self.length_scale = length_scale
        self.velocity_scale = velocity_scale
        self.energy_scale = energy_scale
        self.coupling = coupling
        self.pair_first, self.pair_second, self.pair_valid = pairs
        self.growth_limit = growth_limit
        self.candidates = candidates

    def local_terms(self, state: BubbleCloudState, time: Array, /) -> CloudLocalTerms:
        """Uncoupled per-bubble radial evaluation under the local drive."""
        plan = self.plan
        shape, gradient = plan.pressure_field.shape(state.position)
        group_rates = []
        start = 0
        for group, member_state, regime in zip(
            plan.groups, state.groups, self.regimes, strict=True
        ):
            local_scale = shape[start : start + group.size]
            start += group.size

            def one(
                model: RadialBubbleModel,
                member: BubbleState,
                member_regime: Array,
                scale: Array,
            ) -> RadialBubbleRates:
                return model.rates(
                    member, member_regime, time, _LocalPressureDrive(plan.drive, scale)
                )

            group_rates.append(
                eqx.filter_vmap(one)(group.model, member_state, regime, local_scale)
            )
        radius = jnp.concatenate([member.radius for member in state.groups])
        velocity = jnp.concatenate([member.wall_velocity for member in state.groups])
        fraction = jnp.concatenate([rates.inertia_fraction for rates in group_rates])
        uncoupled = jnp.concatenate([rates.acceleration for rates in group_rates])
        inertia = fraction * radius
        return CloudLocalTerms(
            tuple(group_rates),
            radius,
            velocity,
            inertia,
            inertia * uncoupled,
            uncoupled,
            gradient,
        )

    def _assemble(
        self,
        state: BubbleCloudState,
        time: Array,
        terms: CloudLocalTerms,
        acceleration: Array,
        field: CloudField,
        coupling_successful: Array,
        coupling_iterations: Array,
        coupling_residual: Array,
        /,
    ) -> BubbleCloudRates:
        plan = self.plan
        density = plan.liquid_density()
        radius = terms.radius
        volume = sphere_volume(radius)
        drive = plan.drive.evaluate(time)
        primary = -volume[:, None] * drive.pressure * terms.drive_gradient
        secondary = -density * volume[:, None] * field.gradient
        groups = []
        start = 0
        for group, member, rates in zip(
            plan.groups, state.groups, terms.group_rates, strict=True
        ):
            local = acceleration[start : start + group.size]
            start += group.size
            groups.append(
                BubbleState(
                    member.wall_velocity,
                    local,
                    rates.derivative.gas,
                    rates.derivative.liquid,
                    rates.derivative.interface,
                )
            )
        translation = plan.translation
        if translation is None:
            position_rate = jnp.zeros_like(state.position)
            velocity_rate = jnp.zeros_like(state.velocity)
        else:
            volume_rate = 4.0 * jnp.pi * radius**2 * terms.velocity
            added = translation.added_mass_coefficient * density
            drag = (
                -(
                    translation.drag_coefficient
                    * jnp.pi
                    * translation.liquid_viscosity
                    * radius
                )[:, None]
                * state.velocity
            )
            force = (
                primary
                + secondary
                + drag
                - (added * volume_rate)[:, None] * state.velocity
            )
            position_rate = state.velocity
            velocity_rate = force / (added * volume)[:, None]
        rates = terms.group_rates
        return BubbleCloudRates(
            BubbleCloudState(tuple(groups), position_rate, velocity_rate),
            acceleration,
            terms.uncoupled,
            terms.inertia,
            density * field.potential,
            density * field.gradient,
            primary,
            secondary,
            rates,
            coupling_successful,
            coupling_iterations,
            coupling_residual,
            jnp.sum(jnp.concatenate([rate.wall_work_rate for rate in rates])),
            jnp.sum(jnp.concatenate([rate.dissipation_rate for rate in rates])),
            jnp.sum(jnp.concatenate([rate.gas_heat_rate for rate in rates])),
        )

    def rates(self, state: BubbleCloudState, time: ArrayLike, /) -> BubbleCloudRates:
        """Coupled incompressible rates at `state` and absolute `time` (s)."""
        current = jnp.asarray(time, dtype=jnp.float64)
        terms = self.local_terms(state, current)
        solved = self.coupling.solve(
            state.position, terms.inertia, terms.forcing, terms.radius, terms.velocity
        )
        return self._assemble(
            state,
            current,
            terms,
            solved.acceleration,
            CloudField(solved.potential, solved.gradient, solved.successful),
            solved.successful,
            solved.iterations,
            solved.relative_residual,
        )

    def retarded_rates(
        self,
        state: BubbleCloudState,
        time: ArrayLike,
        potential: Array,
        gradient: Array,
        /,
    ) -> BubbleCloudRates:
        """Rates under a given retarded neighbour source `Σ_j Q̇_j(t − τ_ij)/d_ij`."""
        current = jnp.asarray(time, dtype=jnp.float64)
        terms = self.local_terms(state, current)
        acceleration = (terms.forcing - potential) / terms.inertia
        finite = jnp.all(jnp.isfinite(potential)) & jnp.all(jnp.isfinite(gradient))
        return self._assemble(
            state,
            current,
            terms,
            acceleration,
            CloudField(potential, gradient, finite),
            finite,
            jnp.zeros((), dtype=jnp.int32),
            jnp.zeros(()),
        )

    def contact_ratio(self, state: BubbleCloudState, /) -> Array:
        """Minimum `d_ij/(R_i + R_j)` over the overlap-certificate pairs."""
        radius = cloud_radius(state)
        first = self.pair_first
        second = self.pair_second
        distance = jnp.sqrt(
            jnp.sum((state.position[first] - state.position[second]) ** 2, axis=-1)
        )
        ratio = distance / (radius[first] + radius[second])
        return jnp.min(jnp.where(self.pair_valid, ratio, jnp.inf), initial=jnp.inf)

    def coupling_spectrum(
        self, state: BubbleCloudState, time: ArrayLike, /
    ) -> tuple[Array, Array, Array]:
        """Condition number of `I + W`, its definiteness and `ρ(W)` (dense route)."""
        current = jnp.asarray(time, dtype=jnp.float64)
        terms = self.local_terms(state, current)
        coupling = self.coupling
        if not isinstance(coupling, DenseCloudCoupling):
            raise ValueError(
                "The coupling spectrum is evaluated on the dense route only."
            )
        return coupling_spectrum(
            coupling.inverse_distance(state.position), terms.inertia, terms.radius
        )


class CloudLocalTerms(StrictModule):
    """Per-bubble inertia `a`, forcing `f = a R̈_isolated` and local-drive gradient."""

    group_rates: tuple[RadialBubbleRates, ...]
    radius: Array
    velocity: Array
    inertia: Array
    forcing: Array
    uncoupled: Array
    drive_gradient: Array


def cloud_radius(state: BubbleCloudState, /) -> Array:
    """Radii of every bubble in cloud order."""
    return jnp.concatenate([member.radius for member in state.groups])


class BubbleCloudTrajectory(StrictModule):
    """Requested saved trajectory; rows at unreached save times are NaN.

    Arrays are `(time, bubble)` or `(time, bubble, 3)` in cloud order.
    """

    times: Array
    radius: Array
    wall_velocity: Array
    acceleration: Array
    position: Array
    velocity: Array
    gas_pressure: Array
    wall_pressure: Array
    far_field_pressure: Array
    neighbor_pressure: Array
    primary_bjerknes: Array
    secondary_bjerknes: Array
    states: BubbleCloudState
    valid: Array


class BubbleCloudEvidence(StrictModule):
    """Solver work, events, energy ledger, coupling, overlap and resource evidence.

    `work_residual = ΔK − W` compares the change of the incompressible liquid
    kinetic energy `K = 2πρ Σ_i q_i (R_i Ṙ_i + Σ_{j≠i} q_j/d_ij)` (`q = R²Ṙ`)
    with the integrated wall work; it is an exact identity of the coupled
    Rayleigh–Plesset cloud at fixed positions (`work_identity_exact`).
    `maximum_condition_number` and `maximum_spectral_radius` refer to the
    scaled coupling `I + W` and `W` over the saved rows (NaN on the FMM route).
    `retardation_ratio` is the largest pair delay over the fastest inertial
    time scale and `history_occupancy` the peak number of accepted-step
    interpolants held by the dense delay history (both zero for incompressible
    coupling). `candidate_pairs_*` describe the FMM overlap certificate.
    """

    solver_successful: Array
    accepted_steps: Array
    rejected_steps: Array
    event_kind: Array
    event_time: Array
    wall_work: Array
    kinetic_energy_change: Array
    work_residual: Array
    dissipated_energy: Array
    gas_heat: Array
    minimum_contact_ratio: Array
    maximum_condition_number: Array
    maximum_spectral_radius: Array
    coupling_successful: Array
    maximum_coupling_iterations: Array
    maximum_coupling_residual: Array
    retardation_ratio: Array
    history_occupancy: Array
    candidate_pairs_required: Array
    candidate_pairs_successful: Array
    fmm: CartesianFMMResourceEvidence | None
    validity: BubbleValidityEvidence
    route: ResolvedBubbleCloudRoute = eqx.field(static=True)
    coupling: BubbleCloudCoupling = eqx.field(static=True)
    integrator: CloudIntegrator = eqx.field(static=True)
    work_identity_exact: bool = eqx.field(static=True)


class BubbleCloudResult(StrictModule):
    """Trajectory, exact terminal state, status, evidence and optional emission."""

    trajectory: BubbleCloudTrajectory
    terminal_state: BubbleCloudState
    terminal_time: Array
    status: Array
    evidence: BubbleCloudEvidence
    emission: FarFieldEmissionResult | None
    plan_id: str = eqx.field(static=True)
    bubble_ids: tuple[int, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether the solve reached a physical endpoint without refusal."""
        return bubble_status_successful(self.status) & self.evidence.solver_successful

    @property
    def completed(self) -> Array:
        """Whether the solve reached the final save time."""
        return self.status == int(BubbleDynamicsStatus.SUCCESS)


__all__ = [
    "BubbleCloudCoupling",
    "BubbleCloudEvidence",
    "BubbleCloudPlan",
    "BubbleCloudPressureField",
    "BubbleCloudPressureFieldKind",
    "BubbleCloudRates",
    "BubbleCloudResourcePolicy",
    "BubbleCloudResult",
    "BubbleCloudRoute",
    "BubbleCloudState",
    "BubbleCloudTrajectory",
    "BubbleSpeciesGroup",
    "BubbleTranslation",
    "CloudIntegrator",
    "CloudLocalTerms",
    "PreparedBubbleCloud",
    "ResolvedBubbleCloudRoute",
    "cloud_radius",
]
