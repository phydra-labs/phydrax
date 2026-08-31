#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from types import FunctionType
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import (
    array_tree_fingerprint,
    array_tree_signature,
    canonical_fingerprint,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CochainDiscretization,
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
    StructuredCochainBridge,
)
from ..linalg import (
    DifferentiationPolicy,
    FailurePolicy,
    GeneralizedLSMR,
    LinearSolvePolicy,
    MinimumNormProblem,
    prepare,
    RankPolicy,
    solve as solve_linear,
    SolveResourcePolicy,
    TolerancePolicy,
)
from ..topology import CellSubcomplex
from ._harmonic_constraints import HarmonicConstraint
from ._maxwell_boundaries import MaxwellBoundaryPlan, PreparedMaxwellBoundary
from ._maxwell_observers import (
    AbstractMaxwellObserverPlan,
    AbstractPreparedMaxwellObserver,
)
from ._maxwell_pml import (
    MaxwellCPMLCoefficients,
    MaxwellCPMLPlan,
    MaxwellCPMLState,
    PreparedMaxwellCPML,
)
from ._maxwell_sources import (
    AbstractMaxwellSourcePlan,
    MaxwellSourceForcing,
    PreparedMaxwellSource,
)


MaxwellPolarization: TypeAlias = Literal["full_3d", "tez", "tmz"]


class MaxwellCochainLayout(StrictModule, NonTrainableState):
    """Static assignment of Maxwell field roles to one retained de Rham segment."""

    polarization: MaxwellPolarization = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    electric_degree: int = eqx.field(static=True)
    magnetic_degree: int = eqx.field(static=True)
    charge_degree: int | None = eqx.field(static=True)
    electric_count: int = eqx.field(static=True)
    magnetic_count: int = eqx.field(static=True)
    charge_count: int = eqx.field(static=True)
    electric_orientation_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    magnetic_orientation_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge | CochainDiscretization,
        polarization: MaxwellPolarization = "full_3d",
        /,
    ):
        if polarization not in ("full_3d", "tez", "tmz"):
            raise ValueError("Unknown Maxwell polarization.")
        if isinstance(bridge, StructuredCochainBridge):
            cochain = bridge.cochain
            dimension = bridge.dimension
            shapes = bridge.orientation_shapes
            source_id = bridge.bridge_id
        elif isinstance(bridge, CochainDiscretization):
            cochain = bridge
            dimension = cochain.max_degree
            shapes = tuple(((count,),) for count in cochain.cell_counts)
            source_id = cochain.prepared_id
        else:
            raise TypeError("Maxwell layout requires a cochain or structured bridge.")
        if polarization == "full_3d":
            if dimension != 3:
                raise ValueError("full_3d Maxwell requires a three-dimensional complex.")
            electric_degree, magnetic_degree, charge_degree = 1, 2, 0
        elif polarization == "tez":
            if dimension != 2:
                raise ValueError(
                    "TEz Maxwell requires a genuine two-dimensional complex."
                )
            electric_degree, magnetic_degree, charge_degree = 1, 2, 0
        else:
            if dimension != 2:
                raise ValueError(
                    "TMz Maxwell requires a genuine two-dimensional complex."
                )
            electric_degree, magnetic_degree, charge_degree = 0, 1, None
        self.polarization = polarization
        self.dimension = dimension
        self.electric_degree = electric_degree
        self.magnetic_degree = magnetic_degree
        self.charge_degree = charge_degree
        self.electric_count = cochain.cell_counts[electric_degree]
        self.magnetic_count = cochain.cell_counts[magnetic_degree]
        self.charge_count = (
            0 if charge_degree is None else cochain.cell_counts[charge_degree]
        )
        self.electric_orientation_shapes = shapes[electric_degree]
        self.magnetic_orientation_shapes = shapes[magnetic_degree]
        self.layout_id = canonical_fingerprint(
            {
                "kind": "maxwell-cochain-layout",
                "source": source_id,
                "polarization": polarization,
                "electric_degree": electric_degree,
                "magnetic_degree": magnetic_degree,
                "charge_degree": charge_degree,
                "electric_shapes": shapes[electric_degree],
                "magnetic_shapes": shapes[magnetic_degree],
            }
        )


class MaxwellResourcePolicy(StrictModule, NonTrainableState):
    """Hard x64 byte budgets checked before Maxwell preparation and execution."""

    maximum_state_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_acquisition_bytes: int = eqx.field(static=True)
    maximum_total_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_state_bytes: int = 2 * 1024**3,
        maximum_workspace_bytes: int = 2 * 1024**3,
        maximum_acquisition_bytes: int = 512 * 1024**2,
        maximum_total_bytes: int = 4 * 1024**3,
    ):
        values = tuple(
            int(value)
            for value in (
                maximum_state_bytes,
                maximum_workspace_bytes,
                maximum_acquisition_bytes,
                maximum_total_bytes,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("Maxwell resource budgets must be nonnegative.")
        (
            self.maximum_state_bytes,
            self.maximum_workspace_bytes,
            self.maximum_acquisition_bytes,
            self.maximum_total_bytes,
        ) = values


class MaxwellResourceEstimate(StrictModule, NonTrainableState):
    logical_primary_bytes: int = eqx.field(static=True)
    material_auxiliary_bytes: int = eqx.field(static=True)
    observer_state_bytes: int = eqx.field(static=True)
    cpml_state_bytes: int = eqx.field(static=True)
    projection_workspace_bytes: int = eqx.field(static=True)
    private_padding_bytes: int = eqx.field(static=True)
    case_state_bytes: int = eqx.field(static=True)
    per_device_state_bytes: int = eqx.field(static=True)
    returned_acquisition_bytes: int = eqx.field(static=True)
    total_bytes: int = eqx.field(static=True)


class MaxwellMagneticConstraintPolicy(StrictModule, NonTrainableState):
    mode: Literal["auto", "project", "elide"] = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    solve_policy: LinearSolvePolicy

    def __init__(
        self,
        mode: Literal["auto", "project", "elide"] = "auto",
        /,
        *,
        absolute_tolerance: float = 1e-11,
        relative_tolerance: float = 1e-10,
        solve_policy: LinearSolvePolicy | None = None,
    ):
        if mode not in ("auto", "project", "elide"):
            raise ValueError("Unknown magnetic-constraint policy mode.")
        absolute, relative = float(absolute_tolerance), float(relative_tolerance)
        if (
            not np.isfinite(absolute)
            or not np.isfinite(relative)
            or absolute < 0.0
            or relative < 0.0
        ):
            raise ValueError(
                "Magnetic projection tolerances must be finite and nonnegative."
            )
        policy = solve_policy or LinearSolvePolicy(
            GeneralizedLSMR(),
            tolerance=TolerancePolicy(relative=relative, absolute=absolute),
            rank=RankPolicy(require_full_rank=False),
            differentiation=DifferentiationPolicy("rhs-only"),
            failure=FailurePolicy("error"),
            resources=SolveResourcePolicy(),
        )
        self.mode, self.absolute_tolerance, self.relative_tolerance = (
            mode,
            absolute,
            relative,
        )
        self.solve_policy = policy


class MaxwellMagneticConstraintEvidence(StrictModule):
    residual_norm: Array
    relative_residual: Array
    period_residual: Array
    solver_status: Array
    iterations: Array
    projected: Array
    elided: bool = eqx.field(static=True)


class CompatibleMaxwellRunResult(StrictModule):
    final_state: CompatibleMaxwellState
    observations: tuple[Array, ...]
    diagnostics: CompatibleMaxwellDiagnostics
    status: Array
    step_count: Array
    resource_estimate: MaxwellResourceEstimate


class CompatibleMaxwellRefreshSpec(StrictModule):
    plan: CompatibleMaxwellPlan
    step_size: Array
    dtype: str = eqx.field(static=True)


class MaxwellCapabilities(StrictModule, NonTrainableState):
    """Fail-closed material, execution, and differentiation capabilities."""

    lossless: bool = eqx.field(static=True)
    passive: bool = eqx.field(static=True)
    active: bool = eqx.field(static=True)
    dispersive: bool = eqx.field(static=True)
    nonlinear: bool = eqx.field(static=True)
    reversible: bool = eqx.field(static=True)
    complex_required: bool = eqx.field(static=True)
    structured_only: bool = eqx.field(static=True)
    pml: bool = eqx.field(static=True)
    observers: bool = eqx.field(static=True)
    frequency_domain: bool = eqx.field(static=True)
    distributed: bool = eqx.field(static=True)
    magnetic_closedness_preserving: bool = eqx.field(static=True)
    linear_time_invariant: bool = eqx.field(static=True)
    local_tensors: bool = eqx.field(static=True)
    spatial_distribution: bool = eqx.field(static=True)
    ffi: bool = eqx.field(static=True)
    capability_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        lossless: bool,
        passive: bool,
        active: bool = False,
        dispersive: bool = False,
        nonlinear: bool = False,
        reversible: bool = False,
        complex_required: bool = False,
        structured_only: bool = True,
        pml: bool = False,
        observers: bool = False,
        frequency_domain: bool = False,
        distributed: bool = False,
        magnetic_closedness_preserving: bool = True,
        linear_time_invariant: bool = True,
        local_tensors: bool = False,
        spatial_distribution: bool = False,
        ffi: bool = False,
    ):
        if active and passive:
            raise ValueError("A Maxwell capability set cannot be passive and active.")
        if lossless and (active or not passive):
            raise ValueError(
                "Lossless Maxwell capabilities must be passive and inactive."
            )
        if reversible and (not lossless or dispersive or nonlinear or pml):
            raise ValueError(
                "Reversible Maxwell capabilities require lossless instantaneous dynamics."
            )
        values = {
            "lossless": bool(lossless),
            "passive": bool(passive),
            "active": bool(active),
            "dispersive": bool(dispersive),
            "nonlinear": bool(nonlinear),
            "reversible": bool(reversible),
            "complex_required": bool(complex_required),
            "structured_only": bool(structured_only),
            "pml": bool(pml),
            "observers": bool(observers),
            "frequency_domain": bool(frequency_domain),
            "distributed": bool(distributed),
            "magnetic_closedness_preserving": bool(magnetic_closedness_preserving),
            "linear_time_invariant": bool(linear_time_invariant),
            "local_tensors": bool(local_tensors),
            "spatial_distribution": bool(spatial_distribution),
            "ffi": bool(ffi),
        }
        for name, value in values.items():
            setattr(self, name, value)
        self.capability_id = canonical_fingerprint(
            {"kind": "maxwell-capabilities", **values}
        )


class MaxwellPrimaryState(StrictModule):
    """Conservative electromagnetic fluxes and charge."""

    electric_displacement: Array
    magnetic_flux: Array
    charge: Array


class MaxwellAuxiliaryState(StrictModule):
    """Material and boundary/PML state kept outside the primary fluxes."""

    material: Any
    boundary: Any


class CompatibleMaxwellState(StrictModule):
    """Complete Maxwell carry with primary, auxiliary, and observer state."""

    primary: MaxwellPrimaryState
    auxiliary: MaxwellAuxiliaryState
    observations: Any


class CompatibleMaxwellDiagnostics(StrictModule):
    """Dynamic conservation, energy, source-power, and CFL evidence."""

    energy: Array
    electric_constraint_linf: Array
    pml_dissipation: Array
    magnetic_constraint_linf: Array
    magnetic_period_residual: Array
    magnetic_projection_status: Array
    magnetic_projection_iterations: Array
    gauss_rate_linf: Array
    source_power: Array
    boundary_dissipation: Array
    power_balance_residual: Array
    stable_step: Array
    step_fraction: Array | None


class AbstractPreparedMaxwellConstitutive(StrictModule):
    """Prepared D↔E and B↔H maps with energy and auxiliary-state semantics."""

    capabilities: MaxwellCapabilities
    layout_id: str
    prepared_id: str

    @abc.abstractmethod
    def initialize_state(self, /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_state(self, state: Any, /) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def magnetic_conduction(self, magnetic: Array, state: Any, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def advance_state(
        self,
        time: Array,
        state: Any,
        displacement: Array,
        magnetic_flux: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def energy(
        self,
        displacement: Array,
        magnetic_flux: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def energy_rate(
        self,
        displacement: Array,
        magnetic_flux: Array,
        displacement_rate: Array,
        magnetic_rate: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def wave_speed_bound(self, /) -> Array:
        """Return a conservative material wave-speed bound."""
        raise NotImplementedError


class AbstractMaxwellConstitutivePlan(StrictModule):
    """Plan that binds a material law to exact cochain spaces."""

    plan_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def prepare(
        self,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ) -> AbstractPreparedMaxwellConstitutive:
        raise NotImplementedError


def _apply_hodge_metric(metric: Array, values: Array, /) -> Array:
    return metric * values if metric.ndim == 1 else metric @ values


def _positive_material(name: str, value: ArrayLike, count: int, /) -> Array:
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be real.")
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    if array.shape not in ((), (1,), (count,)):
        raise ValueError(f"{name} must be scalar or have shape ({count},).")
    array = jnp.broadcast_to(array, (count,))
    return eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)) | jnp.any(array <= 0.0),
        f"{name} must be finite and strictly positive.",
    )


class DiagonalMaxwellConstitutivePlan(AbstractMaxwellConstitutivePlan):
    """Positive degree-aligned instantaneous permittivity/permeability."""

    permittivity: Array
    permeability: Array

    def __init__(
        self,
        *,
        permittivity: ArrayLike = 1.0,
        permeability: ArrayLike = 1.0,
        plan_id: str | None = None,
    ):
        epsilon = jnp.asarray(permittivity)
        mu = jnp.asarray(permeability)
        if jnp.iscomplexobj(epsilon) or jnp.iscomplexobj(mu):
            raise TypeError("Diagonal lossless Maxwell materials must be real.")
        if not jnp.issubdtype(epsilon.dtype, jnp.inexact):
            epsilon = epsilon.astype(float)
        if not jnp.issubdtype(mu.dtype, jnp.inexact):
            mu = mu.astype(float)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "diagonal-maxwell-constitutive-plan",
                    "permittivity": array_tree_fingerprint(epsilon),
                    "permeability": array_tree_fingerprint(mu),
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.permittivity = epsilon
        self.permeability = mu
        self.plan_id = identifier

    def prepare(
        self,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ) -> PreparedDiagonalMaxwellConstitutive:
        return PreparedDiagonalMaxwellConstitutive(self, cochain, layout)


class PreparedDiagonalMaxwellConstitutive(AbstractPreparedMaxwellConstitutive):
    """Exact diagonal D↔E and B↔H material maps."""

    permittivity: Array
    permeability: Array
    capabilities: MaxwellCapabilities
    layout_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DiagonalMaxwellConstitutivePlan,
        cochain: CochainDiscretization,
        layout: MaxwellCochainLayout,
        /,
    ):
        if not isinstance(layout, MaxwellCochainLayout):
            raise TypeError("Maxwell constitutive preparation requires a cochain layout.")
        epsilon = _positive_material(
            "permittivity",
            plan.permittivity,
            layout.electric_count,
        )
        mu = _positive_material(
            "permeability",
            plan.permeability,
            layout.magnetic_count,
        )
        self.permittivity = epsilon
        self.permeability = mu
        self.layout_id = layout.layout_id
        self.capabilities = MaxwellCapabilities(
            lossless=True,
            passive=True,
            reversible=True,
            structured_only=False,
            frequency_domain=True,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-diagonal-maxwell-constitutive",
                "plan": plan.plan_id,
                "cochain": cochain.prepared_id,
                "layout": layout.layout_id,
            }
        )

    def initialize_state(self, /) -> None:
        return None

    def validate_state(self, state: Any, /) -> None:
        if state is not None:
            raise ValueError("Diagonal instantaneous material state must be None.")

    def electric_field(self, displacement: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return displacement / self.permittivity

    def electric_displacement(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permittivity * electric

    def magnetic_field(self, flux: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return flux / self.permeability

    def magnetic_flux(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return self.permeability * magnetic

    def electric_conduction(self, electric: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(electric)

    def magnetic_conduction(self, magnetic: Array, state: Any, /) -> Array:
        self.validate_state(state)
        return jnp.zeros_like(magnetic)

    def dissipated_power(
        self,
        electric: Array,
        magnetic: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        del electric, magnetic, electric_star, magnetic_star
        self.validate_state(state)
        return jnp.asarray(0.0)

    def advance_state(
        self,
        time: Array,
        state: Any,
        displacement: Array,
        magnetic_flux: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> None:
        del time, displacement, magnetic_flux, step_size, args
        self.validate_state(state)

    def energy(
        self,
        displacement: Array,
        magnetic_flux: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return 0.5 * jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_flux))
        )

    def energy_rate(
        self,
        displacement: Array,
        magnetic_flux: Array,
        displacement_rate: Array,
        magnetic_rate: Array,
        state: Any,
        electric_star: Array,
        magnetic_star: Array,
        /,
    ) -> Array:
        electric = self.electric_field(displacement, state)
        magnetic = self.magnetic_field(magnetic_flux, state)
        return jnp.real(
            jnp.vdot(electric, _apply_hodge_metric(electric_star, displacement_rate))
            + jnp.vdot(magnetic, _apply_hodge_metric(magnetic_star, magnetic_rate))
        )

    def wave_speed_bound(self, /) -> Array:
        return jnp.sqrt(jnp.max(1.0 / self.permeability) / jnp.min(self.permittivity))


class CompatibleMaxwellPlan(StrictModule):
    """Compatible Maxwell plan over one role-aware retained de Rham segment."""

    bridge: StructuredCochainBridge
    layout: MaxwellCochainLayout
    constitutive: AbstractMaxwellConstitutivePlan
    boundaries: tuple[MaxwellBoundaryPlan, ...]
    observers: tuple[AbstractMaxwellObserverPlan, ...]
    sources: tuple[AbstractMaxwellSourcePlan, ...]
    pml: MaxwellCPMLPlan | None
    harmonic_constraint: HarmonicConstraint | None
    magnetic_constraint: MaxwellMagneticConstraintPolicy
    resources: MaxwellResourcePolicy
    courant_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        polarization: MaxwellPolarization = "full_3d",
        constitutive: AbstractMaxwellConstitutivePlan | None = None,
        boundaries: Sequence[MaxwellBoundaryPlan] = (),
        observers: Sequence[AbstractMaxwellObserverPlan] = (),
        sources: Sequence[AbstractMaxwellSourcePlan] = (),
        pml: MaxwellCPMLPlan | None = None,
        harmonic_constraint: HarmonicConstraint | None = None,
        magnetic_constraint: MaxwellMagneticConstraintPolicy | None = None,
        resources: MaxwellResourcePolicy | None = None,
        courant_factor: float = 0.95,
        plan_id: str | None = None,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("Compatible Maxwell requires a StructuredCochainBridge.")
        layout = MaxwellCochainLayout(bridge, polarization)
        material = (
            DiagonalMaxwellConstitutivePlan() if constitutive is None else constitutive
        )
        if not isinstance(material, AbstractMaxwellConstitutivePlan):
            raise TypeError("constitutive must be an AbstractMaxwellConstitutivePlan.")
        boundary_plans = tuple(boundaries)
        if any(not isinstance(value, MaxwellBoundaryPlan) for value in boundary_plans):
            raise TypeError("boundaries must contain MaxwellBoundaryPlan values.")
        observer_plans = tuple(observers)
        if any(
            not isinstance(value, AbstractMaxwellObserverPlan) for value in observer_plans
        ):
            raise TypeError("observers must contain AbstractMaxwellObserverPlan values.")
        source_plans = tuple(sources)
        if any(
            not isinstance(value, AbstractMaxwellSourcePlan) for value in source_plans
        ):
            raise TypeError("sources must contain AbstractMaxwellSourcePlan values.")
        if pml is not None and not isinstance(pml, MaxwellCPMLPlan):
            raise TypeError("pml must be MaxwellCPMLPlan or None.")
        if harmonic_constraint is not None:
            if not isinstance(harmonic_constraint, HarmonicConstraint):
                raise TypeError("harmonic_constraint must be HarmonicConstraint or None.")
            if harmonic_constraint.frame.degree != layout.magnetic_degree:
                raise ValueError("Maxwell harmonic constraint degree does not match B.")
            expected_source = CellSubcomplex.full(bridge.cochain.topology).subcomplex_id
            if harmonic_constraint.frame.exact_basis.source_id != expected_source:
                raise ValueError(
                    "Maxwell harmonic constraint belongs to another topology."
                )
        constraint_policy = magnetic_constraint or MaxwellMagneticConstraintPolicy()
        resource_policy = resources or MaxwellResourcePolicy()
        factor = float(courant_factor)
        if not np.isfinite(factor) or factor <= 0.0 or factor > 1.0:
            raise ValueError("courant_factor must be finite and lie in (0, 1].")
        identifier = plan_id or canonical_fingerprint(
            {
                "kind": "compatible-maxwell-plan",
                "bridge": bridge.bridge_id,
                "layout": layout.layout_id,
                "constitutive": material.plan_id,
                "boundaries": [value.plan_id for value in boundary_plans],
                "observers": [value.plan_id for value in observer_plans],
                "sources": [value.source_id for value in source_plans],
                "pml": None if pml is None else pml.plan_id,
                "harmonic_constraint": None
                if harmonic_constraint is None
                else harmonic_constraint.constraint_id,
                "magnetic_constraint": constraint_policy.mode,
                "courant_factor": factor,
            }
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.bridge, self.layout, self.constitutive = bridge, layout, material
        self.boundaries, self.observers, self.sources = (
            boundary_plans,
            observer_plans,
            source_plans,
        )
        self.pml, self.harmonic_constraint = pml, harmonic_constraint
        self.magnetic_constraint, self.resources = constraint_policy, resource_policy
        self.courant_factor, self.plan_id = factor, str(identifier)

    def prepare(self, /) -> PreparedCompatibleMaxwell:
        return PreparedCompatibleMaxwell(self)


class PreparedCompatibleMaxwell(StrictModule):
    """Prepared heterogeneous compatible Maxwell evolution."""

    plan: CompatibleMaxwellPlan
    layout: MaxwellCochainLayout
    constitutive: AbstractPreparedMaxwellConstitutive
    boundaries: tuple[PreparedMaxwellBoundary, ...]
    observers: tuple[AbstractPreparedMaxwellObserver, ...]
    sources: tuple[PreparedMaxwellSource, ...]
    capabilities: MaxwellCapabilities
    pml: PreparedMaxwellCPML | None
    magnetic_incidence: Any
    magnetic_constraint_solver: Any
    magnetic_projection_elided: bool = eqx.field(static=True)
    harmonic_constraint: HarmonicConstraint | None
    cfl_limit: Array
    stable_dt: Array
    resource_estimate: MaxwellResourceEstimate
    discretization_bundle: DiscretizationBundle
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: CompatibleMaxwellPlan, /):
        if not isinstance(plan, CompatibleMaxwellPlan):
            raise TypeError("plan must be a CompatibleMaxwellPlan.")
        layout, cochain = plan.layout, plan.bridge.cochain
        scalar_bytes = jnp.dtype(jnp.complex128).itemsize
        primary_bytes = scalar_bytes * (
            layout.electric_count + layout.magnetic_count + layout.charge_count
        )
        if (
            primary_bytes > plan.resources.maximum_state_bytes
            or primary_bytes > plan.resources.maximum_total_bytes
        ):
            raise ValueError(
                "Logical Maxwell state exceeds the declared resource budget."
            )
        constitutive = plan.constitutive.prepare(cochain, layout)
        boundaries = tuple(
            value.prepare(plan.bridge, layout) for value in plan.boundaries
        )
        observers = tuple(value.prepare(layout) for value in plan.observers)
        sources = tuple(value.prepare(plan.bridge, layout) for value in plan.sources)
        pml = None if plan.pml is None else plan.pml.prepare(plan.bridge, layout)
        spacings = tuple(
            jnp.min(axis.interval_widths) for axis in plan.bridge.grid.structured_axes
        )
        inverse_spacing_norm = jnp.sqrt(jnp.sum(1.0 / jnp.asarray(spacings) ** 2))
        cfl_limit = 1.0 / (constitutive.wave_speed_bound() * inverse_spacing_norm)
        stable_dt = plan.courant_factor * cfl_limit
        system_key = DiscretizationKey(
            "compatible_maxwell",
            DiscretizationRole.RESIDUAL,
            domain_labels=plan.bridge.grid.axis_names,
        )
        bundle = DiscretizationBundle(
            (
                DiscretizationRecord(
                    cochain.key,
                    type(cochain).__name__,
                    cochain.prepared_id,
                    numeric_version=cochain.numeric_version,
                    precision_evidence_id=cochain.precision_evidence_id,
                    resource_evidence_id=cochain.resource_evidence_id,
                ),
                DiscretizationRecord(
                    system_key,
                    "compatible-maxwell-system",
                    plan.plan_id,
                    dependency_key_ids=(cochain.key.key_id,),
                ),
            )
        )
        top_form = layout.magnetic_degree == cochain.max_degree
        preserving = top_form or (
            constitutive.capabilities.magnetic_closedness_preserving
            and all(value.magnetic_closedness_preserving for value in boundaries)
            and all(value.magnetic_closedness_preserving for value in sources)
            and pml is None
        )
        if plan.magnetic_constraint.mode == "elide" and not preserving:
            raise ValueError("Magnetic projection elision lacks closedness evidence.")
        projection_elided = top_form or (
            plan.magnetic_constraint.mode in ("auto", "elide") and preserving
        )
        magnetic_incidence = (
            None
            if top_form
            else cochain.topology.incidences[layout.magnetic_degree].exterior_derivative()
        )
        magnetic_constraint_solver = (
            None
            if top_form or plan.magnetic_constraint.mode == "elide"
            else prepare(
                MinimumNormProblem(
                    magnetic_incidence,
                    problem_id=f"{plan.plan_id}:magnetic-constraint-projection",
                ),
                plan.magnetic_constraint.solve_policy,
            )
        )
        material_state = constitutive.initialize_state()
        material_bytes = sum(
            int(leaf.size * max(leaf.dtype.itemsize, scalar_bytes))
            for leaf in jax.tree_util.tree_leaves(material_state)
            if isinstance(leaf, jax.Array)
        )
        observer_states = tuple(value.initialize() for value in observers)
        observer_bytes = sum(
            int(leaf.size * max(leaf.dtype.itemsize, scalar_bytes))
            for leaf in jax.tree_util.tree_leaves(observer_states)
            if isinstance(leaf, jax.Array)
        )
        cpml_bytes = 0 if pml is None else scalar_bytes * pml.state_elements
        target_count = 0 if magnetic_incidence is None else magnetic_incidence.target.size
        projection_workspace = (
            0
            if magnetic_constraint_solver is None
            else scalar_bytes * 20 * (layout.magnetic_count + target_count)
        )
        returned_bytes = observer_bytes
        state_bytes = primary_bytes + material_bytes + observer_bytes + cpml_bytes
        total_bytes = state_bytes + projection_workspace + returned_bytes
        estimate = MaxwellResourceEstimate(
            primary_bytes,
            material_bytes,
            observer_bytes,
            cpml_bytes,
            projection_workspace,
            0,
            state_bytes,
            state_bytes,
            returned_bytes,
            total_bytes,
        )
        if state_bytes > plan.resources.maximum_state_bytes:
            raise ValueError("Prepared Maxwell state exceeds the declared state budget.")
        if projection_workspace > plan.resources.maximum_workspace_bytes:
            raise ValueError("Maxwell projection exceeds the declared workspace budget.")
        if returned_bytes > plan.resources.maximum_acquisition_bytes:
            raise ValueError("Maxwell acquisition exceeds the declared output budget.")
        if total_bytes > plan.resources.maximum_total_bytes:
            raise ValueError(
                "Prepared Maxwell execution exceeds the total resource budget."
            )
        self.plan, self.layout, self.constitutive = plan, layout, constitutive
        self.boundaries, self.observers, self.sources = boundaries, observers, sources
        self.capabilities = MaxwellCapabilities(
            lossless=constitutive.capabilities.lossless
            and all(value.kind != "impedance" for value in boundaries),
            passive=constitutive.capabilities.passive,
            active=constitutive.capabilities.active,
            dispersive=constitutive.capabilities.dispersive,
            nonlinear=constitutive.capabilities.nonlinear,
            reversible=(
                constitutive.capabilities.reversible
                and all(value.kind != "impedance" for value in boundaries)
                and pml is None
                and not sources
            ),
            observers=bool(observers),
            structured_only=True,
            pml=pml is not None,
            frequency_domain=constitutive.capabilities.frequency_domain,
            distributed=False,
            magnetic_closedness_preserving=preserving,
            linear_time_invariant=constitutive.capabilities.linear_time_invariant,
            local_tensors=False,
            spatial_distribution=False,
            ffi=False,
        )
        self.pml, self.harmonic_constraint = pml, plan.harmonic_constraint
        self.magnetic_incidence = magnetic_incidence
        self.magnetic_constraint_solver = magnetic_constraint_solver
        self.magnetic_projection_elided = projection_elided
        self.cfl_limit, self.stable_dt = cfl_limit, stable_dt
        self.resource_estimate, self.discretization_bundle = estimate, bundle
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-compatible-maxwell",
                "plan": plan.plan_id,
                "layout": layout.layout_id,
                "cochain": cochain.prepared_id,
                "constitutive": constitutive.prepared_id,
                "sources": [value.prepared_id for value in sources],
            }
        )

    @property
    def primary_counts(self) -> tuple[int, int, int]:
        return (
            self.layout.electric_count,
            self.layout.magnetic_count,
            self.layout.charge_count,
        )

    def _constrain_primary(
        self,
        displacement: Array,
        magnetic_flux: Array,
        /,
    ) -> tuple[Array, Array]:
        for boundary in self.boundaries:
            displacement, magnetic_flux = boundary.constrain_primary(
                displacement,
                magnetic_flux,
            )
        return displacement, magnetic_flux

    def pack(
        self,
        electric_displacement: ArrayLike,
        magnetic_flux: ArrayLike,
        charge: ArrayLike | None = None,
        /,
        *,
        material_state: Any = None,
        boundary_state: Any = None,
        observations: Any = None,
    ) -> CompatibleMaxwellState:
        displacement = jnp.asarray(electric_displacement)
        flux = jnp.asarray(magnetic_flux)
        electric_count, magnetic_count, charge_count = self.primary_counts
        charge_ = (
            jnp.zeros((charge_count,), dtype=displacement.dtype)
            if charge is None
            else jnp.asarray(charge)
        )
        dtype = jnp.result_type(displacement, flux, charge_)
        if not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.dtype(float)
        displacement = displacement.astype(dtype)
        flux = flux.astype(dtype)
        charge_ = charge_.astype(dtype)
        if boundary_state is None:
            boundary_state_ = (
                None if self.pml is None else self.pml.initialize(dtype=dtype)
            )
        else:
            boundary_state_ = boundary_state
            if self.pml is None:
                raise ValueError("Maxwell boundary state is incompatible without CPML.")
            self.pml.validate_state(boundary_state_)
        displacement, flux = self._constrain_primary(displacement, flux)
        if observations is None:
            observation_state = tuple(value.initialize() for value in self.observers)
        else:
            observation_state = tuple(observations)
            if len(observation_state) != len(self.observers):
                raise ValueError(
                    "Maxwell observation state count does not match observers."
                )
        if (
            displacement.shape != (electric_count,)
            or flux.shape != (magnetic_count,)
            or charge_.shape != (charge_count,)
        ):
            raise ValueError("Maxwell D, B, and charge cochains have wrong sizes.")
        finite = (
            jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(flux))
            & jnp.all(jnp.isfinite(charge_))
        )
        displacement = eqx.error_if(
            displacement,
            ~finite,
            "Maxwell primary cochains must be finite.",
        )
        if self.layout.magnetic_degree < self.plan.bridge.cochain.max_degree:
            if self.plan.magnetic_constraint.mode == "elide":
                residual = self.magnetic_incidence.mv(flux)
                tolerance = max(
                    self.plan.magnetic_constraint.absolute_tolerance,
                    self.plan.magnetic_constraint.relative_tolerance
                    * float(np.linalg.norm(np.asarray(flux))),
                )
                if float(np.linalg.norm(np.asarray(residual))) > tolerance:
                    raise ValueError(
                        "Projection-elided initial magnetic flux is not closed."
                    )
            else:
                flux, _ = self._project_magnetic_constraint(flux, force=True)
        self.constitutive.validate_state(material_state)
        return CompatibleMaxwellState(
            primary=MaxwellPrimaryState(
                electric_displacement=displacement,
                magnetic_flux=flux,
                charge=charge_,
            ),
            auxiliary=MaxwellAuxiliaryState(
                material=material_state,
                boundary=boundary_state_,
            ),
            observations=observation_state,
        )

    def initialize(
        self,
        *,
        electric_displacement: ArrayLike | None = None,
        magnetic_flux: ArrayLike | None = None,
        charge: ArrayLike | None = None,
    ) -> CompatibleMaxwellState:
        electric_count, magnetic_count, _ = self.primary_counts
        displacement = (
            jnp.zeros((electric_count,))
            if electric_displacement is None
            else electric_displacement
        )
        flux = jnp.zeros((magnetic_count,)) if magnetic_flux is None else magnetic_flux
        return self.pack(
            displacement,
            flux,
            charge,
            material_state=self.constitutive.initialize_state(),
        )

    def _state(self, state: CompatibleMaxwellState, /) -> CompatibleMaxwellState:
        if not isinstance(state, CompatibleMaxwellState):
            raise TypeError("state must be CompatibleMaxwellState.")
        electric_count, magnetic_count, charge_count = self.primary_counts
        primary = state.primary
        if (
            primary.electric_displacement.shape != (electric_count,)
            or primary.magnetic_flux.shape != (magnetic_count,)
            or primary.charge.shape != (charge_count,)
        ):
            raise ValueError("Maxwell D, B, and charge cochains have wrong sizes.")
        if len(tuple(state.observations)) != len(self.observers):
            raise ValueError("Maxwell observation state count does not match observers.")
        self.constitutive.validate_state(state.auxiliary.material)
        if self.pml is None:
            if state.auxiliary.boundary is not None:
                raise ValueError("Maxwell boundary state is incompatible without CPML.")
        else:
            self.pml.validate_state(state.auxiliary.boundary)
        finite = (
            jnp.all(jnp.isfinite(primary.electric_displacement))
            & jnp.all(jnp.isfinite(primary.magnetic_flux))
            & jnp.all(jnp.isfinite(primary.charge))
        )
        eqx.error_if(
            primary.electric_displacement,
            ~finite,
            "Maxwell primary cochains must be finite.",
        )
        return state

    def electric_field(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        electric = self.constitutive.electric_field(
            state_.primary.electric_displacement,
            state_.auxiliary.material,
        )
        magnetic = self.constitutive.magnetic_field(
            state_.primary.magnetic_flux,
            state_.auxiliary.material,
        )
        for boundary in self.boundaries:
            electric, magnetic = boundary.constrain_fields(electric, magnetic)
        return electric

    def magnetic_field(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        electric = self.constitutive.electric_field(
            state_.primary.electric_displacement,
            state_.auxiliary.material,
        )
        magnetic = self.constitutive.magnetic_field(
            state_.primary.magnetic_flux,
            state_.auxiliary.material,
        )
        for boundary in self.boundaries:
            electric, magnetic = boundary.constrain_fields(electric, magnetic)
        return magnetic

    def _source_forcing(
        self,
        time: Array,
        state: CompatibleMaxwellState,
        args: Any,
        /,
    ) -> MaxwellSourceForcing:
        dtype = jnp.result_type(
            state.primary.electric_displacement,
            state.primary.magnetic_flux,
        )
        electric = jnp.zeros((self.layout.electric_count,), dtype=dtype)
        magnetic = jnp.zeros((self.layout.magnetic_count,), dtype=dtype)
        for source in self.sources:
            value = source.sample(time, args)
            electric = electric + value.electric_current
            magnetic = magnetic + value.magnetic_current
        finite = jnp.all(jnp.isfinite(electric)) & jnp.all(jnp.isfinite(magnetic))
        electric = eqx.error_if(
            electric, ~finite, "Maxwell source forcing must be finite."
        )
        return MaxwellSourceForcing(electric, magnetic)

    def _boundary_current(self, electric: Array, /) -> Array:
        current = jnp.zeros_like(electric)
        for boundary in self.boundaries:
            current = current + boundary.impedance_current(electric)
        return current

    def boundary_dissipation(self, state: CompatibleMaxwellState, /) -> Array:
        electric = self.electric_field(state)
        total = jnp.asarray(0.0, dtype=electric.real.dtype)
        for boundary in self.boundaries:
            total = total + boundary.dissipated_power(electric)
        return total

    def pml_dissipation(self, state: CompatibleMaxwellState, /) -> Array:
        if self.pml is None:
            return jnp.asarray(0.0)
        return self.pml.diagnostics(
            self.electric_field(state),
            self.magnetic_field(state),
            self.plan.bridge.cochain.hodge_metric(self.layout.electric_degree),
            self.plan.bridge.cochain.hodge_metric(self.layout.magnetic_degree),
        ).absorbed_power

    def material_dissipation(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        return self.constitutive.dissipated_power(
            self.electric_field(state_),
            self.magnetic_field(state_),
            state_.auxiliary.material,
            self.plan.bridge.cochain.hodge_metric(self.layout.electric_degree),
            self.plan.bridge.cochain.hodge_metric(self.layout.magnetic_degree),
        )

    def _electric_curl_components(self, magnetic: Array, /) -> Array:
        return jnp.stack(
            tuple(
                self.plan.bridge.directional_codifferential(
                    self.layout.magnetic_degree,
                    magnetic,
                    axis,
                )
                for axis in range(self.plan.bridge.dimension)
            ),
            axis=0,
        )

    def _magnetic_curl_components(self, electric: Array, /) -> Array:
        return jnp.stack(
            tuple(
                -self.plan.bridge.directional_exterior_derivative(
                    self.layout.electric_degree,
                    electric,
                    axis,
                )
                for axis in range(self.plan.bridge.dimension)
            ),
            axis=0,
        )

    def _rates_with_forcing(
        self,
        state: CompatibleMaxwellState,
        forcing: MaxwellSourceForcing,
        /,
    ) -> MaxwellPrimaryState:
        electric = self.electric_field(state)
        magnetic = self.magnetic_field(state)
        total_current = (
            forcing.electric_current
            + self._boundary_current(electric)
            + self.constitutive.electric_conduction(
                electric,
                state.auxiliary.material,
            )
        )
        magnetic_loss = self.constitutive.magnetic_conduction(
            magnetic,
            state.auxiliary.material,
        )
        electric_components = self._electric_curl_components(magnetic)
        magnetic_components = self._magnetic_curl_components(electric)
        if self.pml is None:
            electric_curl = jnp.sum(electric_components, axis=0)
            magnetic_curl = jnp.sum(magnetic_components, axis=0)
        else:
            boundary_state = state.auxiliary.boundary
            if not isinstance(boundary_state, MaxwellCPMLState):
                raise ValueError("Prepared CPML requires MaxwellCPMLState.")
            electric_curl = self.pml.electric_rate(
                electric_components,
                boundary_state,
            )
            magnetic_curl = self.pml.magnetic_rate(
                magnetic_components,
                boundary_state,
            )
        displacement_rate = electric_curl - total_current
        magnetic_rate = magnetic_curl - magnetic_loss - forcing.magnetic_current
        charge_rate = (
            jnp.zeros((0,), dtype=displacement_rate.dtype)
            if self.layout.charge_degree is None
            else self.plan.bridge.codifferential(
                self.layout.electric_degree,
                displacement_rate,
            )
        )
        return MaxwellPrimaryState(
            electric_displacement=displacement_rate,
            magnetic_flux=magnetic_rate,
            charge=charge_rate,
        )

    def drift(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
    ) -> MaxwellPrimaryState:
        """Return primary D/B/charge rates; complete carries use ``leapfrog_step``."""
        state_ = self._state(state)
        forcing = self._source_forcing(jnp.asarray(time), state_, args)
        return self._rates_with_forcing(state_, forcing)

    def _step_size(self, step_size: ArrayLike, /) -> Array:
        dt = jnp.asarray(step_size)
        if dt.shape != ():
            raise ValueError("Maxwell step_size must be scalar.")
        return eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0.0) | (dt > self.stable_dt),
            "Maxwell step_size must be finite, positive, and no larger than stable_dt.",
        )

    def _project_magnetic_constraint(
        self,
        magnetic_flux: Array,
        /,
        *,
        force: bool = False,
    ) -> tuple[Array, MaxwellMagneticConstraintEvidence]:
        if self.magnetic_incidence is None:
            zero = jnp.asarray(0.0, dtype=magnetic_flux.real.dtype)
            return magnetic_flux, MaxwellMagneticConstraintEvidence(
                zero,
                zero,
                zero,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
                True,
            )
        residual = self.magnetic_incidence.mv(magnetic_flux)
        residual_norm = jnp.linalg.norm(residual)
        scale = jnp.linalg.norm(magnetic_flux)
        relative = residual_norm / jnp.maximum(scale, jnp.finfo(scale.dtype).tiny)
        if self.magnetic_projection_elided and not force:
            return magnetic_flux, MaxwellMagneticConstraintEvidence(
                residual_norm,
                relative,
                jnp.asarray(0.0),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
                True,
            )
        if self.magnetic_constraint_solver is None:
            raise RuntimeError(
                "Magnetic projection was requested without a prepared solve."
            )
        if jnp.iscomplexobj(residual):
            real_result = solve_linear(
                self.magnetic_constraint_solver, jnp.real(residual)
            )
            imag_result = solve_linear(
                self.magnetic_constraint_solver, jnp.imag(residual)
            )
            correction = real_result.value + 1j * imag_result.value
            status = jnp.maximum(real_result.status, imag_result.status)
            iterations = (
                real_result.diagnostics.iterations + imag_result.diagnostics.iterations
            )
        else:
            result = solve_linear(self.magnetic_constraint_solver, residual)
            correction = result.value
            status, iterations = result.status, result.diagnostics.iterations
        projected = magnetic_flux - correction
        period_target = None
        if self.harmonic_constraint is not None:
            if self.harmonic_constraint.policy == "free":
                period_target = self.harmonic_constraint.frame.periods(magnetic_flux)
                projected = self.harmonic_constraint.frame.with_periods(
                    projected, period_target
                )
            else:
                projected = self.harmonic_constraint.apply(projected)
        final_residual = jnp.linalg.norm(self.magnetic_incidence.mv(projected))
        tolerance = jnp.maximum(
            self.plan.magnetic_constraint.absolute_tolerance,
            self.plan.magnetic_constraint.relative_tolerance * jnp.linalg.norm(projected),
        )
        projected = eqx.error_if(
            projected,
            final_residual > tolerance,
            "Magnetic minimum-norm projection did not meet its declared tolerance.",
        )
        if self.harmonic_constraint is None:
            period_residual = jnp.asarray(0.0, dtype=final_residual.dtype)
        elif period_target is not None:
            period_residual = jnp.linalg.norm(
                self.harmonic_constraint.frame.periods(projected) - period_target
            )
        else:
            period_residual = self.harmonic_constraint.residual(projected)
        return projected, MaxwellMagneticConstraintEvidence(
            final_residual,
            final_residual
            / jnp.maximum(
                jnp.linalg.norm(projected), jnp.finfo(final_residual.dtype).tiny
            ),
            period_residual,
            status,
            iterations,
            jnp.asarray(True),
            False,
        )

    def _step_core(
        self,
        time: Array,
        state: CompatibleMaxwellState,
        step_size: Array,
        args: Any,
        /,
        *,
        cpml_coefficients: MaxwellCPMLCoefficients | None = None,
        source_samples: tuple[
            MaxwellSourceForcing, MaxwellSourceForcing, MaxwellSourceForcing
        ]
        | None = None,
    ) -> CompatibleMaxwellState:
        dt, half_step = step_size, 0.5 * step_size
        magnetic_start = (
            self._source_forcing(time, state, args)
            if source_samples is None
            else source_samples[0]
        )
        electric = self.constitutive.electric_field(
            state.primary.electric_displacement, state.auxiliary.material
        )
        magnetic = self.constitutive.magnetic_field(
            state.primary.magnetic_flux, state.auxiliary.material
        )
        for boundary in self.boundaries:
            electric, magnetic = boundary.constrain_fields(electric, magnetic)
        magnetic_components = self._magnetic_curl_components(electric)
        boundary_half = state.auxiliary.boundary
        if self.pml is None:
            magnetic_curl = jnp.sum(magnetic_components, axis=0)
        else:
            magnetic_curl, boundary_half = self.pml.apply_magnetic(
                magnetic_components,
                boundary_half,
                half_step,
                coefficients=cpml_coefficients,
            )
        magnetic_forcing = (
            magnetic_curl
            - self.constitutive.magnetic_conduction(magnetic, state.auxiliary.material)
            - magnetic_start.magnetic_current
        )
        magnetic_half_flux = state.primary.magnetic_flux + half_step * magnetic_forcing
        magnetic_half_flux, _ = self._project_magnetic_constraint(magnetic_half_flux)
        material_half = self.constitutive.advance_state(
            time,
            state.auxiliary.material,
            state.primary.electric_displacement,
            magnetic_half_flux,
            half_step,
            args,
        )
        magnetic_half = self.constitutive.magnetic_field(
            magnetic_half_flux, material_half
        )
        electric_mid = (
            self._source_forcing(time + half_step, state, args)
            if source_samples is None
            else source_samples[1]
        )
        electric_half = self.constitutive.electric_field(
            state.primary.electric_displacement, material_half
        )
        total_current = (
            electric_mid.electric_current
            + self._boundary_current(electric_half)
            + self.constitutive.electric_conduction(electric_half, material_half)
        )
        electric_components = self._electric_curl_components(magnetic_half)
        if self.pml is None:
            electric_curl = jnp.sum(electric_components, axis=0)
        else:
            electric_curl, boundary_half = self.pml.apply_electric(
                electric_components,
                boundary_half,
                dt,
                coefficients=cpml_coefficients,
            )
        electric_forcing = electric_curl - total_current
        displacement_new = state.primary.electric_displacement + dt * electric_forcing
        charge_new = (
            jnp.zeros((0,), dtype=displacement_new.dtype)
            if self.layout.charge_degree is None
            else state.primary.charge
            + dt
            * self.plan.bridge.codifferential(
                self.layout.electric_degree,
                electric_forcing,
            )
        )
        material_new = self.constitutive.advance_state(
            time + half_step,
            material_half,
            displacement_new,
            magnetic_half_flux,
            half_step,
            args,
        )
        electric_new = self.constitutive.electric_field(displacement_new, material_new)
        magnetic_components_new = self._magnetic_curl_components(electric_new)
        boundary_new = boundary_half
        if self.pml is None:
            magnetic_curl_new = jnp.sum(magnetic_components_new, axis=0)
        else:
            magnetic_curl_new, boundary_new = self.pml.apply_magnetic(
                magnetic_components_new,
                boundary_half,
                half_step,
                coefficients=cpml_coefficients,
            )
        magnetic_end = (
            self._source_forcing(time + dt, state, args)
            if source_samples is None
            else source_samples[2]
        )
        magnetic_forcing_new = (
            magnetic_curl_new
            - self.constitutive.magnetic_conduction(
                self.constitutive.magnetic_field(magnetic_half_flux, material_new),
                material_new,
            )
            - magnetic_end.magnetic_current
        )
        magnetic_new = magnetic_half_flux + half_step * magnetic_forcing_new
        displacement_new, magnetic_new = self._constrain_primary(
            displacement_new, magnetic_new
        )
        if self.layout.charge_degree is not None:
            charge_new = self.plan.bridge.codifferential(
                self.layout.electric_degree,
                displacement_new,
            ) - self.electric_constraint(state)
        magnetic_new, _ = self._project_magnetic_constraint(magnetic_new)
        provisional = CompatibleMaxwellState(
            MaxwellPrimaryState(displacement_new, magnetic_new, charge_new),
            MaxwellAuxiliaryState(material_new, boundary_new),
            state.observations,
        )
        electric_observed = self.constitutive.electric_field(
            displacement_new, material_new
        )
        magnetic_observed = self.constitutive.magnetic_field(magnetic_new, material_new)
        for boundary in self.boundaries:
            electric_observed, magnetic_observed = boundary.constrain_fields(
                electric_observed, magnetic_observed
            )
        observation_state = tuple(
            observer.update(time + dt, electric_observed, magnetic_observed, value)
            for observer, value in zip(
                self.observers, tuple(state.observations), strict=True
            )
        )
        return CompatibleMaxwellState(
            provisional.primary,
            provisional.auxiliary,
            observation_state,
        )

    def leapfrog_step(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> CompatibleMaxwellState:
        state_ = self._state(state)
        dt = self._step_size(step_size)
        return self._step_core(jnp.asarray(time), state_, dt, args)

    def energy(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        return self.constitutive.energy(
            state_.primary.electric_displacement,
            state_.primary.magnetic_flux,
            state_.auxiliary.material,
            self.plan.bridge.cochain.hodge_metric(self.layout.electric_degree),
            self.plan.bridge.cochain.hodge_metric(self.layout.magnetic_degree),
        )

    def electric_constraint(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        if self.layout.charge_degree is None:
            return jnp.asarray(0.0, dtype=state_.primary.electric_displacement.real.dtype)
        return (
            self.plan.bridge.codifferential(
                self.layout.electric_degree,
                state_.primary.electric_displacement,
            )
            - state_.primary.charge
        )

    def observe(self, state: CompatibleMaxwellState, /) -> tuple[Array, ...]:
        state_ = self._state(state)
        return tuple(
            observer.value(value)
            for observer, value in zip(
                self.observers,
                tuple(state_.observations),
                strict=True,
            )
        )

    def magnetic_constraint(self, state: CompatibleMaxwellState, /) -> Array:
        state_ = self._state(state)
        if self.magnetic_incidence is None:
            return jnp.asarray(0.0, dtype=state_.primary.magnetic_flux.real.dtype)
        return self.magnetic_incidence.mv(state_.primary.magnetic_flux)

    def magnetic_constraint_evidence(
        self,
        state: CompatibleMaxwellState,
        /,
    ) -> MaxwellMagneticConstraintEvidence:
        state_ = self._state(state)
        residual = self.magnetic_constraint(state_)
        residual_norm = jnp.linalg.norm(residual)
        flux_norm = jnp.linalg.norm(state_.primary.magnetic_flux)
        period_residual = (
            jnp.asarray(0.0, dtype=residual_norm.dtype)
            if self.harmonic_constraint is None
            else self.harmonic_constraint.residual(state_.primary.magnetic_flux)
        )
        return MaxwellMagneticConstraintEvidence(
            residual_norm,
            residual_norm / jnp.maximum(flux_norm, jnp.finfo(residual_norm.dtype).tiny),
            period_residual,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            self.magnetic_projection_elided,
        )

    def source_power(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
    ) -> Array:
        state_ = self._state(state)
        forcing = self._source_forcing(jnp.asarray(time), state_, args)
        electric = self.electric_field(state_)
        magnetic = self.magnetic_field(state_)
        return jnp.real(
            jnp.vdot(
                electric,
                self.plan.bridge.cochain.apply_hodge(
                    self.layout.electric_degree,
                    forcing.electric_current,
                ),
            )
            + jnp.vdot(
                magnetic,
                self.plan.bridge.cochain.apply_hodge(
                    self.layout.magnetic_degree,
                    forcing.magnetic_current,
                ),
            )
        )

    def power_balance_residual(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
    ) -> Array:
        state_ = self._state(state)
        forcing = self._source_forcing(jnp.asarray(time), state_, args)
        rates = self._rates_with_forcing(state_, forcing)
        energy_rate = self.constitutive.energy_rate(
            state_.primary.electric_displacement,
            state_.primary.magnetic_flux,
            rates.electric_displacement,
            rates.magnetic_flux,
            state_.auxiliary.material,
            self.plan.bridge.cochain.hodge_metric(self.layout.electric_degree),
            self.plan.bridge.cochain.hodge_metric(self.layout.magnetic_degree),
        )
        return (
            energy_rate
            + self.source_power(time, state_, args)
            + self.boundary_dissipation(state_)
            + self.pml_dissipation(state_)
            + self.material_dissipation(state_)
        )

    def gauss_rate_residual(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
    ) -> Array:
        state_ = self._state(state)
        if self.layout.charge_degree is None:
            return jnp.asarray(0.0, dtype=state_.primary.electric_displacement.real.dtype)
        forcing = self._source_forcing(jnp.asarray(time), state_, args)
        rates = self._rates_with_forcing(state_, forcing)
        return (
            self.plan.bridge.codifferential(
                self.layout.electric_degree,
                rates.electric_displacement,
            )
            - rates.charge
        )

    def diagnostics(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
        *,
        step_size: ArrayLike | None = None,
    ) -> CompatibleMaxwellDiagnostics:
        electric = self.electric_constraint(state)
        magnetic = self.magnetic_constraint(state)
        gauss_rate = self.gauss_rate_residual(time, state, args)
        projection = self.magnetic_constraint_evidence(state)
        fraction = None if step_size is None else jnp.asarray(step_size) / self.stable_dt
        electric_linf = (
            jnp.abs(electric)
            if electric.ndim == 0
            else jnp.max(jnp.abs(electric), initial=0.0)
        )
        magnetic_linf = (
            jnp.abs(magnetic)
            if magnetic.ndim == 0
            else jnp.max(jnp.abs(magnetic), initial=0.0)
        )
        gauss_linf = (
            jnp.abs(gauss_rate)
            if gauss_rate.ndim == 0
            else jnp.max(jnp.abs(gauss_rate), initial=0.0)
        )
        return CompatibleMaxwellDiagnostics(
            energy=self.energy(state),
            electric_constraint_linf=electric_linf,
            magnetic_constraint_linf=magnetic_linf,
            magnetic_period_residual=projection.period_residual,
            magnetic_projection_status=projection.solver_status,
            magnetic_projection_iterations=projection.iterations,
            gauss_rate_linf=gauss_linf,
            source_power=self.source_power(time, state, args),
            boundary_dissipation=self.boundary_dissipation(state),
            pml_dissipation=self.pml_dissipation(state),
            power_balance_residual=self.power_balance_residual(time, state, args),
            stable_step=self.stable_dt,
            step_fraction=fraction,
        )


def _maxwell_static_semantics(value: Any, /) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(_maxwell_static_semantics(item) for item in value)
    if isinstance(value, FunctionType):
        code = value.__code__
        return (
            "function",
            value.__module__,
            value.__qualname__,
            code.co_firstlineno,
            code.co_code.hex(),
        )
    if callable(value):
        return ("callable", type(value).__module__, type(value).__qualname__)
    if is_dataclass(value):
        entries: list[tuple[str, Any]] = []
        for field in fields(value):
            if field.name.endswith("_id"):
                continue
            child = getattr(value, field.name)
            if field.metadata.get("static", False):
                entries.append((field.name, _maxwell_static_semantics(child)))
            elif is_dataclass(child):
                entries.append((field.name, _maxwell_static_semantics(child)))
            elif isinstance(child, tuple):
                nested = tuple(
                    _maxwell_static_semantics(item)
                    for item in child
                    if is_dataclass(item)
                )
                if nested:
                    entries.append((field.name, nested))
        return (
            "module",
            type(value).__module__,
            type(value).__qualname__,
            tuple(entries),
        )
    return ("value", type(value).__module__, type(value).__qualname__, str(value))


class _MaxwellStepSignature(StrictModule, NonTrainableState):
    bridge_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    material_class: str = eqx.field(static=True)
    boundary_classes: tuple[str, ...] = eqx.field(static=True)
    source_classes: tuple[str, ...] = eqx.field(static=True)
    observer_classes: tuple[str, ...] = eqx.field(static=True)
    material_array_signature: str = eqx.field(static=True)
    boundary_array_signatures: tuple[str, ...] = eqx.field(static=True)
    source_array_signatures: tuple[str, ...] = eqx.field(static=True)
    observer_array_signatures: tuple[str, ...] = eqx.field(static=True)
    pml_array_signature: str = eqx.field(static=True)
    static_semantics_signature: str = eqx.field(static=True)
    material_state_signature: str = eqx.field(static=True)
    observer_state_signatures: tuple[str, ...] = eqx.field(static=True)
    pml_state_signature: str = eqx.field(static=True)
    pml_term_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)


class _MaxwellStepParameters(StrictModule):
    step_size: Array
    cpml_coefficients: MaxwellCPMLCoefficients | None


class _PreparedMaxwellFixedStep(StrictModule):
    runtime: PreparedCompatibleMaxwell
    signature: _MaxwellStepSignature
    parameters: _MaxwellStepParameters
    prepared_id: str = eqx.field(static=True)

    def step(
        self,
        time: ArrayLike,
        state: CompatibleMaxwellState,
        args: Any = None,
        /,
    ) -> CompatibleMaxwellState:
        state_ = self.runtime._state(state)
        return self.runtime._step_core(
            jnp.asarray(time),
            state_,
            self.parameters.step_size,
            args,
            cpml_coefficients=self.parameters.cpml_coefficients,
        )


def _fixed_step(
    runtime: PreparedCompatibleMaxwell,
    step_size: ArrayLike,
    dtype: Any,
    /,
) -> _PreparedMaxwellFixedStep:
    dt = runtime._step_size(step_size)
    coefficients = (
        None if runtime.pml is None else runtime.pml.bind_coefficients(dt, 0.5 * dt)
    )
    pml_shapes = (
        ()
        if runtime.pml is None
        else tuple(
            term.indices.shape
            for term in (*runtime.pml.electric_terms, *runtime.pml.magnetic_terms)
        )
    )
    material_array_signature = canonical_fingerprint(
        {"arrays": array_tree_signature(runtime.constitutive)}
    )
    boundary_array_signatures = tuple(
        canonical_fingerprint({"arrays": array_tree_signature(value)})
        for value in runtime.boundaries
    )
    source_array_signatures = tuple(
        canonical_fingerprint({"arrays": array_tree_signature(value)})
        for value in runtime.sources
    )
    observer_array_signatures = tuple(
        canonical_fingerprint({"arrays": array_tree_signature(value)})
        for value in runtime.observers
    )
    pml_array_signature = canonical_fingerprint(
        {"arrays": array_tree_signature(runtime.pml)}
    )
    static_semantics_signature = canonical_fingerprint(
        {
            "constitutive": _maxwell_static_semantics(runtime.constitutive),
            "boundaries": tuple(
                _maxwell_static_semantics(value) for value in runtime.boundaries
            ),
            "sources": tuple(
                _maxwell_static_semantics(value) for value in runtime.sources
            ),
            "observers": tuple(
                _maxwell_static_semantics(value) for value in runtime.observers
            ),
            "pml": _maxwell_static_semantics(runtime.pml),
        }
    )
    material_state_signature = canonical_fingerprint(
        {"arrays": array_tree_signature(runtime.constitutive.initialize_state())}
    )
    observer_state_signatures = tuple(
        canonical_fingerprint({"arrays": array_tree_signature(value.initialize())})
        for value in runtime.observers
    )
    pml_state_signature = canonical_fingerprint(
        {
            "arrays": array_tree_signature(
                None
                if runtime.pml is None
                else runtime.pml.initialize(dtype=np.dtype(dtype))
            )
        }
    )
    dtype_name = np.dtype(dtype).name
    payload = {
        "bridge": runtime.plan.bridge.bridge_id,
        "layout": runtime.layout.layout_id,
        "material": type(runtime.constitutive).__name__,
        "boundaries": tuple(type(value).__name__ for value in runtime.boundaries),
        "sources": tuple(type(value).__name__ for value in runtime.sources),
        "observers": tuple(type(value).__name__ for value in runtime.observers),
        "material_arrays": material_array_signature,
        "boundary_arrays": boundary_array_signatures,
        "source_arrays": source_array_signatures,
        "observer_arrays": observer_array_signatures,
        "pml_arrays": pml_array_signature,
        "static_semantics": static_semantics_signature,
        "material_state": material_state_signature,
        "observer_states": observer_state_signatures,
        "pml_state": pml_state_signature,
        "pml_shapes": pml_shapes,
        "dtype": dtype_name,
    }
    signature = _MaxwellStepSignature(
        payload["bridge"],
        payload["layout"],
        payload["material"],
        payload["boundaries"],
        payload["sources"],
        payload["observers"],
        material_array_signature,
        boundary_array_signatures,
        source_array_signatures,
        observer_array_signatures,
        pml_array_signature,
        static_semantics_signature,
        material_state_signature,
        observer_state_signatures,
        pml_state_signature,
        pml_shapes,
        dtype_name,
        canonical_fingerprint({"kind": "maxwell-step-signature", **payload}),
    )
    return _PreparedMaxwellFixedStep(
        runtime,
        signature,
        _MaxwellStepParameters(dt, coefficients),
        canonical_fingerprint(
            {
                "kind": "prepared-maxwell-fixed-step",
                "runtime": runtime.prepared_id,
                "signature": signature.signature_id,
                "step_size": float(np.asarray(dt)),
            }
        ),
    )


def solve_compatible_maxwell(
    runtime: PreparedCompatibleMaxwell,
    initial_state: CompatibleMaxwellState,
    start_time: ArrayLike,
    step_size: ArrayLike,
    steps: int,
    args: Any = None,
    /,
) -> CompatibleMaxwellRunResult:
    """Stream a fixed-step run and return only final state and acquisitions."""

    if not isinstance(runtime, PreparedCompatibleMaxwell):
        raise TypeError("runtime must be PreparedCompatibleMaxwell.")
    count = int(steps)
    if count < 0:
        raise ValueError("Maxwell step count must be nonnegative.")
    state = runtime._state(initial_state)
    fixed = _fixed_step(runtime, step_size, state.primary.electric_displacement.dtype)
    start = jnp.asarray(start_time)

    def body(carry: CompatibleMaxwellState, index: Array, /):
        time = start + index * fixed.parameters.step_size
        return fixed.step(time, carry, args), None

    final_state, _ = jax.lax.scan(
        body,
        state,
        jnp.arange(count, dtype=jnp.int32),
    )
    final_time = start + count * fixed.parameters.step_size
    return CompatibleMaxwellRunResult(
        final_state,
        runtime.observe(final_state),
        runtime.diagnostics(
            final_time, final_state, args, step_size=fixed.parameters.step_size
        ),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(count, dtype=jnp.int32),
        runtime.resource_estimate,
    )


def refresh_compatible_maxwell(
    runtime: PreparedCompatibleMaxwell,
    spec: CompatibleMaxwellRefreshSpec,
    /,
) -> _PreparedMaxwellFixedStep:
    """Refresh same-signature numeric data; reject structural changes."""

    if not isinstance(runtime, PreparedCompatibleMaxwell) or not isinstance(
        spec, CompatibleMaxwellRefreshSpec
    ):
        raise TypeError("Maxwell refresh requires a runtime and refresh specification.")
    refreshed = spec.plan.prepare()
    if len(runtime.sources) != len(refreshed.sources) or any(
        previous.envelope is not current.envelope
        for previous, current in zip(runtime.sources, refreshed.sources, strict=True)
    ):
        raise ValueError("Maxwell refresh changed the executable step signature.")
    previous = _fixed_step(runtime, spec.step_size, np.dtype(spec.dtype))
    updated = _fixed_step(refreshed, spec.step_size, np.dtype(spec.dtype))
    if previous.signature.signature_id != updated.signature.signature_id:
        raise ValueError("Maxwell refresh changed the executable step signature.")
    return updated


__all__ = [
    "AbstractMaxwellConstitutivePlan",
    "AbstractPreparedMaxwellConstitutive",
    "CompatibleMaxwellDiagnostics",
    "CompatibleMaxwellPlan",
    "CompatibleMaxwellRefreshSpec",
    "CompatibleMaxwellRunResult",
    "CompatibleMaxwellState",
    "DiagonalMaxwellConstitutivePlan",
    "MaxwellAuxiliaryState",
    "MaxwellCapabilities",
    "MaxwellCochainLayout",
    "MaxwellMagneticConstraintEvidence",
    "MaxwellMagneticConstraintPolicy",
    "MaxwellPolarization",
    "MaxwellPrimaryState",
    "MaxwellResourceEstimate",
    "MaxwellResourcePolicy",
    "PreparedCompatibleMaxwell",
    "PreparedDiagonalMaxwellConstitutive",
    "refresh_compatible_maxwell",
    "solve_compatible_maxwell",
]
