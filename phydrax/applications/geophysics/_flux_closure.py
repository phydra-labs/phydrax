"""Host bindings for native operators predicting paired, closed-column fluxes.

Layers are top-to-bottom. Only interior interfaces are learned; positive flux
is downward. Water is explicitly VAPOR, and energy is TOTAL transported energy
(including the water contribution), in the bound moist caloric reference.
There is no conservation projection, positivity clipping, or hidden recurrence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...discretization import (
    DiscreteFieldSpace,
    DiscreteMeasure,
    EntityDofLayout,
    FieldTransfer,
)
from ...linalg import ArraySpace
from ...nn.operator import (
    FunctionSamples,
    OperatorBatch,
    OperatorCaseProvenance,
    OperatorFieldSpec,
    OperatorProblemSpec,
    OperatorQuerySpec,
    OperatorTargetBatch,
    OperatorTask,
)
from ...nn.operator.adapters import ExternalOperatorAdapter
from ...nn.operator.training import OperatorDataset, TrainedOperator
from ...units import derived_unit, JOULE, KILOGRAM, METER, ONE, SECOND
from ..atmosphere._moist import MoistThermodynamicPlan
from ._quantities import GeophysicalQuantity


_MASS_AREA = derived_unit("kg/m2", ((KILOGRAM, 1), (METER, -2)))
_SPECIFIC_ENERGY = derived_unit("J/kg", ((JOULE, 1), (KILOGRAM, -1)))
_FLUX_UNITS = (
    derived_unit("kg/m2/s", ((KILOGRAM, 1), (METER, -2), (SECOND, -1))),
    derived_unit("W/m2", ((JOULE, 1), (METER, -2), (SECOND, -1))),
)
_FLUX_NAMES = ("vapor_mass_flux", "total_energy_flux")


def column_flux_quantities(thermodynamics_id: str):
    """Named SI interface rates, not increments or sensible-only heat."""
    if not thermodynamics_id:
        raise ValueError("A thermodynamic reference identity is required.")
    return tuple(
        GeophysicalQuantity(
            name,
            kind,
            unit,
            axes=("interior_interface",),
            sign_convention="positive_downward",
            support_association="interior_interface",
            reference_configuration=thermodynamics_id,
        )
        for name, kind, unit in zip(
            _FLUX_NAMES, ("water_mass_flux", "heat_flux"), _FLUX_UNITS, strict=True
        )
    )


def column_flux_tasks(
    training_id: str,
    thermodynamics: MoistThermodynamicPlan,
    /,
    *,
    training_resolutions_m: Sequence[float],
    training_intervals_s: Sequence[float],
    training_regimes: Sequence[str],
    training_forcing_ids: Sequence[str],
) -> tuple[OperatorTask, OperatorTask]:
    """Two scalar native tasks retain distinct dimensions and shared experiment ID.

    Resolution/interval lists describe TRAINING support, not permission to claim
    generalization. New supports can be queried and must be independently scored.
    The native operator artifact owns model, fitted normalization and task identity.
    """
    resolutions, intervals = tuple(training_resolutions_m), tuple(training_intervals_s)
    regimes, forcing = tuple(training_regimes), tuple(training_forcing_ids)
    if not training_id or not regimes or not forcing or not all(regimes + forcing):
        raise ValueError("Training, regime and forcing identities must be nonempty.")
    if (
        not resolutions
        or not intervals
        or any(not np.isfinite(v) or v <= 0 for v in resolutions + intervals)
    ):
        raise ValueError("Training resolutions and intervals must be positive finite SI.")
    quantities = column_flux_quantities(thermodynamics.plan_id)
    sources = (
        OperatorFieldSpec("vapor_per_dry_mass", role="source", dimension=ONE.dimension),
        OperatorFieldSpec(
            "energy_per_dry_mass", role="source", dimension=_SPECIFIC_ENERGY.dimension
        ),
        OperatorFieldSpec("interval_seconds", role="source", dimension=SECOND.dimension),
        OperatorFieldSpec("resolution_m", role="source", dimension=METER.dimension),
        OperatorFieldSpec("forcing", role="source", dimension=ONE.dimension),
    )
    return tuple(
        OperatorTask(
            f"{training_id}/{quantity.name}",
            dimension_basis=tuple(
                sorted(
                    {name for field in sources for name, _, _ in field.dimension.terms}
                    | {name for name, _, _ in quantity.unit.dimension.terms}
                )
            ),
            fields=sources
            + (
                OperatorFieldSpec(
                    quantity.name,
                    role="target",
                    query_name="interfaces",
                    dimension=quantity.unit.dimension,
                ),
            ),
            queries=(
                OperatorQuerySpec(
                    "interfaces",
                    geometry_kind="point_cloud",
                    coordinate_components=("column_fraction",),
                    quadrature="physical_required",
                    fixed_geometry=False,
                ),
            ),
            problem=OperatorProblemSpec(
                source_query_relation="independent", query_is_fixed=False
            ),
            metadata={
                "geophysical_target_kind": "interior_flux_rate",
                "training_id": training_id,
                "thermodynamics_id": thermodynamics.plan_id,
                "mobile_water_phase": "vapor",
                "energy_semantics": "total_including_water",
                "boundary_flux": "closed_zero",
                "layer_order": "top_to_bottom",
                "flux_sign": "positive_downward",
                "quantity_id": quantity.quantity_id,
                "training_resolutions_m": list(resolutions),
                "training_intervals_s": list(intervals),
                "training_regimes": list(regimes),
                "training_forcing_ids": list(forcing),
            },
        )
        for quantity in quantities
    )


def column_flux_space(dry_mass: Any, support_id: str, /):
    """Native cell-average space and physical dry-mass measure (kg/m²)."""
    mass = np.asarray(dry_mass, dtype=float)
    if mass.ndim != 1 or mass.size < 2 or not np.all(np.isfinite(mass) & (mass > 0)):
        raise ValueError(
            "At least two positive finite top-to-bottom dry masses are required."
        )
    measure = DiscreteMeasure("column-dry-mass", support_id, f"{support_id}/layers", mass)
    space = DiscreteFieldSpace(
        "column-specific-inventory",
        support_id,
        EntityDofLayout(measure.entity_set_id, mass.size, mass.size),
        ArraySpace(
            (mass.size,), dtype=mass.dtype, space_id=f"{measure.measure_id}/values"
        ),
        representation="cell_average",
    )
    return space, measure


def _check_measure(space, measure):
    if not isinstance(space, DiscreteFieldSpace) or not isinstance(
        measure, DiscreteMeasure
    ):
        raise TypeError("Closure geometry requires native field spaces and measures.")
    if (
        space.representation != "cell_average"
        or space.support_id != measure.support_id
        or not isinstance(space.layout, EntityDofLayout)
        or space.layout.entity_set_id != measure.entity_set_id
        or space.vector_space.size != measure.weights.size
        or space.layout.component_shape != ()
        or measure.normalization != "physical"
        or not np.all(np.asarray(measure.active_mask))
        or not np.all(
            np.isfinite(np.asarray(measure.weights)) & (np.asarray(measure.weights) > 0)
        )
    ):
        raise ValueError(
            "Closure measure must cover the exact scalar cell-average space."
        )


@dataclass(frozen=True)
class ColumnFluxBinding:
    tasks: tuple[OperatorTask, OperatorTask]
    field: DiscreteFieldSpace
    measure: DiscreteMeasure
    interval_seconds: float
    resolution_m: float
    regime: str
    forcing_id: str
    measure_unit: Any = _MASS_AREA

    def __post_init__(self):
        _check_measure(self.field, self.measure)
        if self.measure.weights.size < 2:
            raise ValueError("Flux closure needs at least two layers.")
        if self.measure_unit != _MASS_AREA:
            raise ValueError("Dry-mass measures must explicitly use SI kg/m².")
        if any(
            not np.isfinite(x) or x <= 0
            for x in (self.interval_seconds, self.resolution_m)
        ):
            raise ValueError(
                "Closure dt and spatial resolution must be positive finite SI."
            )
        if not self.regime or not self.forcing_id or len(self.tasks) != 2:
            raise ValueError(
                "Two native tasks, regime and forcing identity are required."
            )
        reference = self.tasks[0].metadata.get("thermodynamics_id", "")
        training = self.tasks[0].metadata.get("training_id", "")
        for task, quantity in zip(
            self.tasks, column_flux_quantities(reference), strict=True
        ):
            expected = {
                "geophysical_target_kind": "interior_flux_rate",
                "training_id": training,
                "thermodynamics_id": reference,
                "mobile_water_phase": "vapor",
                "energy_semantics": "total_including_water",
                "boundary_flux": "closed_zero",
                "layer_order": "top_to_bottom",
                "flux_sign": "positive_downward",
                "quantity_id": quantity.quantity_id,
            }
            if not training or any(
                task.metadata.get(k) != v for k, v in expected.items()
            ):
                raise ValueError(
                    "Native task does not bind the vapor/total-energy flux contract."
                )
            target = task.field_by_name[quantity.name]
            if (
                not target.is_target
                or target.channels != "scalar"
                or target.dimension != quantity.unit.dimension
            ):
                raise ValueError(
                    "Flux target must be a named scalar with its SI dimension."
                )
            for name, dimension in (
                ("vapor_per_dry_mass", ONE.dimension),
                ("energy_per_dry_mass", _SPECIFIC_ENERGY.dimension),
                ("interval_seconds", SECOND.dimension),
                ("resolution_m", METER.dimension),
                ("forcing", ONE.dimension),
            ):
                source = task.field_by_name[name]
                if (
                    not source.is_source
                    or source.channels != "scalar"
                    or source.dimension != dimension
                ):
                    raise ValueError(
                        "Physical source field dimensions differ from the SI flux contract."
                    )

    @property
    def thermodynamics_id(self):
        return self.tasks[0].metadata["thermodynamics_id"]

    @property
    def binding_id(self):
        return canonical_fingerprint(
            {
                "tasks": [task.fingerprint for task in self.tasks],
                "field": self.field.field_space_id,
                "measure": self.measure.measure_id,
                "measure_unit": self.measure_unit.unit_id,
                "interval_seconds": self.interval_seconds,
                "resolution_m": self.resolution_m,
                "regime": self.regime,
                "forcing": self.forcing_id,
            }
        )

    def batch(self, vapor_mass, total_energy, /, *, forcing):
        """Build a native batch; forcing is a declared dimensionless regime control."""
        water, energy = np.asarray(vapor_mass), np.asarray(total_energy)
        mass = self.measure.weights
        if water.ndim != 2 or water.shape[-1] != mass.size or energy.shape != water.shape:
            raise ValueError("Batch inventories must have (case, ordered layer) shape.")
        if not np.all(np.isfinite(water) & (water >= 0)) or not np.all(
            np.isfinite(energy)
        ):
            raise ValueError(
                "Training/source inventories must be finite with nonnegative vapor."
            )
        forcing_ = np.asarray(forcing)
        if forcing_.shape != (water.shape[0],) or not np.all(np.isfinite(forcing_)):
            raise ValueError("Forcing must be one finite dimensionless control per case.")
        edges = jnp.concatenate((jnp.zeros(1), jnp.cumsum(mass))) / jnp.sum(mass)
        centers = (0.5 * (edges[1:] + edges[:-1]))[:, None]
        values = {
            "vapor_per_dry_mass": water / mass,
            "energy_per_dry_mass": energy / mass,
            "interval_seconds": jnp.full(water.shape, self.interval_seconds),
            "resolution_m": jnp.full(water.shape, self.resolution_m),
            "forcing": jnp.broadcast_to(forcing_[:, None], water.shape),
        }
        batch = OperatorBatch(
            inputs={
                name: FunctionSamples(
                    values=jnp.asarray(value),
                    coordinates=centers,
                    quadrature_weights=mass,
                    support_id=self.field.support_id,
                )
                for name, value in values.items()
            },
            queries={
                "interfaces": FunctionSamples(
                    values=None,
                    coordinates=edges[1:-1, None],
                    # Unit horizontal interface area, not layer inventory averaging.
                    quadrature_weights=jnp.ones(mass.size - 1),
                    support_id=f"{self.field.support_id}/interior-interfaces",
                )
            },
            case_axes=("case",),
            case_shape=(water.shape[0],),
        )
        self.validate_batch(batch)
        return batch

    def validate_batch(self, batch):
        """Require actual runtime coordinates/measures and interval conditioning."""
        mass = self.measure.weights
        edges = jnp.concatenate((jnp.zeros(1), jnp.cumsum(mass))) / jnp.sum(mass)
        centers = (0.5 * (edges[1:] + edges[:-1]))[:, None]
        for task in self.tasks:
            task.validate_batch(batch)
        for name in (
            "vapor_per_dry_mass",
            "energy_per_dry_mass",
            "interval_seconds",
            "resolution_m",
            "forcing",
        ):
            source = batch.input(name)
            if (
                source.support_id != self.field.support_id
                or not np.array_equal(np.asarray(source.coordinates), np.asarray(centers))
                or not np.array_equal(
                    np.asarray(source.quadrature_weights), np.asarray(mass)
                )
                or source.values.shape != batch.case_shape + (mass.size,)
                or not np.all(np.isfinite(np.asarray(source.values)))
            ):
                raise ValueError(
                    "Source geometry/measure does not match the physical closure binding."
                )
        for name, value in (
            ("interval_seconds", self.interval_seconds),
            ("resolution_m", self.resolution_m),
        ):
            if not np.all(np.asarray(batch.input(name).values) == value):
                raise ValueError(
                    "Runtime interval/resolution conditioning differs from the binding."
                )
        query = batch.query("interfaces")
        if (
            query.support_id != f"{self.field.support_id}/interior-interfaces"
            or not np.array_equal(
                np.asarray(query.coordinates), np.asarray(edges[1:-1, None])
            )
            or not np.array_equal(
                np.asarray(query.quadrature_weights), np.ones(mass.size - 1)
            )
        ):
            raise ValueError(
                "Flux query must contain only the bound ordered interior interfaces."
            )


@dataclass(frozen=True)
class ConservativeColumnTransfer:
    transfer: FieldTransfer
    source_measure: DiscreteMeasure
    target_measure: DiscreteMeasure
    thermodynamics_id: str
    measure_unit: Any = _MASS_AREA

    def __post_init__(self):
        _check_measure(self.transfer.source, self.source_measure)
        _check_measure(self.transfer.target, self.target_measure)
        if self.measure_unit != _MASS_AREA or not self.thermodynamics_id:
            raise ValueError(
                "Conservative coarsening requires SI mass and caloric reference."
            )
        properties = self.transfer.properties
        if not (
            properties.conservative
            and properties.constant_preserving
            and properties.positivity_preserving
        ):
            raise ValueError(
                "Coarsening requires conservative positive constant-preserving FieldTransfer."
            )
        # Native transpose pairing checks conservation without materializing a matrix.
        source, target = (
            self.transfer.source.vector_space,
            self.transfer.target.vector_space,
        )

        def action(value):
            return target.flatten(
                self.transfer.primal_operator.mv(source.unflatten(value))
            )

        sw, tw = self.source_measure.weights, self.target_measure.weights
        (pulled,) = jax.linear_transpose(action, jnp.zeros_like(sw))(tw)
        tolerance = 64 * np.finfo(np.asarray(sw).dtype).eps
        if not np.allclose(
            pulled, sw, rtol=tolerance, atol=tolerance * float(jnp.max(sw))
        ):
            raise ValueError("FieldTransfer fails physical dry-mass measure pairing.")
        if not np.allclose(
            action(jnp.ones_like(sw)), 1.0, rtol=tolerance, atol=tolerance
        ):
            raise ValueError("FieldTransfer fails its constant-preserving claim.")

    @property
    def transfer_id(self):
        return canonical_fingerprint(
            {
                "transfer": self.transfer.transfer_id,
                "source_measure": self.source_measure.measure_id,
                "target_measure": self.target_measure.measure_id,
                "unit": self.measure_unit.unit_id,
                "thermodynamics": self.thermodynamics_id,
            }
        )

    def inventories(self, values):
        """Restrict specific inventories, then re-extensify; never average masses."""
        array = jnp.asarray(values)
        if array.ndim != 3 or array.shape[-2:] != (self.source_measure.weights.size, 2):
            raise ValueError("Transfer arrays must be (case, layer, vapor/total-energy).")
        if not np.all(np.isfinite(np.asarray(array))) or np.any(
            np.asarray(array[..., 0]) < 0
        ):
            raise ValueError(
                "Coarsened inventories require finite energy and nonnegative vapor."
            )
        source, target = (
            self.transfer.source.vector_space,
            self.transfer.target.vector_space,
        )

        def one(value):
            specific = value / self.source_measure.weights
            return (
                target.flatten(
                    self.transfer.primal_operator.mv(source.unflatten(specific))
                )
                * self.target_measure.weights
            )

        result = jax.vmap(jax.vmap(one, in_axes=1, out_axes=1))(array)
        if np.any(np.asarray(result[..., 0]) < 0):
            raise ValueError(
                "FieldTransfer produced negative vapor; positivity claim failed."
            )
        return result


@dataclass(frozen=True)
class ColumnFluxTarget:
    flux: Any
    combined_increment: Any
    unresolved_increment: Any | None
    numerical_increment: Any | None
    boundary_residual: Any
    binding_id: str
    transfer_id: str
    reference_id: str


def conditional_column_flux_target(
    binding: ColumnFluxBinding,
    transfer: ConservativeColumnTransfer,
    fine_before,
    fine_after,
    coarse_before,
    coarse_after,
    /,
    *,
    interval_bounds,
    fine_reference_after=None,
    coarse_reference_after=None,
    reference_id="",
) -> ColumnFluxTarget:
    """Coarsened fine evolution minus coarse evolution over the SAME interval.

    Optional reference endpoints separate the numerical discrepancy
    (fine-discrete minus fine-reference) - (coarse-discrete minus coarse-reference).
    Without both references, labels are honestly marked combined/unseparated.
    References must represent the same initial states, forcing and interval;
    their scientific construction is caller-owned and identified explicitly.
    """
    if (
        transfer.target_measure.measure_id != binding.measure.measure_id
        or transfer.transfer.target.field_space_id != binding.field.field_space_id
        or transfer.thermodynamics_id != binding.thermodynamics_id
    ):
        raise ValueError("Target binding and conservative transfer identities disagree.")
    initial = transfer.inventories(fine_before)
    final = transfer.inventories(fine_after)
    if final.shape != initial.shape:
        raise ValueError(
            "Fine endpoints must preserve the complete case/layer/quantity shape."
        )
    before, after = np.asarray(coarse_before), np.asarray(coarse_after)
    if before.shape != initial.shape or after.shape != initial.shape:
        raise ValueError(
            "Coarse endpoints must match the transferred case/layer/quantity shape."
        )
    if (
        not np.all(np.isfinite(before))
        or not np.all(np.isfinite(after))
        or np.any(before[..., 0] < 0)
        or np.any(after[..., 0] < 0)
    ):
        raise ValueError("Coarse endpoints contain invalid inventories.")
    tolerance = 128 * np.finfo(np.asarray(initial).dtype).eps
    if not np.allclose(initial, before, rtol=tolerance, atol=0):
        raise ValueError(
            "Coarse evolution must start from the conservatively coarsened fine state."
        )
    bounds = np.asarray(interval_bounds)
    if (
        bounds.shape != (initial.shape[0], 2)
        or not np.all(np.isfinite(bounds))
        or not np.all(bounds[:, 1] - bounds[:, 0] == binding.interval_seconds)
    ):
        raise ValueError(
            "Every fine/coarse interval must match the declared closure seconds."
        )
    combined = final - after
    unresolved = numerical = None
    if (fine_reference_after is None) != (coarse_reference_after is None):
        raise ValueError("Numerical separation requires BOTH reference evolutions.")
    if fine_reference_after is not None:
        if not reference_id:
            raise ValueError(
                "Separated numerical discrepancy requires a reference identity."
            )
        reference = transfer.inventories(fine_reference_after)
        if reference.shape != initial.shape:
            raise ValueError("Fine reference endpoints must preserve every initial case.")
        coarse_reference = np.asarray(coarse_reference_after)
        if (
            coarse_reference.shape != initial.shape
            or not np.all(np.isfinite(coarse_reference))
            or np.any(coarse_reference[..., 0] < 0)
        ):
            raise ValueError(
                "Coarse reference inventories must match the physical endpoints."
            )
        unresolved = reference - coarse_reference
        numerical = combined - unresolved
    elif reference_id:
        raise ValueError(
            "A reference identity without reference trajectories is misleading."
        )
    label = combined if unresolved is None else unresolved
    residual = jnp.sum(label, axis=-2)
    scale = jnp.sum(jnp.abs(initial) + jnp.abs(final) + jnp.abs(after), axis=-2)
    if not np.all(np.asarray(jnp.abs(residual) <= tolerance * scale)):
        raise ValueError(
            "Conditional target changes external inventory; closed flux cannot represent it."
        )
    flux = -jnp.cumsum(label, axis=-2)[..., :-1, :] / binding.interval_seconds
    return ColumnFluxTarget(
        flux,
        combined,
        unresolved,
        numerical,
        residual,
        binding.binding_id,
        transfer.transfer_id,
        reference_id,
    )


def column_flux_datasets(
    binding: ColumnFluxBinding,
    batch: OperatorBatch,
    target: ColumnFluxTarget,
    /,
    *,
    provenance: Sequence[OperatorCaseProvenance],
    interval_bounds,
) -> tuple[OperatorDataset, OperatorDataset]:
    """Attach real support/forcing/interval provenance to native training datasets."""
    count = batch.case_shape[0] if len(batch.case_shape) == 1 else -1
    records, bounds = tuple(provenance), np.asarray(interval_bounds)
    if target.binding_id != binding.binding_id or target.flux.shape != (
        count,
        binding.measure.weights.size - 1,
        2,
    ):
        raise ValueError("Conditional flux target does not belong to this batch binding.")
    if (
        len(records) != count
        or bounds.shape != (count, 2)
        or not np.all(np.isfinite(bounds))
        or not np.all(bounds[:, 1] - bounds[:, 0] == binding.interval_seconds)
    ):
        raise ValueError(
            "Case provenance and interval bounds must exactly match the closure."
        )
    identities = {
        "resolution": str(binding.resolution_m),
        "closure_interval": str(binding.interval_seconds),
        "regime": binding.regime,
        "forcing": binding.forcing_id,
        "closure_binding": binding.binding_id,
        "conservative_transfer": target.transfer_id,
        "thermodynamics": binding.thermodynamics_id,
        "target_separation": "combined_unseparated"
        if target.unresolved_increment is None
        else "unresolved_only",
        "reference": target.reference_id or "none",
    }
    enriched = []
    for record, bound in zip(records, bounds, strict=True):
        if not {"scenario", "model", "member"}.issubset(record.identities):
            raise ValueError(
                "Native simulation provenance requires scenario/model/member."
            )
        if any(
            key in record.identities and record.identities[key] != value
            for key, value in identities.items()
        ):
            raise ValueError("Case provenance contradicts the actual closure support.")
        enriched.append(
            OperatorCaseProvenance(
                record.case_id,
                identities={**record.identities, **identities},
                order={
                    **record.order,
                    "interval_start": float(bound[0]),
                    "interval_end": float(bound[1]),
                },
            )
        )
    datasets = []
    binding.validate_batch(batch)
    for index, (task, name) in enumerate(zip(binding.tasks, _FLUX_NAMES, strict=True)):
        task.validate_batch(batch)
        datasets.append(
            OperatorDataset(
                batch,
                OperatorTargetBatch.from_arrays(
                    {name: target.flux[..., index]},
                    batch,
                    specs={name: task.field_by_name[name].output_spec},
                    query_names={name: "interfaces"},
                ),
                provenance=tuple(enriched),
            )
        )
    return tuple(datasets)


@dataclass(frozen=True)
class ColumnFluxAdmission:
    vapor_mass: Any
    total_energy: Any
    vapor_increment: Any
    energy_increment: Any
    water_flux: Any
    energy_flux: Any
    inventory_residual: Any
    admitted: Any
    binding_id: str
    artifact_ids: tuple[str, str]

    def require_inventories(self):
        if not np.all(np.asarray(self.admitted)):
            raise ValueError(
                "Learned flux failed physical admission; no proposal may be committed."
            )
        return self.vapor_mass, self.total_energy


def admit_column_flux(
    binding: ColumnFluxBinding,
    vapor_mass,
    total_energy,
    water_flux,
    energy_flux,
    /,
    *,
    thermodynamics: MoistThermodynamicPlan,
    dry_mass,
    liquid_mass=0.0,
    ice_mass=0.0,
    artifact_ids: tuple[str, str],
) -> ColumnFluxAdmission:
    """Derive both increments from paired rates and reject invalid donor inventories.

    Liquid includes any suspended/raining liquid and ice includes snow. They do
    NOT supply vapor to this closure. Their energy/heat capacity is nevertheless
    included in temperature admission. Native column step performs final process
    admission when these same interior rates are passed to its explicit hook.
    """
    if thermodynamics.plan_id != binding.thermodynamics_id:
        raise ValueError("Closure caloric reference differs from physical column.")
    if len(artifact_ids) != 2 or not all(artifact_ids):
        raise ValueError(
            "Both trained/untrained experiment artifact identities are required."
        )
    vapor, energy = jnp.asarray(vapor_mass), jnp.asarray(total_energy)
    dry = jnp.broadcast_to(jnp.asarray(dry_mass), vapor.shape)
    liquid = jnp.broadcast_to(jnp.asarray(liquid_mass), vapor.shape)
    ice = jnp.broadcast_to(jnp.asarray(ice_mass), vapor.shape)
    wf, ef = jnp.asarray(water_flux), jnp.asarray(energy_flux)
    if (
        vapor.ndim < 1
        or vapor.shape[-1] != binding.measure.weights.size
        or energy.shape != vapor.shape
        or wf.shape != vapor.shape[:-1] + (vapor.shape[-1] - 1,)
        or ef.shape != wf.shape
    ):
        raise ValueError(
            "Inventories and interior fluxes must preserve case/layer order."
        )
    if not np.array_equal(
        np.asarray(dry), np.broadcast_to(np.asarray(binding.measure.weights), dry.shape)
    ):
        raise ValueError(
            "Actual dry masses differ from the bound physical layer measure."
        )
    zero = jnp.zeros_like(vapor[..., :1])

    def paired(flux):
        return binding.interval_seconds * (
            jnp.concatenate((zero, flux), axis=-1)
            - jnp.concatenate((flux, zero), axis=-1)
        )

    dv, de = paired(wf), paired(ef)
    proposed_vapor, proposed_energy = vapor + dv, energy + de
    donor_out = binding.interval_seconds * (
        jnp.concatenate((jnp.maximum(wf, 0), zero), axis=-1)
        + jnp.concatenate((zero, jnp.maximum(-wf, 0)), axis=-1)
    )

    def valid_inventory(water, internal):
        capacity = (
            dry * thermodynamics.dry_cv
            + water * thermodynamics.vapor_cv
            + liquid * thermodynamics.liquid_heat_capacity
            + ice * thermodynamics.ice_heat_capacity
        )
        reference = thermodynamics.phase_energies(thermodynamics.reference_temperature)
        offset = (
            dry * reference[0]
            + water * reference[1]
            + liquid * reference[2]
            + ice * reference[3]
        )
        temperature = (
            thermodynamics.reference_temperature + (internal - offset) / capacity
        )
        return jnp.all(
            jnp.isfinite(water)
            & (water >= 0)
            & jnp.isfinite(internal)
            & jnp.isfinite(liquid)
            & (liquid >= 0)
            & jnp.isfinite(ice)
            & (ice >= 0)
            & jnp.isfinite(temperature)
            & (temperature >= thermodynamics.minimum_temperature)
            & (temperature <= thermodynamics.maximum_temperature),
            axis=-1,
        )

    admitted = (
        valid_inventory(vapor, energy)
        & valid_inventory(proposed_vapor, proposed_energy)
        & jnp.all(donor_out <= vapor, axis=-1)
        & jnp.all(jnp.isfinite(wf) & jnp.isfinite(ef), axis=-1)
    )
    residual = jnp.stack((jnp.sum(dv, axis=-1), jnp.sum(de, axis=-1)), axis=-1)
    return ColumnFluxAdmission(
        proposed_vapor,
        proposed_energy,
        dv,
        de,
        wf,
        ef,
        residual,
        admitted,
        binding.binding_id,
        tuple(artifact_ids),
    )


def deploy_column_flux(
    trained: tuple[TrainedOperator, TrainedOperator],
    binding: ColumnFluxBinding,
    batch: OperatorBatch,
    /,
    *,
    vapor_mass,
    total_energy,
    thermodynamics,
    dry_mass,
    artifact_ids: tuple[str, str],
    liquid_mass=0.0,
    ice_mass=0.0,
    key=None,
) -> ColumnFluxAdmission:
    """Predict using native restored artifacts; execution has no private state."""
    if len(trained) != 2 or len(artifact_ids) != 2:
        raise ValueError("Deployment requires both native flux artifacts.")
    binding.validate_batch(batch)
    rates = []
    for model, task, identity, name in zip(
        trained, binding.tasks, artifact_ids, _FLUX_NAMES, strict=True
    ):
        if (
            not isinstance(model, TrainedOperator)
            or model.task_fingerprint != task.fingerprint
            or model.artifact_id != identity
        ):
            raise ValueError(
                "Flux artifact/task identity does not match deployment or restart."
            )
        if (
            isinstance(model.execution_model, ExternalOperatorAdapter)
            or "external_manifest" in model.provenance
        ):
            raise TypeError("Flux closure requires native trained-operator execution.")
        if model.output_pipeline is not None:
            raise ValueError(
                "Flux deployment does not permit hidden output projection/correction."
            )
        rates.append(model.predict(batch, key=key).field(name).values)
    return admit_column_flux(
        binding,
        vapor_mass,
        total_energy,
        *rates,
        thermodynamics=thermodynamics,
        dry_mass=dry_mass,
        liquid_mass=liquid_mass,
        ice_mass=ice_mass,
        artifact_ids=artifact_ids,
    )


__all__ = [
    "ColumnFluxAdmission",
    "ColumnFluxBinding",
    "ColumnFluxTarget",
    "ConservativeColumnTransfer",
    "admit_column_flux",
    "column_flux_datasets",
    "column_flux_quantities",
    "column_flux_space",
    "column_flux_tasks",
    "conditional_column_flux_target",
    "deploy_column_flux",
]
