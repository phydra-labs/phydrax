#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class KineticFieldRole(StrEnum):
    STREAMED_POPULATION = "streamed_population"
    KINETIC_POPULATION = "kinetic_population"
    AUXILIARY_POPULATION = "auxiliary_population"
    MACROSCOPIC = "macroscopic"
    BOUNDARY_HISTORY = "boundary_history"
    GEOMETRY = "geometry"
    LEDGER = "ledger"
    DIAGNOSTIC = "diagnostic"
    COUPLED_STATE = "coupled_state"
    SOURCE = "source"


class KineticFailureScope(StrEnum):
    LOCAL = "local"
    GLOBAL = "global"
    ATOMIC = "atomic"


class KineticFieldSpec(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    role: KineticFieldRole = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    lattice_id: str | None = eqx.field(static=True)
    precision_role: str = eqx.field(static=True)
    units: str = eqx.field(static=True)
    conserved_channels: tuple[str, ...] = eqx.field(static=True)
    halo_width: int = eqx.field(static=True)
    initialized: bool = eqx.field(static=True)
    checkpoint_required: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        role: KineticFieldRole,
        component_shape: Sequence[int] = (),
        /,
        *,
        lattice_id: str | None = None,
        precision_role: str = "compute",
        units: str = "lattice",
        conserved_channels: Sequence[str] = (),
        halo_width: int = 0,
        initialized: bool = False,
        checkpoint_required: bool = False,
        differentiable: bool = True,
    ):
        identifier = str(name)
        shape = tuple(int(value) for value in component_shape)
        precision = str(precision_role)
        units_ = str(units)
        channels = tuple(str(value) for value in conserved_channels)
        width = int(halo_width)
        lattice = None if lattice_id is None else str(lattice_id)
        if not identifier or not precision or not units_:
            raise ValueError(
                "Kinetic field names, precision roles, and units must be nonempty."
            )
        if not isinstance(role, KineticFieldRole):
            raise TypeError("role must be KineticFieldRole.")
        if any(value <= 0 for value in shape) or width < 0:
            raise ValueError(
                "Kinetic component extents must be positive and halo width nonnegative."
            )
        if len(set(channels)) != len(channels) or any(not value for value in channels):
            raise ValueError("Conserved channel names must be unique and nonempty.")
        if role in (
            KineticFieldRole.STREAMED_POPULATION,
            KineticFieldRole.KINETIC_POPULATION,
            KineticFieldRole.AUXILIARY_POPULATION,
        ) and (not shape or lattice is None):
            raise ValueError("Population fields require component_shape and lattice_id.")
        if width and role not in (
            KineticFieldRole.STREAMED_POPULATION,
            KineticFieldRole.KINETIC_POPULATION,
            KineticFieldRole.AUXILIARY_POPULATION,
            KineticFieldRole.MACROSCOPIC,
        ):
            raise ValueError("Only population or macroscopic fields may require halos.")
        self.name = identifier
        self.role = role
        self.component_shape = shape
        self.lattice_id = lattice
        self.precision_role = precision
        self.units = units_
        self.conserved_channels = channels
        self.halo_width = width
        self.initialized = bool(initialized)
        self.checkpoint_required = bool(checkpoint_required)
        self.differentiable = bool(differentiable)
        self.spec_id = canonical_fingerprint(
            {
                "kind": "kinetic-field-spec",
                "name": identifier,
                "role": role.value,
                "component_shape": shape,
                "lattice": lattice,
                "precision_role": precision,
                "units": units_,
                "conserved_channels": channels,
                "halo_width": width,
                "initialized": self.initialized,
                "checkpoint_required": self.checkpoint_required,
                "differentiable": self.differentiable,
            }
        )


class KineticStageSpec(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    order: int = eqx.field(static=True)
    reads: tuple[str, ...] = eqx.field(static=True)
    writes: tuple[str, ...] = eqx.field(static=True)
    exchange_fields: tuple[str, ...] = eqx.field(static=True)
    reductions: tuple[str, ...] = eqx.field(static=True)
    conservation_channels: tuple[str, ...] = eqx.field(static=True)
    failure_scope: KineticFailureScope = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        order: int,
        /,
        *,
        reads: Sequence[str] = (),
        writes: Sequence[str] = (),
        exchange_fields: Sequence[str] = (),
        reductions: Sequence[str] = (),
        conservation_channels: Sequence[str] = (),
        failure_scope: KineticFailureScope = KineticFailureScope.ATOMIC,
    ):
        identifier = str(name)
        order_ = int(order)
        read_names = tuple(str(value) for value in reads)
        write_names = tuple(str(value) for value in writes)
        exchange = tuple(str(value) for value in exchange_fields)
        reductions_ = tuple(str(value) for value in reductions)
        channels = tuple(str(value) for value in conservation_channels)
        groups = (read_names, write_names, exchange, reductions_, channels)
        if not identifier or order_ < 0:
            raise ValueError("Kinetic stage name and order are invalid.")
        if any(
            len(set(group)) != len(group) or any(not value for value in group)
            for group in groups
        ):
            raise ValueError("Kinetic stage declarations must be unique and nonempty.")
        if not isinstance(failure_scope, KineticFailureScope):
            raise TypeError("failure_scope must be KineticFailureScope.")
        if any(value not in read_names for value in exchange):
            raise ValueError("Every exchanged field must also be read by the stage.")
        self.name = identifier
        self.order = order_
        self.reads = read_names
        self.writes = write_names
        self.exchange_fields = exchange
        self.reductions = reductions_
        self.conservation_channels = channels
        self.failure_scope = failure_scope
        self.stage_id = canonical_fingerprint(
            {
                "kind": "kinetic-stage-spec",
                "name": identifier,
                "order": order_,
                "reads": read_names,
                "writes": write_names,
                "exchange_fields": exchange,
                "reductions": reductions_,
                "conservation_channels": channels,
                "failure_scope": failure_scope.value,
            }
        )


class KineticProgramManifest(StrictModule, NonTrainableState):
    fields: tuple[KineticFieldSpec, ...]
    stages: tuple[KineticStageSpec, ...]
    program_kind: str = eqx.field(static=True)
    lattice_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    checkpoint_fields: tuple[str, ...] = eqx.field(static=True)
    dependency_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        program_kind: str,
        lattice_id: str,
        precision_policy_id: str,
        fields: Sequence[KineticFieldSpec],
        stages: Sequence[KineticStageSpec],
        /,
        *,
        dependency_manifest_ids: Sequence[str] = (),
    ):
        kind = str(program_kind)
        lattice = str(lattice_id)
        precision = str(precision_policy_id)
        fields_ = tuple(fields)
        stages_ = tuple(stages)
        dependencies = tuple(str(value) for value in dependency_manifest_ids)
        if not kind or not lattice or not precision or not fields_ or not stages_:
            raise ValueError(
                "Kinetic manifest identity, fields, and stages must be nonempty."
            )
        if any(not isinstance(value, KineticFieldSpec) for value in fields_):
            raise TypeError("fields must contain KineticFieldSpec values.")
        if any(not isinstance(value, KineticStageSpec) for value in stages_):
            raise TypeError("stages must contain KineticStageSpec values.")
        if len(set(dependencies)) != len(dependencies) or any(
            not value for value in dependencies
        ):
            raise ValueError(
                "Dependency manifest identities must be unique and nonempty."
            )
        field_names = tuple(value.name for value in fields_)
        if len(set(field_names)) != len(field_names):
            raise ValueError("Kinetic field names must be unique.")
        stage_names = tuple(value.name for value in stages_)
        orders = tuple(value.order for value in stages_)
        if len(set(stage_names)) != len(stage_names) or len(set(orders)) != len(orders):
            raise ValueError("Kinetic stage names and orders must be unique.")
        if orders != tuple(sorted(orders)):
            raise ValueError("Kinetic stages must be supplied in increasing order.")
        table = {value.name: value for value in fields_}
        available = {value.name for value in fields_ if value.initialized}
        writers: set[tuple[int, str]] = set()
        for stage in stages_:
            unknown = set(stage.reads + stage.writes) - set(field_names)
            if unknown:
                raise ValueError(
                    f"Kinetic stage {stage.name!r} names unknown fields {sorted(unknown)}."
                )
            unavailable = set(stage.reads) - available
            if unavailable:
                raise ValueError(
                    f"Kinetic stage {stage.name!r} reads unavailable fields {sorted(unavailable)}."
                )
            for field in stage.exchange_fields:
                if table[field].halo_width <= 0:
                    raise ValueError(
                        f"Exchanged field {field!r} must declare a positive halo width."
                    )
            for field in stage.writes:
                key = (stage.order, field)
                if key in writers:
                    raise ValueError(
                        "A kinetic stage order may write each field only once."
                    )
                writers.add(key)
                available.add(field)
        checkpoint_fields = tuple(
            value.name for value in fields_ if value.checkpoint_required
        )
        self.fields = fields_
        self.stages = stages_
        self.program_kind = kind
        self.lattice_id = lattice
        self.precision_policy_id = precision
        self.field_names = field_names
        self.checkpoint_fields = checkpoint_fields
        self.dependency_manifest_ids = dependencies
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "kinetic-program-manifest",
                "program_kind": kind,
                "lattice": lattice,
                "precision": precision,
                "fields": [value.spec_id for value in fields_],
                "stages": [value.stage_id for value in stages_],
                "checkpoint_fields": checkpoint_fields,
                "dependency_manifest_ids": dependencies,
            }
        )

    def field(self, name: str, /) -> KineticFieldSpec:
        identifier = str(name)
        for value in self.fields:
            if value.name == identifier:
                return value
        raise ValueError(f"Kinetic manifest has no field {identifier!r}.")


def athermal_lattice_boltzmann_manifest(
    lattice_id: str,
    precision_policy_id: str,
    population_count: int,
    dimension: int,
    /,
) -> KineticProgramManifest:
    fields = (
        KineticFieldSpec(
            "populations",
            KineticFieldRole.STREAMED_POPULATION,
            (population_count,),
            lattice_id=lattice_id,
            precision_role="population",
            conserved_channels=("mass", "momentum"),
            halo_width=1,
            initialized=True,
            checkpoint_required=True,
        ),
        KineticFieldSpec(
            "geometry", KineticFieldRole.GEOMETRY, initialized=True, differentiable=False
        ),
        KineticFieldSpec("density", KineticFieldRole.MACROSCOPIC),
        KineticFieldSpec("velocity", KineticFieldRole.MACROSCOPIC, (dimension,)),
        KineticFieldSpec("force_density", KineticFieldRole.SOURCE, (dimension,)),
        KineticFieldSpec(
            "post_collision",
            KineticFieldRole.AUXILIARY_POPULATION,
            (population_count,),
            lattice_id=lattice_id,
            precision_role="compute",
            conserved_channels=("mass", "momentum"),
            halo_width=1,
        ),
        KineticFieldSpec(
            "streamed",
            KineticFieldRole.AUXILIARY_POPULATION,
            (population_count,),
            lattice_id=lattice_id,
            precision_role="population",
        ),
        KineticFieldSpec("wall_ledger", KineticFieldRole.LEDGER),
        KineticFieldSpec("diagnostics", KineticFieldRole.DIAGNOSTIC),
    )
    stages = (
        KineticStageSpec(
            "macroscopic",
            0,
            reads=("populations", "geometry"),
            writes=("density", "velocity", "force_density"),
        ),
        KineticStageSpec(
            "collision",
            1,
            reads=("populations", "density", "velocity", "force_density"),
            writes=("post_collision",),
            conservation_channels=("mass", "momentum"),
        ),
        KineticStageSpec(
            "population_exchange_stream",
            2,
            reads=("post_collision", "geometry"),
            writes=("streamed",),
            exchange_fields=("post_collision",),
            failure_scope=KineticFailureScope.GLOBAL,
        ),
        KineticStageSpec(
            "boundary",
            3,
            reads=("streamed", "density", "geometry"),
            writes=("populations", "wall_ledger"),
            conservation_channels=("mass", "momentum"),
        ),
        KineticStageSpec(
            "candidate_macroscopic",
            4,
            reads=("populations", "geometry"),
            writes=("density", "velocity", "force_density"),
        ),
        KineticStageSpec(
            "diagnostics",
            5,
            reads=("populations", "density", "velocity", "wall_ledger"),
            writes=("diagnostics",),
            reductions=("mass", "momentum", "minimum_population"),
            failure_scope=KineticFailureScope.GLOBAL,
        ),
    )
    return KineticProgramManifest(
        "athermal_lattice_boltzmann",
        lattice_id,
        precision_policy_id,
        fields,
        stages,
    )


def coupled_population_manifest(
    program_kind: str,
    lattice_id: str,
    precision_policy_id: str,
    population_count: int,
    dimension: int,
    population_names: Sequence[str],
    conservation_channels: Sequence[Sequence[str]],
    /,
) -> KineticProgramManifest:
    names = tuple(str(value) for value in population_names)
    channels = tuple(
        tuple(str(item) for item in group) for group in conservation_channels
    )
    if not names or len(names) != len(channels) or len(set(names)) != len(names):
        raise ValueError(
            "Coupled population names and conservation channels are invalid."
        )
    population_fields = tuple(
        KineticFieldSpec(
            name,
            KineticFieldRole.STREAMED_POPULATION
            if index == 0
            else KineticFieldRole.AUXILIARY_POPULATION,
            (population_count,),
            lattice_id=lattice_id,
            precision_role="population",
            conserved_channels=channels[index],
            halo_width=1,
            initialized=True,
            checkpoint_required=True,
        )
        for index, name in enumerate(names)
    )
    post_names = tuple(f"{name}_post_collision" for name in names)
    streamed_names = tuple(f"{name}_streamed" for name in names)
    post_fields = tuple(
        KineticFieldSpec(
            name,
            KineticFieldRole.AUXILIARY_POPULATION,
            (population_count,),
            lattice_id=lattice_id,
            precision_role="compute",
            halo_width=1,
        )
        for name in post_names
    )
    streamed_fields = tuple(
        KineticFieldSpec(
            name,
            KineticFieldRole.AUXILIARY_POPULATION,
            (population_count,),
            lattice_id=lattice_id,
            precision_role="population",
        )
        for name in streamed_names
    )
    fields = (
        *population_fields,
        *post_fields,
        *streamed_fields,
        KineticFieldSpec(
            "geometry", KineticFieldRole.GEOMETRY, initialized=True, differentiable=False
        ),
        KineticFieldSpec("density", KineticFieldRole.MACROSCOPIC),
        KineticFieldSpec("velocity", KineticFieldRole.MACROSCOPIC, (dimension,)),
        KineticFieldSpec("coupling_force", KineticFieldRole.SOURCE, (dimension,)),
        KineticFieldSpec("ledger", KineticFieldRole.LEDGER),
        KineticFieldSpec("diagnostics", KineticFieldRole.DIAGNOSTIC),
    )
    stages = (
        KineticStageSpec(
            "macroscopic_coupling",
            0,
            reads=(*names, "geometry"),
            writes=("density", "velocity", "coupling_force", "ledger"),
        ),
        KineticStageSpec(
            "coupled_collision",
            1,
            reads=(*names, "density", "velocity", "coupling_force"),
            writes=post_names,
            conservation_channels=tuple(
                dict.fromkeys(item for group in channels for item in group)
            ),
        ),
        KineticStageSpec(
            "population_exchange_stream",
            2,
            reads=(*post_names, "geometry"),
            writes=streamed_names,
            exchange_fields=post_names,
            failure_scope=KineticFailureScope.GLOBAL,
        ),
        KineticStageSpec(
            "boundary_commit",
            3,
            reads=(*streamed_names, "geometry", "ledger"),
            writes=(*names, "ledger"),
            conservation_channels=tuple(
                dict.fromkeys(item for group in channels for item in group)
            ),
        ),
        KineticStageSpec(
            "candidate_macroscopic_coupling",
            4,
            reads=(*names, "geometry"),
            writes=("density", "velocity", "coupling_force", "ledger"),
        ),
        KineticStageSpec(
            "diagnostics",
            5,
            reads=(*names, "density", "velocity", "ledger"),
            writes=("diagnostics",),
            reductions=("conservation", "minimum_population"),
            failure_scope=KineticFailureScope.GLOBAL,
        ),
    )
    return KineticProgramManifest(
        program_kind,
        lattice_id,
        precision_policy_id,
        fields,
        stages,
    )


def transport_population_manifest(
    program_kind: str,
    lattice_id: str,
    precision_policy_id: str,
    population_name: str,
    component_shape: Sequence[int],
    conservation_channels: Sequence[str],
    /,
    *,
    dimension: int,
    source_component_shape: Sequence[int] = (),
) -> KineticProgramManifest:
    population = str(population_name)
    post = f"{population}_post_collision"
    streamed = f"{population}_streamed"
    fields = (
        KineticFieldSpec(
            population,
            KineticFieldRole.STREAMED_POPULATION,
            component_shape,
            lattice_id=lattice_id,
            precision_role="population",
            conserved_channels=conservation_channels,
            halo_width=1,
            initialized=True,
            checkpoint_required=True,
        ),
        KineticFieldSpec(
            post,
            KineticFieldRole.AUXILIARY_POPULATION,
            component_shape,
            lattice_id=lattice_id,
            precision_role="compute",
            halo_width=1,
        ),
        KineticFieldSpec(
            streamed,
            KineticFieldRole.AUXILIARY_POPULATION,
            component_shape,
            lattice_id=lattice_id,
            precision_role="population",
        ),
        KineticFieldSpec(
            "velocity",
            KineticFieldRole.MACROSCOPIC,
            (int(dimension),),
            initialized=True,
        ),
        KineticFieldSpec(
            "source",
            KineticFieldRole.SOURCE,
            source_component_shape,
            initialized=True,
        ),
        KineticFieldSpec(
            "geometry",
            KineticFieldRole.GEOMETRY,
            initialized=True,
            differentiable=False,
        ),
        KineticFieldSpec("ledger", KineticFieldRole.LEDGER),
        KineticFieldSpec("diagnostics", KineticFieldRole.DIAGNOSTIC),
    )
    stages = (
        KineticStageSpec(
            "collision",
            0,
            reads=(population, "velocity", "source"),
            writes=(post, "ledger"),
            conservation_channels=conservation_channels,
        ),
        KineticStageSpec(
            "population_exchange_stream",
            1,
            reads=(post, "geometry"),
            writes=(streamed,),
            exchange_fields=(post,),
            failure_scope=KineticFailureScope.GLOBAL,
        ),
        KineticStageSpec(
            "boundary_ledger_commit",
            2,
            reads=(streamed, "geometry", "ledger"),
            writes=(population, "ledger"),
            conservation_channels=conservation_channels,
        ),
        KineticStageSpec(
            "diagnostics",
            3,
            reads=(population, "ledger"),
            writes=("diagnostics",),
            reductions=("conservation", "minimum_population"),
            failure_scope=KineticFailureScope.GLOBAL,
        ),
    )
    return KineticProgramManifest(
        program_kind,
        lattice_id,
        precision_policy_id,
        fields,
        stages,
    )


def smooth_compressible_dvm_manifest(
    quadrature_id: str,
    precision_policy_id: str,
    population_count: int,
    dimension: int,
    /,
) -> KineticProgramManifest:
    """Describe local two-population compressible DVM collision semantics."""

    population_shape = (int(population_count),)
    fields = (
        KineticFieldSpec(
            "particle_populations",
            KineticFieldRole.KINETIC_POPULATION,
            population_shape,
            lattice_id=quadrature_id,
            precision_role="population",
            conserved_channels=("mass", "momentum"),
            initialized=True,
            checkpoint_required=True,
        ),
        KineticFieldSpec(
            "total_energy_populations",
            KineticFieldRole.KINETIC_POPULATION,
            population_shape,
            lattice_id=quadrature_id,
            precision_role="population",
            conserved_channels=("total_energy",),
            initialized=True,
            checkpoint_required=True,
        ),
        KineticFieldSpec(
            "conserved",
            KineticFieldRole.MACROSCOPIC,
            (int(dimension) + 2,),
        ),
        KineticFieldSpec(
            "particle_equilibrium",
            KineticFieldRole.AUXILIARY_POPULATION,
            population_shape,
            lattice_id=quadrature_id,
            precision_role="compute",
        ),
        KineticFieldSpec(
            "total_energy_equilibrium",
            KineticFieldRole.AUXILIARY_POPULATION,
            population_shape,
            lattice_id=quadrature_id,
            precision_role="compute",
        ),
        KineticFieldSpec("diagnostics", KineticFieldRole.DIAGNOSTIC),
    )
    stages = (
        KineticStageSpec(
            "moments",
            0,
            reads=("particle_populations", "total_energy_populations"),
            writes=("conserved",),
        ),
        KineticStageSpec(
            "equilibrium",
            1,
            reads=("conserved",),
            writes=("particle_equilibrium", "total_energy_equilibrium"),
        ),
        KineticStageSpec(
            "collision",
            2,
            reads=(
                "particle_populations",
                "total_energy_populations",
                "particle_equilibrium",
                "total_energy_equilibrium",
            ),
            writes=("particle_populations", "total_energy_populations"),
            conservation_channels=("mass", "momentum", "total_energy"),
        ),
        KineticStageSpec(
            "diagnostics",
            3,
            reads=(
                "particle_populations",
                "total_energy_populations",
                "conserved",
            ),
            writes=("diagnostics",),
            reductions=("realizability", "conservation_residual"),
            failure_scope=KineticFailureScope.GLOBAL,
        ),
    )
    return KineticProgramManifest(
        "smooth_compressible_discrete_velocity",
        quadrature_id,
        precision_policy_id,
        fields,
        stages,
    )


def finite_volume_dvm_manifest(
    quadrature_id: str,
    precision_policy_id: str,
    population_count: int,
    conservation_channels: Sequence[str],
    /,
    *,
    has_source: bool,
) -> KineticProgramManifest:
    """Describe finite-volume kinetic residual evaluation without lattice streaming."""

    population_shape = (int(population_count),)
    fields = (
        KineticFieldSpec(
            "dvm_populations",
            KineticFieldRole.KINETIC_POPULATION,
            population_shape,
            lattice_id=quadrature_id,
            precision_role="population",
            conserved_channels=conservation_channels,
            initialized=True,
            checkpoint_required=True,
        ),
        KineticFieldSpec(
            "geometry",
            KineticFieldRole.GEOMETRY,
            initialized=True,
            differentiable=False,
        ),
        KineticFieldSpec(
            "face_traces",
            KineticFieldRole.AUXILIARY_POPULATION,
            population_shape,
            lattice_id=quadrature_id,
            precision_role="compute",
        ),
        KineticFieldSpec(
            "flux_divergence",
            KineticFieldRole.SOURCE,
            population_shape,
        ),
        KineticFieldSpec(
            "source_term",
            KineticFieldRole.SOURCE,
            population_shape,
        ),
        KineticFieldSpec(
            "residual",
            KineticFieldRole.SOURCE,
            population_shape,
        ),
        KineticFieldSpec("diagnostics", KineticFieldRole.DIAGNOSTIC),
    )
    source_reads = ("dvm_populations", "geometry") if has_source else ()
    stages = (
        KineticStageSpec(
            "face_reconstruction",
            0,
            reads=("dvm_populations", "geometry"),
            writes=("face_traces",),
        ),
        KineticStageSpec(
            "numerical_flux_divergence",
            1,
            reads=("face_traces", "geometry"),
            writes=("flux_divergence",),
            conservation_channels=conservation_channels,
        ),
        KineticStageSpec(
            "source_evaluation" if has_source else "zero_source",
            2,
            reads=source_reads,
            writes=("source_term",),
            conservation_channels=conservation_channels,
        ),
        KineticStageSpec(
            "residual_assembly",
            3,
            reads=("flux_divergence", "source_term"),
            writes=("residual",),
            conservation_channels=conservation_channels,
        ),
        KineticStageSpec(
            "diagnostics",
            4,
            reads=("dvm_populations", "residual"),
            writes=("diagnostics",),
            reductions=("population_conservation", "declared_moment_conservation"),
            failure_scope=KineticFailureScope.GLOBAL,
        ),
    )
    return KineticProgramManifest(
        "finite_volume_discrete_velocity",
        quadrature_id,
        precision_policy_id,
        fields,
        stages,
    )


def reactive_transport_manifest(
    lattice_id: str,
    precision_policy_id: str,
    thermal_manifest_id: str,
    species_manifest_id: str,
    /,
) -> KineticProgramManifest:
    dependencies = tuple(
        str(value) for value in (thermal_manifest_id, species_manifest_id)
    )
    fields = (
        KineticFieldSpec(
            "thermal_state",
            KineticFieldRole.COUPLED_STATE,
            initialized=True,
            checkpoint_required=True,
        ),
        KineticFieldSpec(
            "species_state",
            KineticFieldRole.COUPLED_STATE,
            initialized=True,
            checkpoint_required=True,
        ),
        KineticFieldSpec(
            "reaction_extent",
            KineticFieldRole.COUPLED_STATE,
            initialized=True,
            checkpoint_required=True,
        ),
        KineticFieldSpec(
            "element_inventory",
            KineticFieldRole.LEDGER,
            initialized=True,
        ),
        KineticFieldSpec("thermal_half", KineticFieldRole.COUPLED_STATE),
        KineticFieldSpec("species_half", KineticFieldRole.COUPLED_STATE),
        KineticFieldSpec("extent_half", KineticFieldRole.COUPLED_STATE),
        KineticFieldSpec("thermal_transport", KineticFieldRole.COUPLED_STATE),
        KineticFieldSpec("species_transport", KineticFieldRole.COUPLED_STATE),
        KineticFieldSpec("diagnostics", KineticFieldRole.DIAGNOSTIC),
    )
    stages = (
        KineticStageSpec(
            "reaction_half_before_transport",
            0,
            reads=("thermal_state", "species_state", "reaction_extent"),
            writes=("thermal_half", "species_half", "extent_half"),
            conservation_channels=("element_amount", "energy"),
        ),
        KineticStageSpec(
            "thermal_species_transport",
            1,
            reads=("thermal_half", "species_half"),
            writes=("thermal_transport", "species_transport"),
            conservation_channels=("species_amount", "energy"),
        ),
        KineticStageSpec(
            "reaction_half_after_transport",
            2,
            reads=("thermal_transport", "species_transport", "extent_half"),
            writes=("thermal_state", "species_state", "reaction_extent"),
            conservation_channels=("element_amount", "energy"),
        ),
        KineticStageSpec(
            "diagnostics",
            3,
            reads=(
                "thermal_state",
                "species_state",
                "reaction_extent",
                "element_inventory",
            ),
            writes=("diagnostics",),
            reductions=("element_residual", "energy_residual"),
            failure_scope=KineticFailureScope.ATOMIC,
        ),
    )
    return KineticProgramManifest(
        "reactive_thermal_species_lattice_boltzmann",
        lattice_id,
        precision_policy_id,
        fields,
        stages,
        dependency_manifest_ids=dependencies,
    )


__all__ = [
    "KineticFailureScope",
    "KineticFieldRole",
    "KineticFieldSpec",
    "KineticProgramManifest",
    "KineticStageSpec",
    "athermal_lattice_boltzmann_manifest",
    "coupled_population_manifest",
    "finite_volume_dvm_manifest",
    "smooth_compressible_dvm_manifest",
    "reactive_transport_manifest",
    "transport_population_manifest",
]
