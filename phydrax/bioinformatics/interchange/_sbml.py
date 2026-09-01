#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...interchange import AdapterError, AdapterLoss, AdapterReport, AdapterStatus
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..systems._kinetics import KineticReaction, KineticReactionSystem, RateLawKind
from ..systems._network import (
    ChemicalComposition,
    Compartment,
    GeneReactionRule,
    Reaction,
    Species,
    StoichiometricNetwork,
    SUBSTANCE,
    SUBSTANCE_FLUX,
    TIME,
    UnitDimension,
    VOLUME,
)


class SBMLSemanticStatus(IntEnum):
    """Semantic validation/lowering status aligned with adapter outcomes."""

    SUCCESS = 0
    UNSUPPORTED_LEVEL_VERSION = 1
    UNSUPPORTED_PACKAGE = 2
    UNSUPPORTED_RULE = 3
    UNSUPPORTED_EVENT = 4
    UNSUPPORTED_KINETIC_LAW = 5
    INCONSISTENT_REFERENCE = 6
    INCONSISTENT_UNIT = 7
    INCOMPLETE_KINETICS = 8
    MALFORMED_MODEL = 9


@dataclass(frozen=True, slots=True)
class SBMLPackageDeclaration:
    """Host-only package declaration from a parsed SBML document."""

    name: str
    version: int
    required: bool = True

    def __post_init__(self):
        if not self.name.strip() or self.version < 1:
            raise ValueError("SBML package name and version must be valid.")
        object.__setattr__(self, "name", self.name.strip().lower())


@dataclass(frozen=True, slots=True)
class SBMLUnitDefinitionAST:
    """Host-only normalized unit definition; scale is base-10 and offset is explicit."""

    unit_id: str
    dimension: UnitDimension
    multiplier: float = 1.0
    scale: int = 0
    offset: float = 0.0

    def __post_init__(self):
        if not self.unit_id.strip() or not isinstance(self.dimension, UnitDimension):
            raise ValueError("SBML unit definitions require an id and UnitDimension.")
        if (
            not isfinite(self.multiplier)
            or self.multiplier == 0.0
            or not isfinite(self.offset)
        ):
            raise ValueError(
                "SBML unit multiplier/offset must be finite and multiplier non-zero."
            )
        object.__setattr__(self, "unit_id", self.unit_id.strip())
        object.__setattr__(self, "scale", int(self.scale))


@dataclass(frozen=True, slots=True)
class SBMLCompartmentAST:
    """Host-only SBML-like compartment node."""

    compartment_id: str
    size: float = 1.0
    units: str = "volume"
    spatial_dimensions: int = 3
    constant: bool = True


@dataclass(frozen=True, slots=True)
class SBMLSpeciesAST:
    """Host-only SBML-like species node with optional exact composition."""

    species_id: str
    compartment: str
    initial_amount: float = 0.0
    substance_units: str = "substance"
    elements: tuple[tuple[str, int], ...] | None = None
    charge: int = 0
    boundary_condition: bool = False
    constant: bool = False


@dataclass(frozen=True, slots=True)
class SBMLSpeciesReferenceAST:
    """Host-only reaction species reference with a constant stoichiometry."""

    species: str
    stoichiometry: float = 1.0


@dataclass(frozen=True, slots=True)
class SBMLKineticLawAST:
    """Host-only supported kinetic-law node without expression evaluation."""

    kind: str
    parameters: tuple[float, ...]
    reactant_orders: tuple[tuple[str, float], ...] = ()
    product_orders: tuple[tuple[str, float], ...] = ()
    rate_units: str = "substance_per_time"


@dataclass(frozen=True, slots=True)
class SBMLReactionAST:
    """Host-only SBML/FBC-like reaction node."""

    reaction_id: str
    reactants: tuple[SBMLSpeciesReferenceAST, ...]
    products: tuple[SBMLSpeciesReferenceAST, ...]
    reversible: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None
    objective_coefficient: float = 0.0
    flux_units: str = "substance_per_time"
    gpr_clauses: tuple[tuple[str, ...], ...] = ()
    exchange: bool = False
    kinetic_law: SBMLKineticLawAST | None = None


@dataclass(frozen=True, slots=True)
class SBMLRuleAST:
    """Host-only rule node retained solely for explicit semantic rejection."""

    rule_id: str
    kind: str
    target: str | None = None
    expression: str = ""


@dataclass(frozen=True, slots=True)
class SBMLEventAST:
    """Host-only event node retained solely for explicit semantic rejection."""

    event_id: str
    trigger: str
    assignments: tuple[tuple[str, str], ...]
    delay: str | None = None
    priority: str | None = None


@dataclass(frozen=True, slots=True)
class SBMLModelAST:
    """Host-only, parser-neutral SBML-like model tree."""

    model_id: str
    compartments: tuple[SBMLCompartmentAST, ...]
    species: tuple[SBMLSpeciesAST, ...]
    reactions: tuple[SBMLReactionAST, ...]
    unit_definitions: tuple[SBMLUnitDefinitionAST, ...] = ()
    rules: tuple[SBMLRuleAST, ...] = ()
    events: tuple[SBMLEventAST, ...] = ()


@dataclass(frozen=True, slots=True)
class SBMLDocumentAST:
    """Host-only document root. It is deliberately not a JAX PyTree."""

    level: int
    version: int
    model: SBMLModelAST
    packages: tuple[SBMLPackageDeclaration, ...] = ()
    source_id: str = "in-memory-sbml"


@dataclass(frozen=True, slots=True)
class SBMLSemanticProfile:
    """One explicit supported Level/Version/package semantic row."""

    level: int
    version: int
    supported_packages: tuple[tuple[str, int], ...]
    supports_core_reactions: bool
    supports_fbc_bounds_objectives_gpr: bool
    supports_kinetic_laws: bool
    supports_rules: bool
    supports_events: bool

    @property
    def profile_id(self) -> str:
        return f"SBML-L{self.level}V{self.version}"


SBML_SEMANTIC_MATRIX = (
    SBMLSemanticProfile(2, 4, (), True, False, True, False, False),
    SBMLSemanticProfile(3, 1, (("fbc", 2),), True, True, True, False, False),
    SBMLSemanticProfile(3, 2, (("fbc", 2), ("fbc", 3)), True, True, True, False, False),
)


class SBMLSemanticEvidence(StrictModule):
    """Explicit selected profile, supported semantics, and rejected paths."""

    rejected_count: Array
    level: int = eqx.field(static=True)
    version: int = eqx.field(static=True)
    packages: tuple[tuple[str, int], ...] = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    supported_semantics: tuple[str, ...] = eqx.field(static=True)
    rejected_paths: tuple[str, ...] = eqx.field(static=True)
    rejection_reasons: tuple[str, ...] = eqx.field(static=True)
    checked_before_lowering: bool = eqx.field(static=True)


class SBMLValidationResult(StrictModule):
    """Host-AST semantic validation result produced before native lowering."""

    valid: Array
    status: Array
    evidence: SBMLSemanticEvidence
    report: AdapterReport
    method_contract: BioinformaticsMethodContract
    source_id: str = eqx.field(static=True)


class SBMLLoweringEvidence(StrictModule):
    """Lossless native identities and unit-scale evidence for one lowering."""

    compartment_scales: Array
    species_scales: Array
    reaction_scales: Array
    validation: SBMLValidationResult
    network_id: str = eqx.field(static=True)
    kinetics_id: str | None = eqx.field(static=True)
    lossless: bool = eqx.field(static=True)


class SBMLLoweringResult(StrictModule):
    """Native network/kinetics lowering with validation and adapter evidence."""

    valid: Array
    status: Array
    network: StoichiometricNetwork | None
    kinetics: KineticReactionSystem | None
    evidence: SBMLLoweringEvidence
    report: AdapterReport
    method_contract: BioinformaticsMethodContract


class SBMLSemanticError(AdapterError):
    """Pre-lowering rejection carrying the complete semantic validation result."""

    validation: SBMLValidationResult

    def __init__(self, validation: SBMLValidationResult, message: str, /):
        self.validation = validation
        super().__init__(AdapterStatus(validation.report.status), message)


def _method_contract(method_name: str, /) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        method_name,
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Unit multipliers and initial values are lowered in floating point; semantic "
            "support decisions are exact over the declared matrix."
        ),
        truncation_statement="No unsupported rule, event, package, or expression is dropped.",
        capacity_semantics="The finite host AST is traversed completely before lowering.",
        assumptions=(
            "The host AST has already resolved XML syntax and identifier lexical rules.",
        ),
        nondifferentiable_outputs=("status", "valid", "adapter report"),
    )


def _profile(level: int, version: int, /) -> SBMLSemanticProfile | None:
    for profile in SBML_SEMANTIC_MATRIX:
        if profile.level == level and profile.version == version:
            return profile
    return None


def _unit_table(model: SBMLModelAST, /) -> dict[str, SBMLUnitDefinitionAST]:
    builtins = {
        "substance": SBMLUnitDefinitionAST("substance", SUBSTANCE),
        "mole": SBMLUnitDefinitionAST("mole", SUBSTANCE),
        "volume": SBMLUnitDefinitionAST("volume", VOLUME),
        "litre": SBMLUnitDefinitionAST("litre", VOLUME, multiplier=1.0e-3),
        "substance_per_time": SBMLUnitDefinitionAST("substance_per_time", SUBSTANCE_FLUX),
    }
    for unit in model.unit_definitions:
        if unit.unit_id in builtins:
            raise ValueError(f"Duplicate or reserved SBML unit id {unit.unit_id!r}.")
        builtins[unit.unit_id] = unit
    return builtins


def _uses_fbc(model: SBMLModelAST, /) -> bool:
    return any(
        reaction.lower_bound is not None
        or reaction.upper_bound is not None
        or reaction.objective_coefficient != 0.0
        or bool(reaction.gpr_clauses)
        for reaction in model.reactions
    )


def validate_sbml_document(document: SBMLDocumentAST, /) -> SBMLValidationResult:
    """Validate the complete supported semantic matrix before any native object exists."""

    if not isinstance(document, SBMLDocumentAST):
        raise TypeError("document must be an SBMLDocumentAST.")
    profile = _profile(document.level, document.version)
    rejected: list[tuple[str, str, SBMLSemanticStatus]] = []
    if profile is None:
        rejected.append(
            (
                "document.level-version",
                f"SBML Level {document.level} Version {document.version} is unsupported.",
                SBMLSemanticStatus.UNSUPPORTED_LEVEL_VERSION,
            )
        )
        supported_packages: set[tuple[str, int]] = set()
        profile_id = "unsupported"
    else:
        supported_packages = set(profile.supported_packages)
        profile_id = profile.profile_id
    declared_packages = tuple((item.name, item.version) for item in document.packages)
    for index, package in enumerate(document.packages):
        if (package.name, package.version) not in supported_packages:
            rejected.append(
                (
                    f"document.packages[{index}]",
                    f"Package {package.name} version {package.version} is unsupported for this profile.",
                    SBMLSemanticStatus.UNSUPPORTED_PACKAGE,
                )
            )
    fbc_declared = any(package.name == "fbc" for package in document.packages)
    if _uses_fbc(document.model) and not fbc_declared:
        rejected.append(
            (
                "model.reactions.fbc-semantics",
                "Bounds, objectives, or GPR clauses require an explicitly supported FBC package.",
                SBMLSemanticStatus.UNSUPPORTED_PACKAGE,
            )
        )
    for index, rule in enumerate(document.model.rules):
        rejected.append(
            (
                f"model.rules[{index}]",
                f"SBML {rule.kind!r} rules are not supported by the native lowering.",
                SBMLSemanticStatus.UNSUPPORTED_RULE,
            )
        )
    for index, _ in enumerate(document.model.events):
        rejected.append(
            (
                f"model.events[{index}]",
                "SBML event trigger/delay/priority semantics are not supported.",
                SBMLSemanticStatus.UNSUPPORTED_EVENT,
            )
        )
    model = document.model
    if not model.model_id.strip():
        rejected.append(
            (
                "model.model_id",
                "Model identifier must be non-empty.",
                SBMLSemanticStatus.MALFORMED_MODEL,
            )
        )
    compartment_ids = [item.compartment_id for item in model.compartments]
    species_ids = [item.species_id for item in model.species]
    reaction_ids = [item.reaction_id for item in model.reactions]
    for owner, values in (
        ("model.compartments", compartment_ids),
        ("model.species", species_ids),
        ("model.reactions", reaction_ids),
    ):
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            rejected.append(
                (
                    owner,
                    "Identifiers must be unique and non-empty.",
                    SBMLSemanticStatus.MALFORMED_MODEL,
                )
            )
    compartment_set = set(compartment_ids)
    species_set = set(species_ids)
    for index, species in enumerate(model.species):
        if species.compartment not in compartment_set:
            rejected.append(
                (
                    f"model.species[{index}].compartment",
                    f"Unknown compartment {species.compartment!r}.",
                    SBMLSemanticStatus.INCONSISTENT_REFERENCE,
                )
            )
        if not isfinite(species.initial_amount) or species.initial_amount < 0.0:
            rejected.append(
                (
                    f"model.species[{index}].initial_amount",
                    "Initial amount must be finite and non-negative.",
                    SBMLSemanticStatus.MALFORMED_MODEL,
                )
            )
    supported_laws = {item.value for item in RateLawKind}
    kinetic_count = 0
    for index, reaction in enumerate(model.reactions):
        lower = (
            reaction.lower_bound
            if reaction.lower_bound is not None
            else -float("inf")
            if reaction.reversible
            else 0.0
        )
        upper = reaction.upper_bound if reaction.upper_bound is not None else float("inf")
        if (
            lower != lower
            or upper != upper
            or lower > upper
            or not isfinite(reaction.objective_coefficient)
        ):
            rejected.append(
                (
                    f"model.reactions[{index}].fbc-values",
                    "Reaction bounds must be ordered/non-NaN and objective finite.",
                    SBMLSemanticStatus.MALFORMED_MODEL,
                )
            )
        references = reaction.reactants + reaction.products
        for reference_index, reference in enumerate(references):
            if reference.species not in species_set:
                rejected.append(
                    (
                        f"model.reactions[{index}].references[{reference_index}]",
                        f"Unknown species {reference.species!r}.",
                        SBMLSemanticStatus.INCONSISTENT_REFERENCE,
                    )
                )
            if not isfinite(reference.stoichiometry) or reference.stoichiometry <= 0.0:
                rejected.append(
                    (
                        f"model.reactions[{index}].references[{reference_index}].stoichiometry",
                        "Stoichiometry must be finite and positive.",
                        SBMLSemanticStatus.MALFORMED_MODEL,
                    )
                )
        kinetic = reaction.kinetic_law
        if kinetic is not None:
            kinetic_count += 1
            if kinetic.kind not in supported_laws:
                rejected.append(
                    (
                        f"model.reactions[{index}].kinetic_law",
                        f"Kinetic law {kinetic.kind!r} has no native closed-form implementation.",
                        SBMLSemanticStatus.UNSUPPORTED_KINETIC_LAW,
                    )
                )
            if any(not isfinite(value) or value < 0.0 for value in kinetic.parameters):
                rejected.append(
                    (
                        f"model.reactions[{index}].kinetic_law.parameters",
                        "Kinetic parameters must be finite and non-negative.",
                        SBMLSemanticStatus.MALFORMED_MODEL,
                    )
                )
            if any(
                not isfinite(order) or order < 0.0
                for _, order in kinetic.reactant_orders + kinetic.product_orders
            ):
                rejected.append(
                    (
                        f"model.reactions[{index}].kinetic_law.orders",
                        "Kinetic orders must be finite and non-negative.",
                        SBMLSemanticStatus.MALFORMED_MODEL,
                    )
                )
            for species_id, _ in kinetic.reactant_orders + kinetic.product_orders:
                if species_id not in species_set:
                    rejected.append(
                        (
                            f"model.reactions[{index}].kinetic_law.orders",
                            f"Unknown kinetic species {species_id!r}.",
                            SBMLSemanticStatus.INCONSISTENT_REFERENCE,
                        )
                    )
    if kinetic_count not in (0, len(model.reactions)):
        rejected.append(
            (
                "model.reactions.kinetic_laws",
                "Native kinetic lowering requires a law for every reaction or for none.",
                SBMLSemanticStatus.INCOMPLETE_KINETICS,
            )
        )
    builtin_units = {
        "substance": SUBSTANCE,
        "mole": SUBSTANCE,
        "volume": VOLUME,
        "litre": VOLUME,
        "substance_per_time": SUBSTANCE_FLUX,
    }
    unit_dimensions = dict(builtin_units)
    custom_ids: set[str] = set()
    for index, unit in enumerate(model.unit_definitions):
        if unit.unit_id in builtin_units or unit.unit_id in custom_ids:
            rejected.append(
                (
                    f"model.unit_definitions[{index}].unit_id",
                    f"Duplicate or reserved unit definition {unit.unit_id!r}.",
                    SBMLSemanticStatus.INCONSISTENT_UNIT,
                )
            )
        else:
            custom_ids.add(unit.unit_id)
            unit_dimensions[unit.unit_id] = unit.dimension
        if unit.offset != 0.0:
            rejected.append(
                (
                    f"model.unit_definitions[{index}].offset",
                    "Affine-offset units cannot represent amounts, volumes, or rates.",
                    SBMLSemanticStatus.INCONSISTENT_UNIT,
                )
            )
    unit_references = [item.units for item in model.compartments]
    unit_references.extend(item.substance_units for item in model.species)
    unit_references.extend(item.flux_units for item in model.reactions)
    unit_references.extend(
        reaction.kinetic_law.rate_units
        for reaction in model.reactions
        if reaction.kinetic_law is not None
    )
    for index, unit_id in enumerate(unit_references):
        if unit_id not in unit_dimensions:
            rejected.append(
                (
                    f"model.unit_references[{index}]",
                    f"Unknown unit definition {unit_id!r}.",
                    SBMLSemanticStatus.INCONSISTENT_UNIT,
                )
            )
    for index, compartment in enumerate(model.compartments):
        if (
            compartment.units in unit_dimensions
            and unit_dimensions[compartment.units].exponents != VOLUME.exponents
        ):
            rejected.append(
                (
                    f"model.compartments[{index}].units",
                    "Compartment size units must have volume dimensions.",
                    SBMLSemanticStatus.INCONSISTENT_UNIT,
                )
            )
        if not isfinite(compartment.size) or compartment.size <= 0.0:
            rejected.append(
                (
                    f"model.compartments[{index}].size",
                    "Compartment size must be finite and positive.",
                    SBMLSemanticStatus.MALFORMED_MODEL,
                )
            )
    species_by_id = {item.species_id: item for item in model.species}
    for index, reaction in enumerate(model.reactions):
        if reaction.flux_units not in unit_dimensions:
            continue
        flux_dimension = unit_dimensions[reaction.flux_units].exponents
        for reference_index, reference in enumerate(
            reaction.reactants + reaction.products
        ):
            species = species_by_id.get(reference.species)
            if species is None or species.substance_units not in unit_dimensions:
                continue
            expected = (
                unit_dimensions[species.substance_units]
                .multiply(TIME.power(-1))
                .exponents
            )
            if flux_dimension != expected:
                rejected.append(
                    (
                        f"model.reactions[{index}].references[{reference_index}].units",
                        "Reaction flux dimensions must equal species amount per time.",
                        SBMLSemanticStatus.INCONSISTENT_UNIT,
                    )
                )
        law = reaction.kinetic_law
        if (
            law is not None
            and law.rate_units in unit_dimensions
            and unit_dimensions[law.rate_units].exponents != flux_dimension
        ):
            rejected.append(
                (
                    f"model.reactions[{index}].kinetic_law.rate_units",
                    "Kinetic-law rate dimensions must match reaction flux dimensions.",
                    SBMLSemanticStatus.INCONSISTENT_UNIT,
                )
            )
    if rejected:
        status_ = rejected[0][2]
        adapter_status = (
            AdapterStatus.MALFORMED_SOURCE
            if status_ is SBMLSemanticStatus.MALFORMED_MODEL
            else AdapterStatus.INCONSISTENT_SOURCE
            if status_
            in (
                SBMLSemanticStatus.INCONSISTENT_REFERENCE,
                SBMLSemanticStatus.INCONSISTENT_UNIT,
                SBMLSemanticStatus.INCOMPLETE_KINETICS,
            )
            else AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
        )
        losses = tuple(
            AdapterLoss(
                path,
                "import",
                "unsupported",
                reason,
                changes_interpretation=True,
            )
            for path, reason, _ in rejected
        )
    else:
        status_ = SBMLSemanticStatus.SUCCESS
        adapter_status = AdapterStatus.LOSSLESS
        losses = ()
    supported_semantics = (
        "core:compartments",
        "core:species",
        "core:constant-stoichiometric-reactions",
        "core:normalized-closed-form-kinetic-laws",
        "fbc:flux-bounds-objectives-gpr" if fbc_declared else "core:no-fbc-semantics",
        "rules:rejected-before-lowering",
        "events:rejected-before-lowering",
    )
    evidence = SBMLSemanticEvidence(
        rejected_count=jnp.asarray(len(rejected), dtype=jnp.int32),
        level=document.level,
        version=document.version,
        packages=declared_packages,
        profile_id=profile_id,
        supported_semantics=supported_semantics,
        rejected_paths=tuple(item[0] for item in rejected),
        rejection_reasons=tuple(item[1] for item in rejected),
        checked_before_lowering=True,
    )
    report = AdapterReport(
        adapter_status,
        f"SBML-L{document.level}V{document.version}",
        "phydrax.bioinformatics.systems",
        source_id=document.source_id,
        target_id=(
            f"native:{model.model_id}" if not rejected else f"unlowered:{model.model_id}"
        ),
        preserved_fields=(
            (
                "compartments",
                "species",
                "reactions",
                "units",
                "kinetic-laws",
                "fbc-bounds-objectives-gpr",
            )
            if not rejected
            else ()
        ),
        assumptions=("AST syntax was parsed by the caller.",),
        losses=losses,
    )
    valid = jnp.asarray(not rejected)
    return SBMLValidationResult(
        valid=valid,
        status=jnp.asarray(int(status_), dtype=jnp.int32),
        evidence=evidence,
        report=report,
        method_contract=_method_contract("sbml-semantic-validation"),
        source_id=document.source_id,
    )


def _scale(unit: SBMLUnitDefinitionAST, /) -> float:
    return unit.multiplier * (10.0**unit.scale)


def lower_sbml_document(
    document: SBMLDocumentAST,
    /,
    *,
    reject_unsupported: bool = True,
) -> SBMLLoweringResult:
    """Lower only a fully supported host AST; unsupported semantics are never dropped."""

    validation = validate_sbml_document(document)
    if not bool(validation.valid):
        if reject_unsupported:
            reason = "; ".join(validation.evidence.rejection_reasons)
            raise SBMLSemanticError(validation, reason)
        empty_scales = jnp.zeros((0,))
        evidence = SBMLLoweringEvidence(
            compartment_scales=empty_scales,
            species_scales=empty_scales,
            reaction_scales=empty_scales,
            validation=validation,
            network_id="unlowered",
            kinetics_id=None,
            lossless=False,
        )
        return SBMLLoweringResult(
            valid=jnp.asarray(False),
            status=validation.status,
            network=None,
            kinetics=None,
            evidence=evidence,
            report=validation.report,
            method_contract=_method_contract("sbml-native-lowering"),
        )
    model = document.model
    units = _unit_table(model)
    compartment_scales = jnp.asarray(
        [_scale(units[item.units]) for item in model.compartments]
    )
    species_scales = jnp.asarray(
        [_scale(units[item.substance_units]) for item in model.species]
    )
    reaction_scales = jnp.asarray(
        [_scale(units[item.flux_units]) for item in model.reactions]
    )
    species_scale_by_id = {
        item.species_id: float(species_scales[index])
        for index, item in enumerate(model.species)
    }
    compartments = tuple(
        Compartment(
            item.compartment_id,
            volume=item.size * float(compartment_scales[index]),
            volume_unit=units[item.units].dimension,
            spatial_dimensions=item.spatial_dimensions,
            constant=item.constant,
        )
        for index, item in enumerate(model.compartments)
    )
    species = tuple(
        Species(
            item.species_id,
            item.compartment,
            initial_amount=item.initial_amount * float(species_scales[index]),
            substance_unit=units[item.substance_units].dimension,
            composition=(
                None
                if item.elements is None
                else ChemicalComposition(item.elements, charge=item.charge)
            ),
            boundary_condition=item.boundary_condition,
            constant=item.constant,
        )
        for index, item in enumerate(model.species)
    )
    reactions = []
    for index, item in enumerate(model.reactions):
        coefficients: dict[str, float] = {}
        for reference in item.reactants:
            coefficients[reference.species] = (
                coefficients.get(reference.species, 0.0) - reference.stoichiometry
            )
        for reference in item.products:
            coefficients[reference.species] = (
                coefficients.get(reference.species, 0.0) + reference.stoichiometry
            )
        coefficients = {
            species_id: coefficient
            * species_scale_by_id[species_id]
            / float(reaction_scales[index])
            for species_id, coefficient in coefficients.items()
            if coefficient != 0.0
        }
        if not coefficients:
            raise SBMLSemanticError(
                validation,
                f"Reaction {item.reaction_id!r} has zero net stoichiometry.",
            )
        lower = (
            item.lower_bound
            if item.lower_bound is not None
            else -jnp.inf
            if item.reversible
            else 0.0
        )
        upper = item.upper_bound if item.upper_bound is not None else jnp.inf
        reaction_scale = float(reaction_scales[index])
        lower = lower * reaction_scale
        upper = upper * reaction_scale
        gpr = GeneReactionRule(item.gpr_clauses) if item.gpr_clauses else None
        reactions.append(
            Reaction(
                item.reaction_id,
                tuple(coefficients),
                jnp.asarray(tuple(coefficients.values())),
                lower_bound=lower,
                upper_bound=upper,
                objective_coefficient=item.objective_coefficient / reaction_scale,
                flux_unit=units[item.flux_units].dimension,
                gene_reaction_rule=gpr,
                exchange=item.exchange,
            )
        )
    network = StoichiometricNetwork(
        compartments,
        species,
        tuple(reactions),
        objective_sense="maximize",
    )
    kinetics = None
    if model.reactions and all(item.kinetic_law is not None for item in model.reactions):
        species_index = {item.species_id: index for index, item in enumerate(species)}
        kinetic_reactions = []
        for index, item in enumerate(model.reactions):
            law = item.kinetic_law
            if law is None:
                raise RuntimeError("Validated complete kinetic laws cannot be missing.")
            kinetic_reactions.append(
                KineticReaction(
                    index,
                    jnp.asarray([species_index[name] for name, _ in law.reactant_orders]),
                    jnp.asarray([order for _, order in law.reactant_orders]),
                    jnp.asarray(law.parameters),
                    rate_law=RateLawKind(law.kind),
                    product_indices=jnp.asarray(
                        [species_index[name] for name, _ in law.product_orders]
                    ),
                    product_orders=jnp.asarray(
                        [order for _, order in law.product_orders]
                    ),
                    rate_unit=units[law.rate_units].dimension,
                    rate_scale=(
                        _scale(units[law.rate_units]) / float(reaction_scales[index])
                    ),
                    kinetic_id=f"{item.reaction_id}:kinetics",
                )
            )
        kinetics = KineticReactionSystem(
            network,
            tuple(kinetic_reactions),
            system_id=f"{model.model_id}:kinetics",
        )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        f"SBML-L{document.level}V{document.version}",
        "phydrax.bioinformatics.systems",
        source_id=document.source_id,
        target_id=network.network_id,
        coordinate_mapping=("SBML species order -> native species axis",),
        preserved_fields=(
            "compartments",
            "species",
            "reactions",
            "units",
            "kinetic-laws",
            "fbc-bounds-objectives-gpr",
        ),
        assumptions=("AST syntax was parsed by the caller.",),
    )
    evidence = SBMLLoweringEvidence(
        compartment_scales=compartment_scales,
        species_scales=species_scales,
        reaction_scales=reaction_scales,
        validation=validation,
        network_id=network.network_id,
        kinetics_id=None if kinetics is None else kinetics.system_id,
        lossless=True,
    )
    return SBMLLoweringResult(
        valid=jnp.asarray(True),
        status=jnp.asarray(int(SBMLSemanticStatus.SUCCESS), dtype=jnp.int32),
        network=network,
        kinetics=kinetics,
        evidence=evidence,
        report=report,
        method_contract=_method_contract("sbml-native-lowering"),
    )


__all__ = [
    "lower_sbml_document",
    "validate_sbml_document",
    "SBML_SEMANTIC_MATRIX",
    "SBMLCompartmentAST",
    "SBMLDocumentAST",
    "SBMLEventAST",
    "SBMLKineticLawAST",
    "SBMLLoweringEvidence",
    "SBMLLoweringResult",
    "SBMLModelAST",
    "SBMLPackageDeclaration",
    "SBMLReactionAST",
    "SBMLRuleAST",
    "SBMLSemanticError",
    "SBMLSemanticEvidence",
    "SBMLSemanticProfile",
    "SBMLSemanticStatus",
    "SBMLSpeciesAST",
    "SBMLSpeciesReferenceAST",
    "SBMLUnitDefinitionAST",
    "SBMLValidationResult",
]
