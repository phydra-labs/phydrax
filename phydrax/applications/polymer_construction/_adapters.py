#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Mapping

import equinox as eqx

from ..._artifact_security import (
    AdmittedExternalArtifact,
    ExternalArtifactPolicy,
    read_admitted_artifact,
)
from ..._strict import StrictModule
from ...atomistic import AtomisticUnitSystem
from ...atomistic.interchange._core import canonical_source_digest, require_mapping_fields
from ...discretization import PeriodicCell
from ...interchange import AdapterCapability, AdapterReport, AdapterStatus
from ._recipes import (
    PolymerChainSpec,
    PolymerConnectionPortPlan,
    PolymerMaterialRecipePlan,
)


_CAPABILITIES = tuple(
    AdapterCapability(value)
    for value in (
        "polymer.material-id",
        "polymer.bead-types",
        "polymer.explicit-sequences",
        "polymer.chain-rings",
        "polymer.connection-ports",
        "polymer.fixed-capacity",
    )
)


class PolymerRecipeAdapterResult(StrictModule):
    recipe: PolymerMaterialRecipePlan
    report: AdapterReport
    source_id: str = eqx.field(static=True)


def polymer_recipe_from_mapping(
    source: Mapping,
    units: AtomisticUnitSystem,
    /,
    *,
    cell: PeriodicCell | None = None,
) -> PolymerRecipeAdapterResult:
    if not isinstance(source, Mapping):
        raise TypeError("source must be a mapping.")
    if not isinstance(units, AtomisticUnitSystem):
        raise TypeError("units must be AtomisticUnitSystem.")
    require_mapping_fields(
        source,
        ("material_id", "bead_types", "chains", "maximum_particles"),
    )
    bead_types = tuple(source["bead_types"])
    if not bead_types or any(not isinstance(value, Mapping) for value in bead_types):
        raise TypeError("bead_types must contain mapping records.")
    for value in bead_types:
        require_mapping_fields(value, ("id", "mass"))
    chains_source = tuple(source["chains"])
    if not chains_source or any(
        not isinstance(value, Mapping) for value in chains_source
    ):
        raise TypeError("chains must contain mapping records.")
    for value in chains_source:
        require_mapping_fields(value, ("id", "sequence"))
    ports_source = tuple(source.get("ports", ()))
    if any(not isinstance(value, Mapping) for value in ports_source):
        raise TypeError("ports must contain mapping records.")
    for value in ports_source:
        require_mapping_fields(
            value,
            ("id", "chain_id", "bead_offset", "compatibility_class"),
        )
    recipe = PolymerMaterialRecipePlan(
        str(source["material_id"]),
        tuple(str(value["id"]) for value in bead_types),
        [float(value["mass"]) for value in bead_types],
        tuple(
            PolymerChainSpec(
                str(value["id"]),
                value["sequence"],
                ring=bool(value.get("ring", False)),
            )
            for value in chains_source
        ),
        units,
        bead_charges=[float(value.get("charge", 0.0)) for value in bead_types],
        ports=tuple(
            PolymerConnectionPortPlan(
                str(value["id"]),
                str(value["chain_id"]),
                int(value["bead_offset"]),
                str(value["compatibility_class"]),
                maximum_uses=int(value.get("maximum_uses", 1)),
            )
            for value in ports_source
        ),
        cell=cell,
        maximum_particles=int(source["maximum_particles"]),
        particle_id_start=int(source.get("particle_id_start", 1)),
        bonded_lennard_jones_scale=float(source.get("bonded_lennard_jones_scale", 1.0)),
    )
    source_id = canonical_source_digest(source)
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "polymer-recipe-mapping",
        "phydrax-polymer-material-recipe",
        source_id=source_id,
        target_id=recipe.plan_id,
        preserved_fields=(
            "material_id",
            "bead_types",
            "chains",
            "ports",
            "maximum_particles",
        ),
        capabilities=_CAPABILITIES,
    )
    return PolymerRecipeAdapterResult(recipe, report, source_id)


def polymer_recipe_from_admitted_json(
    artifact: AdmittedExternalArtifact,
    units: AtomisticUnitSystem,
    /,
    *,
    policy: ExternalArtifactPolicy,
    cell: PeriodicCell | None = None,
) -> PolymerRecipeAdapterResult:
    payload = read_admitted_artifact(artifact, policy=policy)
    source = json.loads(payload.decode("utf-8"))
    if not isinstance(source, Mapping):
        raise TypeError("Admitted polymer recipe JSON must decode to a mapping.")
    return polymer_recipe_from_mapping(source, units, cell=cell)


def polymer_recipe_to_mapping(recipe: PolymerMaterialRecipePlan, /) -> dict:
    if not isinstance(recipe, PolymerMaterialRecipePlan):
        raise TypeError("recipe must be PolymerMaterialRecipePlan.")
    return {
        "material_id": recipe.material_id,
        "bead_types": [
            {
                "id": identifier,
                "mass": float(recipe.bead_masses[index]),
                "charge": float(recipe.bead_charges[index]),
            }
            for index, identifier in enumerate(recipe.bead_type_ids)
        ],
        "chains": [
            {
                "id": chain.chain_id,
                "sequence": [int(value) for value in chain.bead_type_indices],
                "ring": chain.ring,
            }
            for chain in recipe.chains
        ],
        "ports": [
            {
                "id": port.port_id,
                "chain_id": port.chain_id,
                "bead_offset": port.bead_offset,
                "compatibility_class": port.compatibility_class,
                "maximum_uses": port.maximum_uses,
            }
            for port in recipe.ports
        ],
        "maximum_particles": recipe.maximum_particles,
        "particle_id_start": recipe.particle_id_start,
        "bonded_lennard_jones_scale": recipe.bonded_lennard_jones_scale,
    }


__all__ = [
    "PolymerRecipeAdapterResult",
    "polymer_recipe_from_admitted_json",
    "polymer_recipe_from_mapping",
    "polymer_recipe_to_mapping",
]
