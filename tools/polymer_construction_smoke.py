from __future__ import annotations

import json

import phydrax as phx
from phydrax.applications import polymer_construction as pc


def main() -> int:
    source = {
        "material_id": "smoke-network",
        "bead_types": [{"id": "A", "mass": 1.0}],
        "chains": [
            {"id": "left", "sequence": [0, 0]},
            {"id": "right", "sequence": [0, 0]},
        ],
        "ports": [
            {
                "id": "left",
                "chain_id": "left",
                "bead_offset": 0,
                "compatibility_class": "donor",
            },
            {
                "id": "right",
                "chain_id": "right",
                "bead_offset": 0,
                "compatibility_class": "acceptor",
            },
        ],
        "maximum_particles": 4,
    }
    adapted = pc.polymer_recipe_from_mapping(
        source, phx.atomistic.AtomisticUnitSystem.reduced()
    )
    construction = pc.lower_polymer_recipe(adapted.recipe)
    state = pc.initialize_polymer_reaction_state(construction)
    reacted = pc.apply_polymer_reaction(
        state,
        pc.PolymerReactionTemplate("crosslink", "donor", "acceptor"),
        "left",
        "right",
    )
    network = pc.polymer_network_observables(reacted.state)
    successful = bool(
        adapted.report.valid
        and construction.successful
        and reacted.successful
        and network.successful
    )
    print(
        json.dumps(
            {
                "successful": successful,
                "component_count": int(network.component_count),
                "conversion": float(network.conversion),
            }
        )
    )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
