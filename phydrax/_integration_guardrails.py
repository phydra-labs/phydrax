#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


FORBIDDEN_RUNTIME_PACKAGES = frozenset(
    {
        "opensbli",
        "ops",
        "maple",
        "rmanak-fd",
        "juolrabi-gfd",
        "gfdflow",
        "mgfd",
        "fdtdx",
    }
)

CANONICAL_CORE_OWNERS = {
    "cochain": "phydrax.discretization.CochainDiscretization",
    "cell_complex": "phydrax.discretization.CellComplexTopology",
    "homology": "phydrax.topology.compute_homology",
    "persistence": "phydrax.topology.compute_persistence",
    "cellular_chain_map": "phydrax.topology.CellularChainMap",
    "extended_persistence": "phydrax.topology.compute_extended_persistence",
    "integral_homology": "phydrax.topology.compute_integral_homology",
    "linear_operator": "phydrax.linalg.AbstractLinearOperator",
    "lattice_harmonic": "phydrax.discretization.spectral.LatticeHarmonicDiscretization",
    "real_coordinate_map": "phydrax.linalg.AbstractRealCoordinateMap",
    "finite_real_algebra": "phydrax.metrix.algebra.AbstractFiniteRealAlgebraSpec",
    "point_topology": "phydrax.discretization.PointTopology",
    "weno_teno": "phydrax.discretization.finite_volume",
    "maxwell": "phydrax.solver.CompatibleMaxwellPlan",
    "pde_ir": "phydrax.equations.PDEProblemIR",
    "biological_sequence": "phydrax.bioinformatics.sequence.SequenceBatch",
    "biological_feature_dictionary": "phydrax.bioinformatics.foundation.FeatureDictionary",
    "biospecimen_lineage": "phydrax.bioinformatics.foundation.BiospecimenLineage",
    "genomic_coordinate": "phydrax.bioinformatics.genomics.IntervalSet",
    "biological_assay": "phydrax.bioinformatics.omics.CountAssay",
    "phylogenetic_tree": "phydrax.bioinformatics.phylogenetics.TreeTopology",
    "macromolecular_structure": "phydrax.bioinformatics.structure.MacromolecularStructure",
    "mass_spectrum": "phydrax.bioinformatics.spectrometry.SpectrumBatch",
    "biochemical_network": "phydrax.bioinformatics.systems.StoichiometricNetwork",
}


def reject_external_runtime(package: str, /) -> None:
    normalized = str(package).strip().lower()
    if normalized in FORBIDDEN_RUNTIME_PACKAGES:
        raise ValueError(
            f"External runtime {package!r} is forbidden in core Phydrax; "
            "use a neutral out-of-tree data adapter."
        )


class CoreAbstractionRegistry(StrictModule, NonTrainableState):
    owners: tuple[tuple[str, str], ...] = eqx.field(static=True)
    registry_id: str = eqx.field(static=True)

    def __init__(self, additions: Mapping[str, str] | None = None, /):
        owners = dict(CANONICAL_CORE_OWNERS)
        for kind, owner in ({} if additions is None else additions).items():
            kind_ = str(kind)
            owner_ = str(owner)
            if not kind_ or not owner_:
                raise ValueError("Core abstraction kind/owner must be nonempty.")
            if kind_ in owners and owners[kind_] != owner_:
                raise ValueError(
                    f"Core abstraction {kind_!r} already belongs to {owners[kind_]!r}."
                )
            owners[kind_] = owner_
        values = tuple(sorted(owners.items()))
        self.owners = values
        self.registry_id = canonical_fingerprint(
            {"kind": "core-abstraction-registry", "owners": dict(values)}
        )

    def owner(self, kind: str, /) -> str:
        mapping = dict(self.owners)
        if kind not in mapping:
            raise KeyError(f"Unknown core abstraction {kind!r}.")
        return mapping[kind]


__all__ = [
    "CANONICAL_CORE_OWNERS",
    "FORBIDDEN_RUNTIME_PACKAGES",
    "CoreAbstractionRegistry",
    "reject_external_runtime",
]
