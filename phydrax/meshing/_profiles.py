# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Scientific native workflow scopes; qualification is not implied by registration."""

from __future__ import annotations

from ..qualification import CapabilityProfile, SupportTuple


_SPECS = (
    (
        "meshing.native-cad",
        {
            "source": "native-brep-step-iges",
            "operations": "qualified-construction-query-boolean-partition-exchange",
            "consumer": "certified-curved-finite-elements",
            "repair": "explicit-derived-source-with-displacement-evidence",
            "external-geometry-engine": False,
        },
    ),
    (
        "meshing.native-constrained-solids",
        {
            "source": "declared-plc-regions-materials-cavities-and-features",
            "boundary": "fixed-or-certified-conforming",
            "coverage": "independent-source-bound-proof",
            "consumer": "finite-elements-and-finite-volumes",
        },
    ),
    (
        "meshing.native-curves-planar-surfaces",
        {
            "source": "junction-networks-planar-holes-open-and-closed-surfaces",
            "identity": "explicit-source-entity-and-occurrence",
            "fidelity": "source-bound-deviation-and-coverage",
            "family": "requested-native-family-without-silent-substitution",
        },
    ),
    (
        "meshing.native-design",
        {
            "objective": "identified-physical-error-or-well-posed-qoi",
            "proposals": "analytic-uniform-and-trained-decision-components",
            "derivative": "qualified-fixed-event-only",
            "publication": "fresh-source-bound-physical-reanalysis-and-atomic-rollback",
        },
    ),
    (
        "meshing.native-distributed",
        {
            "execution": "owner-local-generation-adaptation-repartition",
            "identity": "explicit-global-entity-and-owner-placement",
            "restart": "producer-exit-fresh-reader-changed-placement",
            "state": "declared-fe-fv-material-history-auxiliary-integrator-rng-clock",
        },
    ),
    (
        "meshing.native-hybrid-layers",
        {
            "source": "qualified-convex-concave-curved-and-periodic-walls",
            "geometry": "physical-layer-height-collision-policy-cap-and-native-core",
            "family": "requested-mixed-and-high-order-cells",
            "transfer": "field-specific-conservative-and-compatible",
        },
    ),
    (
        "meshing.native-implicit",
        {
            "source": "explicit-implicit-field-contract",
            "discovery": "adaptive-or-explicit-fixed-lattice-policy",
            "events": "topology-and-branch-changes-are-discrete",
            "singularity": "explicit-unresolved-or-refusal",
            "consumer": "source-faithful-adaptation-and-physical-convergence",
        },
    ),
    (
        "meshing.native-labeled-images",
        {
            "source": "labels-with-explicit-geometric-interpretation-and-frame",
            "regions": "material-interfaces-and-junctions",
            "identity": "declared-label-to-region-binding",
            "consumer": "inventory-and-interface-flux-balance",
        },
    ),
    (
        "meshing.native-moving-overset",
        {
            "source": "accepted-moving-source-and-exact-priority-partition",
            "coupling": "interpolation-distinct-from-conservative-overlap",
            "consumer": "continued-physical-update",
            "failure": "orphan-refusal-and-complete-rollback",
        },
    ),
    (
        "meshing.native-periodic",
        {
            "source": "admitted-exact-periodic-actions-and-region-sides",
            "identity": "winding-occurrence-and-source-vertex-witnesses",
            "adaptation": "orbit-synchronized-refine-coarsen-and-metric-change",
            "transfer": "declared-compatible-field-actions",
        },
    ),
    (
        "meshing.native-polyhedral",
        {
            "source": "restricted-voronoi-power-with-material-and-periodic-ancestry",
            "topology": "reciprocal-oriented-faces-and-explicit-site-components",
            "consumer": "finite-volumes-and-virtual-elements",
            "remap": "independently-checked-nonnested-conservative-overlap",
        },
    ),
    (
        "meshing.native-quads-hexes",
        {
            "routes": "structured-sweep-multiblock-dual-and-qualified-general-routes",
            "quad-guidance": "source-bound-field-changes-validated-layout-topology",
            "hex-guidance": "feature-aware-cut-cells-and-admitted-frame-grid-maps",
            "family": "all-quad-or-all-hex-without-mixed-family-fallback",
        },
    ),
    (
        "meshing.native-reconstruction",
        {
            "source": "declared-noisy-point-or-envelope-interpretation",
            "approximation": "explicit-original-source-deviation-evidence",
            "topology": "components-and-genus-checked-independently",
            "repair": "explicit-source-revision-not-silent-healing",
        },
    ),
    (
        "meshing.native-tspline-cad-analysis",
        {
            "source": "admitted-analysis-suitable-tspline-with-explicit-axis-identity",
            "refinement": "genuinely-local-geometry-preserving-weighted-prolongation",
            "consumer": "native-cad-meshing-and-qualified-isogeometric-analysis",
            "derivative": "fixed-topology-and-qualified-physical-regularity",
            "exchange": "lossless-native-and-explicit-external-format-limits",
        },
    ),
)

_REQUIRED_GATES = (
    "analytic-control",
    "documentation-nonclaims",
    "independent-cold-restoration",
    "installed-native-isolation",
    "physical-consumer",
    "public-workflow",
    "qualification-campaign",
    "source-and-topology-evidence",
)


def native_meshing_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Describe scoped candidates, never a release or completed-campaign claim.

    The authored scenario inventory and its original scientific controls remain
    owned by ``tools.meshing_qualification``. These profiles name workflow
    capabilities without duplicating its case registry or execution runtime.
    """
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (
                SupportTuple(
                    name, {**attributes, "backend": "cpu", "precision": "float64"}
                ),
            ),
            required_gates=_REQUIRED_GATES,
        )
        for name, attributes in _SPECS
    )


__all__ = ["native_meshing_candidate_profiles"]
