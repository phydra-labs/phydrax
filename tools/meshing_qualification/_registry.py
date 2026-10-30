from __future__ import annotations


# Keep this module stdlib-only: installed-wheel gates can read the canonical
# mandatory inventory without importing Phydrax or the qualification runtime.
NATIVE_CASE_REGISTRY = {
    "native-provider-curve": (".scenarios.curve_surface", "native_curve"),
    "native-planar-feature-matrix": (
        ".scenarios.native_planar_profiles",
        "native_planar_feature_matrix",
    ),
    "native-surface": (".scenarios.curve_surface", "native_surface"),
    "native-brep-workflow": (".scenarios.cad_volume", "native_brep_workflow"),
    "native-volume": (".scenarios.cad_volume", "native_volume"),
    "native-mapped": (".scenarios.mapped_periodic", "native_mapped"),
    "native-periodic": (".scenarios.mapped_periodic", "native_periodic"),
    "native-periodic-profile-matrix": (
        ".scenarios.native_periodic_profiles",
        "native_periodic_profile_matrix",
    ),
    "native-transfer": (".scenarios.transfer", "native_transfer"),
    "native-tspline-cad-analysis": (".scenarios.tspline", "native_tspline_cad_analysis"),
    "native-image-material": (
        ".scenarios.image_material_profiles",
        "native_image_material",
    ),
    "native-cad-partition": (".scenarios.native_cad", "qualify_cad_partition"),
    "native-cad-profile-matrix": (
        ".scenarios.native_cad_profiles",
        "native_cad_profile_matrix",
    ),
    "native-planar-semantic": (".scenarios.native_cad", "qualify_planar_semantic"),
    "native-supermesh": (".scenarios.mesh_transfer", "qualify_supermesh"),
    "native-remap-p1": (".scenarios.mesh_transfer", "qualify_remap_p1"),
    "native-bisection": (".scenarios.mesh_transfer", "qualify_bisection"),
    "native-predicates": (".scenarios.native_kernel", "qualify_predicates"),
    "native-triangulation": (".scenarios.native_kernel", "qualify_triangulation"),
    "native-boundary-layer": (".scenarios.native_layers", "qualify_boundary_layer"),
    "native-hybrid-layer-matrix": (
        ".scenarios.native_hybrid_profiles",
        "native_hybrid_layer_profile_matrix",
    ),
    "native-reconstruction": (".scenarios.reconstruction", "native_reconstruction"),
    "native-envelope": (".scenarios.reconstruction", "native_envelope"),
    "native-constrained-solids": (
        ".scenarios.constrained_solid_matrix",
        "native_constrained_solids",
    ),
    "native-implicit-adaptation": (
        ".scenarios.native_implicit",
        "native_implicit_adaptation",
    ),
    "native-general-families": (
        ".scenarios.native_family_matrix",
        "native_general_families",
    ),
    "native-polyhedral": (".scenarios.polyhedral_matrix", "native_polyhedral"),
    "native-distributed-lifecycle": (
        ".scenarios.native_distributed",
        "native_distributed_lifecycle",
    ),
    "native-moving-overset": (
        ".scenarios.moving_overset_matrix",
        "native_moving_overset",
    ),
    "native-high-order-curving": (".scenarios.high_order", "native_high_order_curving"),
    "native-cold-fe-fv-continuation": (
        ".scenarios.cold_fe_fv",
        "native_cold_fe_fv_continuation",
    ),
    "native-design-loop": (".scenarios.native_design", "native_design_loop"),
}

NATIVE_SCENARIOS = tuple(NATIVE_CASE_REGISTRY)

OPTIONAL_SCENARIOS = (
    "manifold",
    "mmg",
    "ftetwild",
    "gmsh",
    "vorocrust",
    "poisson",
    "omega-h",
    "omega-h-distributed",
    "gmsh-metric",
    "gmsh-boundary-layer",
    "cad-curving",
    "planar-bands",
)
