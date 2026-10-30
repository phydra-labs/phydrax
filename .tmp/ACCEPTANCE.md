# AdaptationPolyhedralOwner acceptance checklist

Worktree: /Users/lgleyzer/PHYDRA/phydra-labs/.worktrees/ncm-adaptation-polyhedral (from 6bffc1354).

## A1. AdaptationRouteOwner unverified work
- tests/unit/meshing/test_mixed_adaptation.py (all; incl. high-order source geometry + native mixed route coarsening)
- tests/unit/meshing/test_layer_curving.py (all; mapped coarsening/root composition)
- tests/unit/meshing/test_surface_metric.py::test_native_surface_metric_route_reconstructs_and_transfers_implicit_sphere
- tests/unit/meshing/test_surface_metric.py::test_native_surface_metric_route_rebinds_trimmed_sheet_source_charts
- tests/unit/meshing/test_surface_metric.py::test_closed_brep_source_chart_metric_adaptation[brep-sphere|brep-torus]
- tests/unit/meshing/test_source_association_transfer.py::test_trimmed_sheet_association_transfer_retains_patches_and_curve_intervals (periodic finalizer guard)
- Triage tests/unit/geometry/test_implicit_source_charts.py::test_adaptive_implicit_root_chart_requires_smooth_or_profile_regular_source (owner: implicit root charts → batch note for SurfaceMeshingOwner if not mine)

## A2. Mixed/polyhedral adaptation + 3D metric routes
- tests/unit/meshing/test_mixed_adaptation.py, tests/unit/meshing/test_polyhedral_adaptation.py
- tests/unit/meshing/test_tetra_metric.py::test_nonperiodic_refinement_reaches_the_unit_metric
- tests/unit/meshing/test_tetra_metric.py::test_coarsening_updates_live_topology_and_conserves_region_volume
- tests/unit/meshing/test_device_tetra_metric.py::test_device_tetra_metric_agrees_with_native_on_cube_and_curved_boundary

## A3. PolyhedralNestedGaps remaining
- Layer + polyhedral core: LayerCoreConstruction.polyhedral_core field; conforming cap interface; FV consumption; independent volume oracle
  - tests/unit/meshing/test_native_layer_polyhedral_core.py (both nodes)
- Nonnested bounded surface reconstruction through edits: route (_adaptation surface metric finalizer) + tests/unit/discretization/test_cell_geometry_surface_reconstruction.py
- Predecessor reuse certifying only changed cells: tests/unit/discretization/test_cell_geometry_predecessor.py
- PLC periodic + layer controls: tests/unit/meshing/test_native_polyhedral_controls.py, test_native_polyhedral_acceptance.py
- Material-L polyhedral: test_native_polyhedral_controls.py::test_periodic_material_l_polyhedral_fill_preserves_material_and_coverage
- Mapped measure embedded/pyramids: tests/unit/discretization/test_mapped_geometry_measure.py, test_mapped_geometry_measure_extensions.py
- Polygon query / source realization: tests/unit/discretization/test_polygon_query.py, test_source_geometry_realization.py
- Finish queue LayerOversetGaps: a native (no gmsh) whole-slab layer sweep test must exist.

## A4. GeneralFamilies
- tests/unit/meshing/test_decision.py, test_proposals.py (incl. 3 implicit proposal nodes), test_quad_generation.py, test_hex_generation.py,
  test_native_structured_routes.py, test_hex_cut_generation.py (6 nodes), test_hex_dominant.py (2), test_hex_frame_map.py (2),
  test_native_general_families.py, test_native_polyhedral_generation.py

## Gates
- ruff check / ruff format --check / ty on touched files; .tmp/changelog-entries/AdaptationPolyhedral.md; guides.
