# AdaptationPolyhedralOwner handoff

Worktree /Users/lgleyzer/PHYDRA/phydra-labs/.worktrees/ncm-adaptation-polyhedral, branch feature/native-cad-meshing-adaptation-polyhedral.
Scope after Main's split: (a) adaptation route verification, mixed/polyhedral adaptation, 3D metric routes, GeneralFamilies gates,
LayerOversetGaps native whole-slab test. Layer/polyhedral-core/nonnested/predecessor/mapped-measure work went to PolyhedralLayerOwner.

## Done + verified (source .tmp/dev-env.sh; -n 0)
Fixes:
- _mixed_adaptation.py: restored lost `prescribed = _prescribed_entities(...)`; source cells use vertex global IDs (were row indices);
  closure branches on marked cells first (refined cell keeps preferred template, neighbours close); `_exact_child_to_parent_affine_map`
  basis offset bug (singular for hex/pyramid); direct target charts via exact affine maps (pyramid rational form refused);
  `_exact_target_vertex_coordinates` evaluates target maps at the target cell's own reference corners; removed
  `__strict_contract__` on static-only classes; typing cleanups.
- _topology_edit.py `_cell_facets`: reference facets for fixed families, packed loops only for polyhedral blocks (shared-face witnesses
  were checked against canonicalized packed loops).
- _adaptation.py: restored lost `for exact_point in old_reference_exact[old_row]` loop in mixed chart coarsening; prepare_mesh_adaptation
  admits PolyhedralAdaptationRequest; surface-chart retarget no longer passes unknown `mixed_coarsening_maps` kwarg; degree narrowing.
- _polyhedral_plane_restriction.py: undefined `face`; no-cut result arity; edge-key typing.
- Tests: test_mixed_adaptation (evaluate_exact point-row shapes; positional MarkedMeshAdaptation; target certification is None when the
  source has no acceptance inputs — whole-cell validity asserted instead), test_layer_curving (constant exponent (0,0,0)),
  new test_native_structured_routes::test_native_graded_whole_slab_sweep_keeps_the_physical_layer_schedule (native, no gmsh; stations,
  first-cell height, independent prism-volume oracle) — closes the LayerOversetGaps finish-queue item.
Results:
- tests/unit/meshing/test_mixed_adaptation.py: 5/5 pass.
- tests/unit/meshing/test_polyhedral_adaptation.py: 3/4 pass (regeneration needs PolyhedralLayerOwner 78e349798).
- test_layer_curving.py::test_mapped_layer_transition_composes_exact_target_to_root_maps passes.
- test_native_structured_routes.py: all pass incl. the new slab test.
- test_tetra_metric (2 nodes), test_device_tetra_metric cube/curved: pass.
- ruff check, ruff format --check, ty (`--python <native-cad-meshing>/.venv-cad`) clean on all 7 touched files.

## Remaining (exact nodes, diagnosis)
- test_polyhedral_adaptation.py::test_material_l_site_regeneration_uses_physical_conservative_overlap and
  test_native_general_families.py::test_polyhedral_power_keeps_region_sites_and_fv_field_measure: carrier face-slot bug in
  _polyhedral_generation.py, fixed by PolyhedralLayerOwner commit 78e349798 (also changes polyhedral_support_issues signature) — rerun after rebase.
- test_layer_curving.py other nodes: _layer_core.py undefined `core_spec`/`periodic_topology_owner` (PolyhedralLayerOwner).
- test_source_association_transfer.py::test_trimmed_sheet_association_transfer_retains_patches_and_curve_intervals and
  test_surface_metric.py::test_native_surface_metric_route_rebinds_trimmed_sheet_source_charts: need SurfaceMeshingOwner 122019741,
  685a4adaf, 9862ec91f, 850643c6d, 0ba858f48; the `mixed_coarsening_maps` TypeError they hit at _adaptation.py is fixed here.
- test_surface_metric.py::test_native_surface_metric_route_reconstructs_and_transfers_implicit_sphere: fixture refused by implicit
  adaptive discovery (providers/_native_implicit.py, SurfaceMeshingOwner).
- test_implicit_source_charts.py::test_adaptive_implicit_root_chart_requires_smooth_or_profile_regular_source: not adaptation; >900 s
  here; noted in .tmp/batch-notes.md for SurfaceMeshingOwner.
- GeneralFamilies failures at checkpoint (not yet diagnosed; 95 pass / 14 fail / 3 errors in the family batch):
  test_hex_cut_generation.py::test_balanced_cut_all_hex_closure_covers_a_nonsweepable_tetrahedral_cavity (12 cut cells, determinant
  enclosures straddle 0 at ~1e-34 — validity UNRESOLVED);
  test_hex_dominant.py::{test_hex_dominant_accepts_dimension_qualified_simplex_source_blocks, test_hex_dominant_core_has_conforming_pyramid_and_tet_closure,
  test_mixed_closure_flattens_nested_per_cell_root_charts} (cells 4,5,6 status 2) and ::test_curved_source_map_is_composed_through_mixed_families
  (source tet fill lacks global embedding proof);
  test_native_general_families.py::{test_public_planar_dual_quad_route_executes_feature_guided_acceptance (QuadLayoutObstruction),
  test_public_plc_dual_hex_route_keeps_material_and_facet_ancestry, test_public_plc_hex_dominant_route_reports_real_mixed_transitions,
  test_public_plc_balanced_grid_route_accepts_cut_cell_hexes (meshcore PLC INVALID_SEED), test_public_plc_frame_grid_route_uses_varying_integer_layout
  (certification source_boundary_map_unresolved + patch_zone_adjacency), test_public_plc_balanced_grid_accepts_source_authored_curved_chart,
  test_public_plc_hex_dominant_composes_the_mapped_curved_source (MappedReferenceDomain root maps)} plus 3 setup errors;
  Main notes `_accepted_public_result` wrongly asserts `inputs.source is source` (should check source ID + revision).
- test_decision.py / test_proposals.py were in the batch; failures listed above are the complete -rf list (they passed).
- Docs/guides: no guide change made in this slice.
