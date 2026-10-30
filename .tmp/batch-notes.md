# Cross-owner notes (AdaptationPolyhedralOwner)

## SurfaceMeshingOwner — implicit root charts
- Node: tests/unit/geometry/test_implicit_source_charts.py::test_adaptive_implicit_root_chart_requires_smooth_or_profile_regular_source
- Owner: phydrax/geometry/implicit/_source_charts.py (`prepare_adaptive_implicit_root_charts` → `_build_face` → `_make_face_pieces` → `_face_piece_proof`).
- Previously reported failure (AssociationTransferFinish): `ValueError: normal-line root chart lacks strict interval endpoint signs` raised in `_face_piece_proof`.
- In this worktree (checkpoint 6bffc1354) the node did not finish within 900 s (`-n 0`, dev-env.sh), no traceback captured.
- Not part of the adaptation finalizer (`_adaptation._finalize_native`); no adaptation-side change is required.

## PolyhedralLayerOwner — restricted-power carrier face slots (sent by IRC)
- phydrax/meshing/_polyhedral_generation.py `_carrier_tetrahedralization`: native `restricted_power_cells` expects `tet_face_facets[:, k]` to be the face opposite vertex k; the code indexes by `connectivity.cell_faces` order, which differs. Patch: key each slot by `sorted(np.delete(tetrahedra[cell], local_face))` for `local_face in range(4)`.
- Blocks: tests/unit/meshing/test_polyhedral_adaptation.py::test_material_l_site_regeneration_uses_physical_conservative_overlap and tests/unit/meshing/test_native_general_families.py::test_polyhedral_power_keeps_region_sites_and_fv_field_measure.
- phydrax/meshing/_layer_core.py undefined `core_spec` (lines 793, 862, 865, 878) and `periodic_topology_owner` (446, 456, 470) block tests/unit/meshing/test_layer_curving.py mapped/mixed layer-core nodes.
