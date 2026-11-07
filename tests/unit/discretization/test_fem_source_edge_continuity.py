import copy
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry._brep_capped_material import BRepCappedMaterialMap
from phydrax.geometry._brep_sphere_material import BRepSphereMaterialMap
from tools.meshing_qualification.scenarios import native_planar_profiles as profiles


@pytest.mark.parametrize(
    ("profile", "geometry", "size"),
    (("sphere-feature-surface", "sphere", 0.4),
     ("native-surface-pde:capped_cylinder", "capped_cylinder", 0.3),
     ("native-surface-pde:torus", "torus", 0.45)),
)
def test_source_immersion_matches_both_sides_of_every_interior_edge(
    profile: str, geometry: str, size: float
) -> None:
    domain = profiles._surface_definition(profile, geometry).domain
    scope = profiles.MeshingScope.whole(domain, 2)
    featured = geometry == "sphere"
    deviation = 0.25 * size * size
    controls = (profiles.UniformSizeControl(
        scope, size, maximum_size=1.5 * size if featured else None,
        strength=profiles.SizeControlStrength.HARD if featured else profiles.SizeControlStrength.SOFT,
    ),)
    features = ((profiles.ProtectedFeature(scope, profiles.FeatureKind.SURFACE,
                                           maximum_deviation=deviation),) if featured else ())
    compliance = (profiles.SizeCompliancePolicy(relative_tolerance=0.5, target_statistics=("p50",))
                  if featured else None)
    spec = profiles.SurfaceMeshingSpec(
        profiles.CellMeshingTarget(2, 3, profiles.CellFamilyPolicy(required=("triangle",))),
        scope, size_controls=controls, protected_features=features,
        size_compliance=compliance, maximum_deviation=min(0.02, deviation) if featured else 0.02,
    )
    result = profiles.NativeMeshingProvider(
        profiles.NativeMeshingOptions("parametric_surface")
    ).plan(domain, spec, coordinate_contract=profiles.SpatialCoordinateContract.si()).execute()
    assert result.certification is not None
    assert result.certification.inputs.source is not None
    immersion = phx.discretization.SurfaceFiniteElementGeometry.from_result(
        result, result.certification.inputs.source
    )
    if geometry == "sphere":
        assert isinstance(immersion.material_map, BRepSphereMaterialMap)
        physical = np.asarray(immersion.evaluate_points(0, np.asarray(((0.2, 0.3),))))
        np.testing.assert_allclose(np.linalg.norm(physical, axis=-1), 1.0, rtol=0.0, atol=2e-12)
        changed = eqx.tree_at(
            lambda value: value.material_map.radius,
            immersion,
            immersion.material_map.radius + 0.01,
        )
        with pytest.raises(Exception, match="Source surface geometry changed after proof"):
            changed.evaluate_points(0, np.asarray(((0.2, 0.3),)))
    if geometry == "capped_cylinder":
        assert isinstance(immersion.material_map, BRepCappedMaterialMap)
        material = immersion.material_map
        reference_vertices = np.asarray(((0., 0.), (1., 0.), (0., 1.)))
        assert np.isfinite(np.asarray(immersion.evaluate_points(0, reference_vertices))).all()
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="no unique Jacobian"):
            immersion.evaluate(0, reference_vertices)
        element = phx.discretization.lagrange_element("triangle", 1)
        finite_element = phx.discretization.FiniteElementPlan(
            result.mesh,
            phx.discretization.FiniteElementFieldSpec("u", element),
            coordinate_spec=result.geometry,
            source_surface=immersion,
        ).prepare()
        dofs = finite_element.dof_maps[0]
        coefficients = np.arange(dofs.global_dof_count, dtype=np.float64)
        reconstructed = np.asarray(finite_element.reconstruct(
            "u", coefficients, result.mesh.blocks[0].name, reference_vertices,
        ))
        local = np.einsum(
            "cij,cj->ci", np.asarray(dofs.cell_transforms[0]),
            coefficients[np.asarray(dofs.cell_dofs[0])],
        )
        expected = np.einsum("qi,ci->cq", np.asarray(element.tabulate(reference_vertices)[0]), local)
        np.testing.assert_allclose(reconstructed, expected, rtol=0.0, atol=1e-12)
        fan = material.blocks[0].fans
        side = material.blocks[0].sides[0]
        rows = np.unique(np.asarray((0, int(fan.indices[0]), int(side.indices[0])), dtype=np.int64))
        workset = material.prepare_rows(0, rows)
        quadrature = np.asarray(((0.2, 0.3), (0.4, 0.1)))
        full_points, full_jacobian = material.evaluate(0, quadrature)
        row_points, row_jacobian = workset.evaluate(quadrature)
        np.testing.assert_allclose(row_points, np.asarray(full_points)[rows], rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(row_jacobian, np.asarray(full_jacobian)[rows], rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(workset.evaluate_points(quadrature), row_points, rtol=0.0, atol=2e-12)
        from phydrax.discretization.fem._local_provider import SourceSurfaceGeometryActions

        local_geometry = SourceSurfaceGeometryActions(
            immersion, 0, rows, quadrature, np.ones(len(quadrature)),
        )
        local_metric = local_geometry.realize(finite_element.default_runtime)
        np.testing.assert_allclose(local_metric.points, row_points, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(local_metric.jacobian, row_jacobian, rtol=0.0, atol=2e-12)
        changed_cache = eqx.tree_at(
            lambda value: value.cached_points, local_geometry,
            local_geometry.cached_points.at[0, 0, 0].add(0.01),
        )
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="changed"):
            changed_cache.realize(finite_element.default_runtime)
        empty = material.prepare_rows(0, np.asarray((), dtype=np.int64))
        empty_points, empty_jacobian = empty.evaluate(quadrature)
        assert empty_points.shape == (0, len(quadrature), 3)
        assert empty_jacobian.shape == (0, len(quadrature), 3, 2)
        all_rows = np.arange(result.mesh.blocks[0].cell_count, dtype=np.int64)
        whole_points, whole_jacobian = material.prepare_rows(0, all_rows).evaluate(quadrature)
        np.testing.assert_allclose(whole_points, full_points, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(whole_jacobian, full_jacobian, rtol=0.0, atol=2e-12)
        forged_rows = eqx.tree_at(
            lambda value: value.block.corners, workset,
            workset.block.corners.at[0, 0, 0].add(0.01),
        )
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="changed after proof"):
            forged_rows.evaluate_points(quadrature)
        with pytest.raises(ValueError, match="unique, sorted"):
            material.prepare_rows(0, np.asarray((2, 1), dtype=np.int64))
        changed = eqx.tree_at(
            lambda value: value.material_map.blocks[0].fans.apex,
            immersion,
            fan.apex.at[0, 0].add(0.01),
        )
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="changed after proof"):
            changed.evaluate_points(0, np.asarray(((0.2, 0.3),)))
        source_arrays = tuple(
            leaf for leaf in jax.tree.leaves(
                (material.charts, material.patches, material.blocks, material.carrier_coordinates)
            ) if eqx.is_array(leaf)
        )
        apex_occurrence = next(i for i, value in enumerate(source_arrays) if value is fan.apex)
        group_index, position = next(
            (group_index, group.index(apex_occurrence))
            for group_index, group in enumerate(material.guard_groups)
            if apex_occurrence in group
        )
        # Changing a live leaf and its packed baseline together must still be
        # rejected by the enclosing source geometry's independent baseline.
        changed_both = eqx.tree_at(
            lambda value: (
                value.material_map.blocks[0].fans.apex,
                value.material_map.guard_baselines[group_index],
            ),
            immersion,
            (
                fan.apex.at[0, 0].add(0.01),
                material.guard_baselines[group_index].at[position, 0, 0].add(0.01),
            ),
        )
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="changed after proof"):
            changed_both.evaluate_points(0, np.asarray(((0.2, 0.3),)))
        forged_static = copy.copy(material)
        object.__setattr__(forged_static, "source_tree", jax.tree.structure(None))
        changed_static = eqx.tree_at(
            lambda value: value.material_map, immersion, forged_static,
        )
        with pytest.raises(ValueError, match="metadata changed after proof"):
            changed_static.evaluate_points(0, np.asarray(((0.2, 0.3),)))
        from phydrax.geometry._brep_circle_rim_proof import _phase_roots
        from phydrax.geometry._meshing_domain import PatchPoleUse
        from phydrax.geometry.brep._intersection import NativePeriodEndpoint
        from phydrax.geometry.brep._patches import CircleCurve, LineCurve

        source = material.charts.domain
        assert source.brep_authority is not None
        source_geometry = source.brep_authority.geometry
        edge = next(
            edge for edge, index in enumerate(source_geometry.edge_curves)
            if index >= 0 and isinstance(source_geometry.curves[index], CircleCurve)
        )
        curve_index = source_geometry.edge_curves[edge]
        root, _ = source_geometry.edge_endpoint_roots[edge]
        assert isinstance(root, NativePeriodEndpoint)
        gap = NativePeriodEndpoint(root.curve, root.patch, root.axis, turns=Fraction(1, 2))
        with pytest.raises(ValueError, match="native phase"):
            _phase_roots((root, gap), source_geometry.curves[curve_index])
        curve = source_geometry.curves[curve_index]
        assert isinstance(curve, CircleCurve)
        changed_source = eqx.tree_at(
            lambda value: value.material_map.charts.domain.brep_authority.geometry.curves[curve_index].first_axis,
            immersion,
            curve.first_axis.at[0].add(0.01),
        )
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="changed after proof"):
            changed_source.evaluate_points(0, np.asarray(((0.2, 0.3),)))
        center_zero = int(np.flatnonzero(np.asarray(curve.center) == 0.0)[0])
        center_bits = jax.lax.bitcast_convert_type(curve.center, jnp.uint64)
        flipped_center = jax.lax.bitcast_convert_type(
            center_bits.at[center_zero].set(center_bits[center_zero] ^ jnp.uint64(1 << 63)),
            jnp.float64,
        )
        changed_zero = eqx.tree_at(
            lambda value: value.material_map.charts.domain.brep_authority.geometry.curves[curve_index].center,
            immersion, flipped_center,
        )
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="changed after proof"):
            changed_zero.evaluate_points(0, np.asarray(((0.2, 0.3),)))
        nan_group = next(
            i for i, baseline in enumerate(material.guard_baselines)
            if baseline.dtype == jnp.float64 and np.isnan(np.asarray(baseline)).any()
        )
        nan_baseline = material.guard_baselines[nan_group]
        nan_index = tuple(int(value) for value in np.argwhere(np.isnan(np.asarray(nan_baseline)))[0])
        nan_bits = jax.lax.bitcast_convert_type(nan_baseline, jnp.uint64)
        changed_nan = jax.lax.bitcast_convert_type(
            nan_bits.at[nan_index].set(nan_bits[nan_index] ^ jnp.uint64(1)),
            jnp.float64,
        )
        forged_nan = eqx.tree_at(
            lambda value: value.material_map.guard_baselines[nan_group],
            immersion, changed_nan,
        )
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="changed after proof"):
            forged_nan.evaluate_points(0, np.asarray(((0.2, 0.3),)))
        patch, position = next(
            (patch_index, use_index)
            for patch_index, patch_value in enumerate(source.patches)
            for use_index, use in enumerate(patch_value.loops[0])
            if isinstance(use, PatchPoleUse)
        )
        pole = source.patches[patch].loops[0][position]
        assert isinstance(pole, PatchPoleUse)
        assert isinstance(pole.pcurve, LineCurve)
        changed_pole = eqx.tree_at(
            lambda value: value.material_map.charts.domain.patches[patch].loops[0][position].pcurve.origin,
            immersion,
            pole.pcurve.origin.at[1].add(0.01),
        )
        with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="changed after proof"):
            changed_pole.evaluate_points(0, np.asarray(((0.2, 0.3),)))
    vertices = np.asarray(((0., 0.), (1., 0.), (0., 1.)))
    fractions = np.asarray((0.2, 0.5, 0.8))[:, None]
    incidence = {}
    derivatives = {}
    for block_index, block in enumerate(result.mesh.blocks):
        for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
            points = vertices[local_a] * (1 - fractions) + vertices[local_b] * fractions
            physical = np.asarray(immersion.evaluate_points(block_index, points))
            if geometry == "capped_cylinder":
                _, jacobian = immersion.evaluate(block_index, points)
                tangent = np.einsum(
                    "cqdk,k->cqd", np.asarray(jacobian),
                    vertices[local_b] - vertices[local_a],
                )
            for row, cell in enumerate(np.asarray(block.vertices)):
                first, second = int(cell[local_a]), int(cell[local_b])
                key = (min(first, second), max(first, second))
                values = physical[row] if first < second else physical[row, ::-1]
                incidence.setdefault(key, []).append(values)
                if geometry == "capped_cylinder":
                    oriented = tangent[row] if first < second else -tangent[row, ::-1]
                    derivatives.setdefault(key, []).append(oriented)
    interior = [items for items in incidence.values() if len(items) == 2]
    assert interior
    for first, second in interior:
        np.testing.assert_allclose(first, second, rtol=0.0, atol=2e-12)
    if geometry == "capped_cylinder":
        for first, second in derivatives.values():
            np.testing.assert_allclose(first, second, rtol=0.0, atol=2e-11)
