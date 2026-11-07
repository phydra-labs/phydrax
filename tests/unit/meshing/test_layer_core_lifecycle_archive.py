"""Real mapped layer/core recipe validation across a mixed refinement."""

from copy import copy

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.meshing._layer_core_root_geometry import ComputedLayerCoreRoot
from phydrax.meshing._model_codec import (
    _layer_core_accepted_lifecycle_recipes,
    _register_model_archive_types,
    _restore_layer_core_accepted_lifecycle,
    _same_model_content,
    layer_core_accepted_adaptation_recipe,
    layer_core_accepted_result_recipe,
    restore_layer_core_accepted_adaptation,
    restore_layer_core_accepted_result,
)
from tests._support.tiny_mapped_lifecycle import tiny_affine_mapped_lifecycle


@pytest.mark.skipif(not meshcore_available(), reason="native layer/core generation requires meshcore")
def test_mapped_lifecycle_recipe_reuses_only_the_actual_shared_root() -> None:
    source, spec, initial, prepared, adaptation = tiny_affine_mapped_lifecycle()
    root = initial.layer_core_root_geometry
    assert root is not None and root is adaptation.target.layer_core_root_geometry
    assert prepared.policy.association_transfer.source is source
    assert prepared.policy.association_transfer.specification is spec
    _register_model_archive_types()
    combined = _layer_core_accepted_lifecycle_recipes(
        initial, prepared, adaptation, source, spec
    )
    standalone = (
        layer_core_accepted_result_recipe(initial, source, spec),
        layer_core_accepted_adaptation_recipe(prepared, adaptation),
        layer_core_accepted_result_recipe(adaptation.target, source, spec),
    )
    assert all(
        _same_model_content(actual, expected)
        for actual, expected in zip(combined, standalone, strict=True)
    )
    restored = restore_layer_core_accepted_result(combined[2], source, spec)
    assert _same_model_content(restored, adaptation.target)
    restored_initial, restored_target, restored_adaptation, transfer = (
        _restore_layer_core_accepted_lifecycle(
            combined[0], combined[2], combined[1], source, spec
        )
    )
    assert restored_initial.layer_core_root_geometry is restored_target.layer_core_root_geometry
    assert _same_model_content(restored_initial, initial)
    assert _same_model_content(restored_target, adaptation.target)
    assert restored_adaptation.result_id == adaptation.result_id
    standalone_adaptation = restore_layer_core_accepted_adaptation(
        combined[1], restored_initial, restored_target, transfer
    )
    assert standalone_adaptation.result_id == adaptation.result_id
    restored_root = restored_initial.layer_core_root_geometry
    assert restored_root is not None
    assert restored_initial.mesh is restored_root.mesh
    assert restored_initial.geometry is restored_root.geometry
    assert restored_target.mesh is not restored_root.mesh
    assert restored_target.geometry is not restored_root.geometry
    computed_root = ComputedLayerCoreRoot.reprove(
        source, restored_root.specification, restored_root
    )
    foreign_parent = copy(restored_initial)
    object.__setattr__(foreign_parent, "layer_core_root_geometry", copy(restored_root))
    with pytest.raises(ValueError, match="different exact operands"):
        transfer._classes(foreign_parent, computed_root=computed_root)
    forged_mesh = copy(initial.mesh)
    changed = np.asarray(forged_mesh.coordinates, dtype=np.float64).copy()
    changed[0, 0] += 0.125
    object.__setattr__(forged_mesh, "coordinates", jnp.asarray(changed))
    assert forged_mesh.mesh_id == initial.mesh.mesh_id
    forged_parent = copy(initial)
    object.__setattr__(forged_parent, "mesh", forged_mesh)
    with pytest.raises((ValueError, TypeError)):
        _restore_layer_core_accepted_lifecycle(
            combined[0]._replace(result=forged_parent), combined[2], combined[1], source, spec
        )

    wrong_reference = copy(root.reference_mesh)
    changed = np.asarray(wrong_reference.coordinates, dtype=np.float64).copy()
    changed[0, 0] += 0.125
    object.__setattr__(wrong_reference, "coordinates", jnp.asarray(changed))
    wrong_root = copy(root)
    object.__setattr__(wrong_root, "reference_mesh", wrong_reference)
    wrong_target = copy(adaptation.target)
    object.__setattr__(wrong_target, "layer_core_root_geometry", wrong_root)
    wrong_adaptation = copy(adaptation)
    object.__setattr__(wrong_adaptation, "target", wrong_target)
    with pytest.raises((ValueError, TypeError)):
        _layer_core_accepted_lifecycle_recipes(
            initial, prepared, wrong_adaptation, source, spec
        )
    with pytest.raises((ValueError, TypeError)):
        _restore_layer_core_accepted_lifecycle(
            combined[0], combined[2]._replace(result=wrong_target), combined[1], source, spec
        )

    changed_geometry = copy(adaptation.target.geometry)
    changed = np.asarray(changed_geometry.coordinates, dtype=np.float64).copy()
    changed[0, 0] += 0.125
    object.__setattr__(changed_geometry, "coordinates", jnp.asarray(changed))
    changed_target = copy(adaptation.target)
    object.__setattr__(changed_target, "geometry", changed_geometry)
    changed_adaptation = copy(adaptation)
    object.__setattr__(changed_adaptation, "target", changed_target)
    with pytest.raises((ValueError, TypeError)):
        _layer_core_accepted_lifecycle_recipes(
            initial, prepared, changed_adaptation, source, spec
        )
    with pytest.raises((ValueError, TypeError)):
        restore_layer_core_accepted_result(
            combined[2]._replace(result=changed_target), source, spec
        )
    with pytest.raises((ValueError, TypeError)):
        _restore_layer_core_accepted_lifecycle(
            combined[0], combined[2]._replace(result=changed_target), combined[1], source, spec
        )


@pytest.mark.skipif(not meshcore_available(), reason="native layer/core generation requires meshcore")
def test_mapped_inverse_preserves_multiple_source_charts_on_one_parent() -> None:
    import phydrax as phx
    from phydrax._model._structure import MixedAdaptationPolicyArchiveRecipe
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
    from phydrax.meshing._model_codec import restore_mixed_adaptation_policy

    _register_model_archive_types()
    source, spec, initial, prepared, refinement = tiny_affine_mapped_lifecycle()
    initial_recipe, adaptation_recipe, target_recipe = _layer_core_accepted_lifecycle_recipes(
        initial, prepared, refinement, source, spec
    )
    restored_initial, restored_target, restored_refinement, adapter = (
        _restore_layer_core_accepted_lifecycle(
            initial_recipe, target_recipe, adaptation_recipe, source, spec
        )
    )
    assert isinstance(adaptation_recipe.policy, MixedAdaptationPolicyArchiveRecipe)
    policy = restore_mixed_adaptation_policy(adaptation_recipe.policy, adapter)
    assert isinstance(restored_refinement.hierarchy, phx.meshing.MixedAdaptationHierarchy)
    active = {
        int(identifier)
        for block in restored_target.mesh.blocks
        for identifier in np.asarray(block.global_ids)
    }
    children = np.asarray(
        sorted(
            child
            for sibling in restored_refinement.hierarchy.siblings
            for child in sibling.child_ids
            if child in active
        ),
        dtype=np.int64,
    )
    inverse_prepared = phx.meshing.prepare_mesh_adaptation(
        restored_target,
        phx.meshing.MarkedMeshAdaptation((), children, hierarchy=restored_refinement.hierarchy),
        policy=policy,
    )
    inverse = phx.meshing.execute_mesh_adaptation(inverse_prepared)
    assert inverse.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert inverse.target.mesh.topology_id == restored_initial.mesh.topology_id
    assert inverse.target.certification is not None and inverse.target.certification.passed
    witness = inverse.target.certification.inputs.fidelity_witness
    assert isinstance(witness, SourceBoundaryChartWitness)
    charts: dict[int, set[str]] = {}
    for entity, chart in zip(witness.entity_global_ids, witness.source_chart_ids, strict=True):
        charts.setdefault(int(entity), set()).add(chart)
    assert any(len(identities) > 1 for identities in charts.values())
