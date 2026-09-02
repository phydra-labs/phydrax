from __future__ import annotations

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.closure_data._alignment import (
    conservative_prolong,
    conservative_restrict,
    ConservativeAlignmentPlan,
)
from phydrax.closure_data._analysis import (
    ClosureAnalysisDAG,
    ClosureField,
    ClosureQualityReport,
    enthalpy_flux_target,
    sgs_energy_target,
    sgs_stress_target,
    source_target,
    species_flux_target,
)
from phydrax.closure_data._binding import LearnedClosureBindingPlan
from phydrax.closure_data._dataset import (
    ChunkedClosureDatasetManifest,
    ClosureArtifactRepository,
    ClosureDatasetChunk,
    ClosureSample,
    ClosureSampleKey,
    DatasetChunkLayoutError,
    DatasetExtent,
    LeakageSafePartition,
    LeakageSafePartitionPlan,
    PartitionAssignment,
    TrainOnlyNormalizer,
)
from phydrax.closure_data._filters import (
    FavreFilter,
    filter_commutation,
    filter_refinement_commutation,
    FilterSpec,
)
from phydrax.closure_data._state import ClosureSeries, ClosureSnapshot, FlowStateSchema
from phydrax.discretization._axis_domain import AxisDomain
from phydrax.discretization.spectral._basis import FourierBasisPlan
from phydrax.discretization.spectral._coordinates import HermitianSpectralCoordinates
from phydrax.discretization.spectral._dealias import PaddingDealiasingPlan
from phydrax.discretization.spectral._incompressible import PeriodicLerayProjector
from phydrax.discretization.spectral._space import TensorSpectralPlan


def _schema() -> FlowStateSchema:
    return FlowStateSchema(
        ("rho", "u", "v", "species", "enthalpy"),
        ("kg/m^3", "m/s", "m/s", "1", "J/kg"),
        (1.2, 10.0, 10.0, 1.0, 1000.0),
        density_name="rho",
        velocity_names=("u", "v"),
        species_names=("species",),
        enthalpy_name="enthalpy",
    )


def _sample(
    index: int, case: str = "case-a", trajectory: str = "trajectory-a"
) -> ClosureSample:
    key = ClosureSampleKey(
        case_id=case,
        trajectory_id=trajectory,
        realization_id="realization-a",
        time_block_id="block-a",
        time_index=index,
    )
    return ClosureSample(
        jnp.asarray((float(index), float(index + 1))), key, schema_id="schema"
    )


def test_state_dimensionalization_series_and_lineage_are_exact():
    schema = _schema()
    values = jnp.ones((4, 5))
    first = ClosureSnapshot(
        values,
        schema,
        time=0.0,
        case_id="case",
        trajectory_id="trajectory",
        realization_id="realization",
        time_block_id="block",
        mesh_id="mesh",
    )
    dimensional = first.dimensionalize()
    restored = dimensional.nondimensionalize()
    expected_dimensional = jnp.broadcast_to(
        jnp.asarray(schema.reference_scales), values.shape
    )
    np.testing.assert_allclose(dimensional.values, expected_dimensional)
    np.testing.assert_allclose(restored.values, values)
    assert first.snapshot_id in dimensional.parent_ids
    second = ClosureSnapshot(
        2.0 * values,
        schema,
        time=1.0,
        case_id="case",
        trajectory_id="trajectory",
        realization_id="realization",
        time_block_id="block",
        mesh_id="mesh",
    )
    series = ClosureSeries((first, second))
    assert series.stack().shape == (2, 4, 5)
    with pytest.raises(ValueError, match="strictly increasing"):
        ClosureSeries((second, first))


def test_box_filter_preserves_constants_affines_and_linear_algebra():
    prepared = FilterSpec.box((5,), boundary="linear").prepare((17,))
    x = jnp.arange(17.0)
    constant = jnp.full((17,), 3.25)
    np.testing.assert_allclose(prepared(constant), constant, atol=1e-12)
    np.testing.assert_allclose(prepared(2.0 * x - 4.0), 2.0 * x - 4.0, atol=1e-12)
    np.testing.assert_allclose(
        prepared(2.0 * x + 3.0 * constant),
        2.0 * prepared(x) + 3.0 * prepared(constant),
        atol=1e-12,
    )


def test_favre_identities_and_nonpositive_density_rejection():
    prepared = FilterSpec.box((3,), boundary="periodic").prepare((16,))
    favre = FavreFilter(prepared)
    x = jnp.arange(16.0)
    density = 2.0 + 0.1 * jnp.cos(2.0 * jnp.pi * x / 16.0)
    field = jnp.sin(2.0 * jnp.pi * x / 16.0)
    mean_density = favre.mean_density(density)
    filtered = favre(field, density)
    np.testing.assert_allclose(
        mean_density * filtered,
        prepared(density * field),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(favre(jnp.ones_like(field), density), 1.0, atol=1e-12)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="density"):
        jax.block_until_ready(favre(field, density.at[3].set(0.0)))


def test_filter_commutes_with_periodic_difference_and_reports_refinement_defect():
    fine = FilterSpec.box((3,), boundary="periodic").prepare((24,))
    x = 2.0 * jnp.pi * jnp.arange(24.0) / 24.0
    field = jnp.sin(x) + 0.25 * jnp.cos(2.0 * x)
    report = filter_commutation(fine, field, axis=0, spacing=2.0 * np.pi / 24.0)
    np.testing.assert_allclose(report.defect, 0.0, atol=1e-12)
    constant = jnp.ones((24,))
    coarse = FilterSpec.box((3,), boundary="periodic").prepare((12,))
    refinement = filter_refinement_commutation(fine, coarse, constant, (2,))
    np.testing.assert_allclose(refinement.defect, 0.0, atol=1e-12)


def test_restriction_prolongation_and_prepared_alignment_are_conservative():
    fine = jnp.arange(24.0).reshape((6, 4, 1))
    restricted = conservative_restrict(fine, (2, 2))
    np.testing.assert_allclose(
        restricted, jnp.asarray([[[2.5], [4.5]], [[10.5], [12.5]], [[18.5], [20.5]]])
    )
    prolonged = conservative_prolong(restricted, (2, 2))
    assert prolonged.shape == fine.shape
    prepared = ConservativeAlignmentPlan().prepare((6, 4), (3, 2))
    result = prepared.execute(fine)
    np.testing.assert_allclose(result.source_integral, result.target_integral, atol=1e-12)
    np.testing.assert_allclose(result.values, restricted, atol=1e-12)


def test_analysis_dag_target_units_and_lineage_are_deterministic():
    velocity = ClosureField(
        jnp.stack((jnp.arange(8.0), 2.0 * jnp.arange(8.0)), axis=-1),
        name="velocity",
        units="m/s",
        schema_id="schema",
        lineage_ids=("snapshot",),
    )
    density = ClosureField(
        jnp.ones((8,)),
        name="density",
        units="kg/m^3",
        schema_id="schema",
        lineage_ids=("snapshot",),
    )
    species = ClosureField(
        jnp.linspace(0.0, 1.0, 8),
        name="species",
        units="1",
        schema_id="schema",
        lineage_ids=("snapshot",),
    )
    enthalpy = ClosureField(
        jnp.linspace(1.0, 2.0, 8),
        name="enthalpy",
        units="J/kg",
        schema_id="schema",
        lineage_ids=("snapshot",),
    )
    fine_source = ClosureField(
        jnp.linspace(-1.0, 1.0, 8),
        name="fine_source",
        units="kg/(m^3*s)",
        schema_id="schema",
        lineage_ids=("snapshot",),
    )
    resolved_source = ClosureField(
        jnp.zeros((8,)),
        name="resolved_source",
        units="kg/(m^3*s)",
        schema_id="schema",
        lineage_ids=("coarse_snapshot",),
    )
    prepared = FilterSpec.box((3,), boundary="linear").prepare((8,))
    stress = sgs_stress_target(velocity, prepared, density=density)
    energy = sgs_energy_target(stress)
    flux = species_flux_target(velocity, species, prepared, density=density)
    enthalpy_flux = enthalpy_flux_target(velocity, enthalpy, prepared, density=density)
    source = source_target(fine_source, resolved_source, prepared)
    duplicate = sgs_stress_target(velocity, prepared, density=density)
    assert duplicate.node.node_id == stress.node.node_id
    assert duplicate.target_id == stress.target_id
    assert stress.units == "(kg/m^3)*(m/s)^2"
    assert stress.node.node_id in energy.lineage_ids
    assert flux.units == "(kg/m^3)*(m/s)*(1)"
    dag_inputs = (
        velocity.field_id,
        density.field_id,
        species.field_id,
        enthalpy.field_id,
        fine_source.field_id,
        resolved_source.field_id,
    )
    dag_nodes = (
        stress.node,
        energy.node,
        flux.node,
        enthalpy_flux.node,
        source.node,
    )
    dag = ClosureAnalysisDAG(dag_inputs, dag_nodes)
    duplicate_dag = ClosureAnalysisDAG(
        dag_inputs,
        (duplicate.node, *dag_nodes[1:]),
    )
    assert dag.dag_id == duplicate_dag.dag_id
    assert enthalpy_flux.units == "(kg/m^3)*(m/s)*(J/kg)"
    assert source.units == "kg/(m^3*s)"
    assert ClosureQualityReport((stress, energy, flux, enthalpy_flux, source)).passed


def test_chunk_manifest_rejects_holes_and_overlaps_and_uses_repository_protocol():
    extent = DatasetExtent(
        case_id="case",
        trajectory_id="trajectory",
        realization_id="realization",
        time_block_id="block",
        sample_count=4,
    )
    payloads = (b"first", b"second")
    first = ClosureDatasetChunk.from_payload(
        payloads[0],
        extent_id=extent.extent_id,
        logical_name="states",
        chunk_index=0,
        sample_start=0,
        sample_stop=2,
        byte_offset=0,
    )
    second = ClosureDatasetChunk.from_payload(
        payloads[1],
        extent_id=extent.extent_id,
        logical_name="states",
        chunk_index=1,
        sample_start=2,
        sample_stop=4,
        byte_offset=len(payloads[0]),
    )
    manifest = ChunkedClosureDatasetManifest(
        dataset_id="dataset",
        schema_id="schema",
        analysis_dag_id="dag",
        extents=(extent,),
        chunks=(first, second),
    )

    class MemoryRepository:
        def begin(self, artifact_id, writer_id, *, attempt_id=None, started_at=None):
            return (artifact_id, writer_id)

        def write_chunk(
            self,
            transaction,
            logical_name,
            index,
            offset,
            payload,
            *,
            encoding="identity",
        ):
            return (logical_name, index, offset, payload, encoding)

        def commit(self, transaction, chunks, *, metadata=(), committed_at=None):
            return SimpleNamespace(
                artifact_id=transaction[0], chunks=chunks, metadata=metadata
            )

        def get_manifest(self, artifact_id):
            return SimpleNamespace(artifact_id=artifact_id)

        def read_chunk(self, manifest, chunk, *, maximum_plaintext_bytes=None):
            return chunk[3]

    repository = MemoryRepository()
    assert isinstance(repository, ClosureArtifactRepository)
    committed = manifest.write(repository, payloads, writer_id="writer")
    assert committed.artifact_id == "dataset"
    assert manifest.read(repository, committed.chunks) == payloads
    hole = ClosureDatasetChunk.from_payload(
        b"hole",
        extent_id=extent.extent_id,
        logical_name="states",
        chunk_index=1,
        sample_start=3,
        sample_stop=4,
        byte_offset=5,
    )
    overlap = ClosureDatasetChunk.from_payload(
        b"overlap",
        extent_id=extent.extent_id,
        logical_name="states",
        chunk_index=1,
        sample_start=1,
        sample_stop=4,
        byte_offset=5,
    )
    with pytest.raises(DatasetChunkLayoutError, match="hole"):
        ChunkedClosureDatasetManifest(
            dataset_id="dataset",
            schema_id="schema",
            analysis_dag_id="dag",
            extents=(extent,),
            chunks=(first, hole),
        )
    with pytest.raises(DatasetChunkLayoutError, match="overlap"):
        ChunkedClosureDatasetManifest(
            dataset_id="dataset",
            schema_id="schema",
            analysis_dag_id="dag",
            extents=(extent,),
            chunks=(first, overlap),
        )


@pytest.mark.parametrize("level", ("case", "trajectory", "realization", "time_block"))
def test_partitioning_never_splits_the_selected_leakage_group(level):
    samples = tuple(_sample(index) for index in range(5))
    plan = LeakageSafePartitionPlan(
        level,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        salt="experiment",
    )
    partition = plan.assign(samples)
    assert len({assignment.split for assignment in partition.assignments}) == 1
    assert len({assignment.group_key for assignment in partition.assignments}) == 1


def test_normalizer_statistics_and_provenance_are_train_only():
    samples = (_sample(0), _sample(2), _sample(100))
    assignments = (
        PartitionAssignment(
            sample_id=samples[0].sample_id, group_key=("train-a",), split="train"
        ),
        PartitionAssignment(
            sample_id=samples[1].sample_id, group_key=("train-b",), split="train"
        ),
        PartitionAssignment(
            sample_id=samples[2].sample_id, group_key=("test",), split="test"
        ),
    )
    plan = LeakageSafePartitionPlan(
        "case",
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
        salt="normalizer",
    )
    partition = LeakageSafePartition(plan, assignments)
    normalizer = TrainOnlyNormalizer.fit(samples, partition, feature_name="state")
    np.testing.assert_allclose(normalizer.mean, jnp.asarray((1.0, 2.0)))
    assert set(normalizer.provenance.training_sample_ids) == {
        samples[0].sample_id,
        samples[1].sample_id,
    }
    assert samples[2].sample_id not in normalizer.provenance.training_sample_ids
    values = jnp.asarray((3.0, 4.0))
    np.testing.assert_allclose(
        normalizer.denormalize(normalizer.normalize(values)), values
    )


def test_binding_rejects_schema_mismatch_and_inserts_face_correction():
    schema = _schema()

    def correction(system, left, right, baseline, axis, args):
        del system, baseline, axis
        return args * (right - left)

    binding = LearnedClosureBindingPlan(
        correction,
        deployment_kind="conservative_face",
        schema_id=schema.schema_id,
        input_component_names=schema.component_names,
        output_component_names=schema.component_names,
        model_artifact_id="model",
        normalizer_provenance_id="normalizer",
    )
    plan = binding.bind_conservative_faces(schema)
    left = jnp.arange(5.0)
    right = left + 1.0
    baseline = jnp.full((5,), 2.0)
    result = plan.apply(
        SimpleNamespace(component_names=schema.component_names),
        left,
        right,
        baseline,
        0,
        0.25,
    )
    np.testing.assert_allclose(result, 2.25)
    other = FlowStateSchema(("rho",), ("kg/m^3",), (1.0,), density_name="rho")
    with pytest.raises(ValueError, match="schema identity"):
        binding.bind_conservative_faces(other)


def _spectral_contract():
    space = TensorSpectralPlan(
        (FourierBasisPlan(6), FourierBasisPlan(6)),
        axis_names=("x", "y"),
        field_name="velocity",
    ).prepare(
        (
            AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            AxisDomain.periodic(0.0, 2.0 * jnp.pi),
        )
    )
    projector = PeriodicLerayProjector(space)
    coordinates = HermitianSpectralCoordinates(space, component_shape=(2,))
    dealiasing = PaddingDealiasingPlan(2).prepare(space, required_polynomial_degree=2)
    x, y = jnp.meshgrid(space.axes[0].nodes, space.axes[1].nodes, indexing="ij")
    physical = jnp.stack((jnp.sin(y), jnp.sin(x)), axis=-1)
    state = projector.project(coordinates.project(space.project(physical)))
    schema = FlowStateSchema(
        ("u", "v"),
        ("m/s", "m/s"),
        (1.0, 1.0),
        velocity_names=("u", "v"),
    )
    return space, projector, coordinates, dealiasing, state, schema


def test_spectral_binding_preserves_energy_hermitian_projection_and_dealiasing_evidence():
    _, projector, coordinates, dealiasing, state, schema = _spectral_contract()
    binding = LearnedClosureBindingPlan(
        lambda value, args: args * value,
        deployment_kind="spectral_drift",
        schema_id=schema.schema_id,
        input_component_names=("u", "v"),
        output_component_names=("u", "v"),
        model_artifact_id="spectral-model",
        normalizer_provenance_id="normalizer",
    )
    hook = binding.bind_spectral_drift(schema, projector, coordinates, dealiasing)
    result = hook.apply(state, 1.0)
    assert float(result.evidence.constrained_energy_rate) <= 1e-10
    assert float(result.evidence.divergence_norm) <= 1e-10
    assert float(result.evidence.hermitian_defect) <= 1e-10
    assert result.evidence.projector_id == projector.projector_id
    assert result.evidence.dealiasing_id == dealiasing.prepared_id
    assert result.evidence.dealiasing_exact
    assert not bool(result.fallback.used)


def test_spectral_nonfinite_prediction_returns_explicit_typed_fallback_artifact():
    _, projector, coordinates, dealiasing, state, schema = _spectral_contract()
    binding = LearnedClosureBindingPlan(
        lambda value, args: jnp.full_like(value, jnp.nan + 0.0j),
        deployment_kind="spectral_drift",
        schema_id=schema.schema_id,
        input_component_names=("u", "v"),
        output_component_names=("u", "v"),
        model_artifact_id="bad-model",
        normalizer_provenance_id="normalizer",
    )
    hook = binding.bind_spectral_drift(schema, projector, coordinates, dealiasing)
    result = hook.apply(state)
    np.testing.assert_allclose(result.drift, 0.0)
    assert bool(result.fallback.used)
    assert int(result.fallback.reason_code) == 1
    assert result.fallback.fallback_kind == "zero_spectral_drift"
    assert not bool(result.evidence.valid)
