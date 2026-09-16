#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...measurement._operations import ResolvedConditionSnapshot
from ...particle_physics._operations import HEPRunContext


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class HEPFileReplica(StrictModule, NonTrainableState):
    logical_file_name: str = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    byte_size: int = eqx.field(static=True)
    event_count: int = eqx.field(static=True)
    replica_uris: tuple[str, ...] = eqx.field(static=True)
    file_id: str = eqx.field(static=True)

    def __init__(
        self,
        logical_file_name: str,
        checksum: str,
        /,
        *,
        byte_size: int,
        event_count: int,
        replica_uris: Sequence[str],
    ):
        replicas = tuple(
            sorted(_identifier(value, "Replica URI") for value in replica_uris)
        )
        size = int(byte_size)
        events = int(event_count)
        if size < 0 or events < 0 or not replicas or len(set(replicas)) != len(replicas):
            raise ValueError("HEP file size, event count, and replicas are invalid.")
        self.logical_file_name = _identifier(logical_file_name, "Logical file name")
        self.checksum = _identifier(checksum, "Checksum")
        self.byte_size = size
        self.event_count = events
        self.replica_uris = replicas
        self.file_id = canonical_fingerprint(
            {
                "kind": "hep-file-replica",
                "logical_file_name": self.logical_file_name,
                "checksum": self.checksum,
                "byte_size": size,
                "event_count": events,
                "replicas": list(replicas),
            }
        )


class HEPDatasetSnapshot(StrictModule, NonTrainableState):
    dataset_id: str = eqx.field(static=True)
    external_did: str = eqx.field(static=True)
    files: tuple[HEPFileReplica, ...]
    closed: bool = eqx.field(static=True)
    expected_file_count: int = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        dataset_id: str,
        external_did: str,
        files: Sequence[HEPFileReplica],
        /,
        *,
        closed: bool,
        expected_file_count: int,
    ):
        files_ = tuple(files)
        expected = int(expected_file_count)
        if not files_ or any(not isinstance(value, HEPFileReplica) for value in files_):
            raise TypeError("files must contain typed non-empty HEP replicas.")
        file_ids = tuple(value.file_id for value in files_)
        logical_names = tuple(value.logical_file_name for value in files_)
        if (
            len(set(file_ids)) != len(file_ids)
            or len(set(logical_names)) != len(logical_names)
            or expected < len(files_)
        ):
            raise ValueError("Dataset file identities or expected count are invalid.")
        closed_ = bool(closed)
        if closed_ and expected != len(files_):
            raise ValueError(
                "A closed dataset snapshot must contain every expected file."
            )
        self.dataset_id = _identifier(dataset_id, "Dataset ID")
        self.external_did = _identifier(external_did, "External DID")
        self.files = tuple(sorted(files_, key=lambda value: value.logical_file_name))
        self.closed = closed_
        self.expected_file_count = expected
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "hep-dataset-snapshot",
                "dataset": self.dataset_id,
                "external_did": self.external_did,
                "files": [value.file_id for value in self.files],
                "closed": closed_,
                "expected_file_count": expected,
            }
        )

    @property
    def event_count(self) -> int:
        return sum(value.event_count for value in self.files)

    @property
    def byte_size(self) -> int:
        return sum(value.byte_size for value in self.files)


class WorkloadBackend(StrEnum):
    HTCONDOR = "htcondor"
    SLURM = "slurm"
    KUBERNETES = "kubernetes"
    PANDA = "panda"
    DIRAC = "dirac"
    REANA = "reana"


class HEPSoftwareEnvironment(StrictModule, NonTrainableState):
    source_revision: str = eqx.field(static=True)
    container_digest: str = eqx.field(static=True)
    cvmfs_revisions: tuple[str, ...] = eqx.field(static=True)
    provider_binding_ids: tuple[str, ...] = eqx.field(static=True)
    sbom_checksum: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_revision: str,
        container_digest: str,
        cvmfs_revisions: Sequence[str] = (),
        provider_binding_ids: Sequence[str],
        sbom_checksum: str,
    ):
        cvmfs = tuple(
            sorted(_identifier(value, "CVMFS revision") for value in cvmfs_revisions)
        )
        providers = tuple(
            sorted(
                _identifier(value, "Provider binding ID")
                for value in provider_binding_ids
            )
        )
        if (
            not providers
            or len(set(cvmfs)) != len(cvmfs)
            or len(set(providers)) != len(providers)
        ):
            raise ValueError(
                "Software environment provider/CVMFS identities are invalid."
            )
        self.source_revision = _identifier(source_revision, "Source revision")
        self.container_digest = _identifier(container_digest, "Container digest")
        self.cvmfs_revisions = cvmfs
        self.provider_binding_ids = providers
        self.sbom_checksum = _identifier(sbom_checksum, "SBOM checksum")
        self.environment_id = canonical_fingerprint(
            {
                "kind": "hep-software-environment",
                "source_revision": self.source_revision,
                "container_digest": self.container_digest,
                "cvmfs": list(cvmfs),
                "providers": list(providers),
                "sbom": self.sbom_checksum,
            }
        )


class HEPWorkloadExport(StrictModule, NonTrainableState):
    backend: WorkloadBackend = eqx.field(static=True)
    workset_plan_id: str = eqx.field(static=True)
    dataset_snapshot_id: str = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)
    output_repository_id: str = eqx.field(static=True)
    read_only_credentials: bool = eqx.field(static=True)
    export_id: str = eqx.field(static=True)

    def __init__(
        self,
        backend: WorkloadBackend,
        /,
        *,
        workset_plan_id: str,
        dataset_snapshot: HEPDatasetSnapshot,
        environment: HEPSoftwareEnvironment,
        output_repository_id: str,
        read_only_credentials: bool = True,
    ):
        if (
            not isinstance(backend, WorkloadBackend)
            or not isinstance(dataset_snapshot, HEPDatasetSnapshot)
            or not isinstance(environment, HEPSoftwareEnvironment)
        ):
            raise TypeError(
                "Workload backend, dataset, and environment must be typed values."
            )
        if not dataset_snapshot.closed:
            raise ValueError(
                "Distributed workload export requires a closed dataset snapshot."
            )
        if not bool(read_only_credentials):
            raise ValueError(
                "Phydrax workload exports permit read-only credential handles only."
            )
        self.backend = backend
        self.workset_plan_id = _identifier(workset_plan_id, "Workset plan ID")
        self.dataset_snapshot_id = dataset_snapshot.snapshot_id
        self.environment_id = environment.environment_id
        self.output_repository_id = _identifier(
            output_repository_id, "Output repository ID"
        )
        self.read_only_credentials = True
        self.export_id = canonical_fingerprint(
            {
                "kind": "hep-workload-export",
                "backend": backend.value,
                "workset": self.workset_plan_id,
                "dataset": self.dataset_snapshot_id,
                "environment": self.environment_id,
                "output_repository": self.output_repository_id,
                "read_only_credentials": True,
            }
        )


class HEPPreservationBundle(StrictModule, NonTrainableState):
    run_context: HEPRunContext
    conditions: ResolvedConditionSnapshot
    dataset: HEPDatasetSnapshot
    environment: HEPSoftwareEnvironment
    workload: HEPWorkloadExport
    statistical_model_ids: tuple[str, ...] = eqx.field(static=True)
    output_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    qualification_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    rights_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    external_approval_ids: tuple[str, ...] = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)

    def __init__(
        self,
        run_context: HEPRunContext,
        conditions: ResolvedConditionSnapshot,
        dataset: HEPDatasetSnapshot,
        environment: HEPSoftwareEnvironment,
        workload: HEPWorkloadExport,
        /,
        *,
        statistical_model_ids: Sequence[str],
        output_artifact_ids: Sequence[str],
        qualification_evidence_ids: Sequence[str],
        rights_manifest_ids: Sequence[str],
        external_approval_ids: Sequence[str] = (),
    ):
        if (
            not isinstance(run_context, HEPRunContext)
            or not isinstance(conditions, ResolvedConditionSnapshot)
            or not isinstance(dataset, HEPDatasetSnapshot)
            or not isinstance(environment, HEPSoftwareEnvironment)
            or not isinstance(workload, HEPWorkloadExport)
        ):
            raise TypeError(
                "Preservation bundle requires typed run, conditions, dataset, environment, and workload values."
            )
        if (
            run_context.conditions.snapshot_id != conditions.snapshot_id
            or workload.dataset_snapshot_id != dataset.snapshot_id
            or workload.environment_id != environment.environment_id
        ):
            raise ValueError("Preservation bundle references are inconsistent.")

        def ids(
            values: Sequence[str], name: str, *, required: bool = True
        ) -> tuple[str, ...]:
            result = tuple(sorted(_identifier(value, name) for value in values))
            if (required and not result) or len(set(result)) != len(result):
                raise ValueError(f"{name} values must be unique and satisfy cardinality.")
            return result

        models = ids(statistical_model_ids, "Statistical model ID")
        outputs = ids(output_artifact_ids, "Output artifact ID")
        evidence = ids(qualification_evidence_ids, "Qualification evidence ID")
        rights = ids(rights_manifest_ids, "Rights manifest ID")
        approvals = ids(external_approval_ids, "External approval ID", required=False)
        self.run_context = run_context
        self.conditions = conditions
        self.dataset = dataset
        self.environment = environment
        self.workload = workload
        self.statistical_model_ids = models
        self.output_artifact_ids = outputs
        self.qualification_evidence_ids = evidence
        self.rights_manifest_ids = rights
        self.external_approval_ids = approvals
        self.bundle_id = canonical_fingerprint(
            {
                "kind": "hep-preservation-bundle",
                "run": run_context.context_id,
                "conditions": conditions.snapshot_id,
                "dataset": dataset.snapshot_id,
                "environment": environment.environment_id,
                "workload": workload.export_id,
                "models": list(models),
                "outputs": list(outputs),
                "evidence": list(evidence),
                "rights": list(rights),
                "external_approvals": list(approvals),
            }
        )

    @property
    def externally_approved(self) -> bool:
        return bool(self.external_approval_ids)


__all__ = [
    "HEPDatasetSnapshot",
    "HEPFileReplica",
    "HEPPreservationBundle",
    "HEPSoftwareEnvironment",
    "HEPWorkloadExport",
    "WorkloadBackend",
]
