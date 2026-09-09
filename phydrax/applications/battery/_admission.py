#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-side, time-scoped admission, reverified before production dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..._fingerprint import canonical_fingerprint
from ...qualification._evidence import SupportDependency
from ...qualification._registry import (
    CapabilityProfile,
    ReleaseIndex,
    require_profile,
    SupportTuple,
)
from ...qualification._runtime_distribution import RuntimeDistributionAttestation
from ...qualification._trust import AsymmetricReleaseTrustPolicy
from ._qualification import validate_battery_candidate_profile
from ._release import BatteryReleaseRecord
from ._validity import BatteryValidityEnvelope


@dataclass(frozen=True, slots=True)
class BatteryExecutionAdmission:
    index: ReleaseIndex
    profile: CapabilityProfile
    support_tuple: SupportTuple
    envelope: BatteryValidityEnvelope
    distribution_id: str
    issued_at: int
    expires_at: int
    trust_policy: AsymmetricReleaseTrustPolicy = field(repr=False, compare=False)
    predictive_profile_id: str | None = None
    predictive_support: SupportTuple | None = None
    parameter_manifest_id: str | None = None
    runtime_attestation: RuntimeDistributionAttestation | None = None

    @property
    def profile_id(self) -> str:
        return self.profile.profile_id

    @property
    def support_tuple_id(self) -> str:
        return self.support_tuple.support_tuple_id

    @property
    def envelope_id(self) -> str:
        return self.envelope.envelope_id

    @property
    def numerical_admission_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "battery-numerical-admission",
                "index_id": self.index.index_id,
                "profile_id": self.profile_id,
                "support_tuple_id": self.support_tuple_id,
                "envelope_id": self.envelope_id,
                "distribution_id": self.distribution_id,
                "issued_at": self.issued_at,
                "expires_at": self.expires_at,
                "runtime_attestation_id": None
                if self.runtime_attestation is None
                else self.runtime_attestation.signature.attestation_id,
            }
        )

    @property
    def predictive_admission_id(self) -> str | None:
        if self.predictive_profile_id is None:
            return None
        return canonical_fingerprint(
            {
                "kind": "battery-predictive-admission",
                "numerical_admission_id": self.numerical_admission_id,
                "profile_id": self.predictive_profile_id,
                "support_tuple_id": self.predictive_support.support_tuple_id,
                "parameter_manifest_id": self.parameter_manifest_id,
            }
        )

    def validate_adapter(self, model, /) -> None:
        """Refuse model-ID lookalikes and subclasses outside the signed implementation."""
        from ._circuit_ecm import CircuitConnectedEcmAdapter
        from ._ecm import ThermalEquivalentCircuitAdapter
        from ._spme_marquis2019 import Marquis2019SpmeAdapter

        implementations = {
            "battery:ecm:thermal-prescribed-current": ThermalEquivalentCircuitAdapter,
            "battery:spme:marquis-2019:isothermal-prescribed-current": Marquis2019SpmeAdapter,
            "battery:ecm:circuit-connected-electrothermal": CircuitConnectedEcmAdapter,
        }
        implementation = implementations.get(self.envelope.model_id)
        if implementation is None or type(model) is not implementation:
            raise ValueError(
                "Production adapter is not the exact authenticated model implementation."
            )
        if (
            model.model_id != self.envelope.model_id
            or model.equation_form != dict(self.support_tuple.attributes)["equation_form"]
        ):
            raise ValueError(
                "Production adapter model/equation coordinates were substituted."
            )

    def validate_run(
        self,
        prepared_model,
        parameters,
        initial_condition,
        protocol,
        protocol_values,
        /,
        *,
        at_time: int | None = None,
    ) -> None:
        """Enforce finite ex-ante numeric leaves, layout, schedule and model domains."""
        import jax
        import numpy as np

        from ..._fingerprint import array_tree_fingerprint

        _verify_admission(self, int(time.time()) if at_time is None else at_time)
        self.envelope.require_production_bounds()

        def check_tree(tree, declared, group):
            bounds = {name: (lower, upper) for name, lower, upper in declared}
            actual = {}
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
                value = np.asarray(leaf)
                if value.dtype.kind not in "fciub":
                    raise ValueError(f"{group} contains a nonnumeric dynamic leaf.")
                if value.dtype.kind in "fc" and value.dtype != np.dtype("float64"):
                    raise ValueError(
                        "Production battery execution is qualified only for float64."
                    )
                devices = getattr(leaf, "devices", lambda: ())()
                if any(device.platform != "cpu" for device in devices):
                    raise ValueError(
                        "Production battery execution is qualified only for CPU."
                    )
                actual[jax.tree_util.keystr(path) or "<root>"] = value
            if set(actual) != set(bounds):
                raise ValueError(
                    f"{group} leaves are not exactly covered by the finite release envelope."
                )
            for name, value in actual.items():
                lower, upper = bounds[name]
                if not np.all(np.isfinite(value) & (value >= lower) & (value <= upper)):
                    raise ValueError(
                        f"{group} coordinate {name} is outside its finite release envelope."
                    )

        check_tree(parameters, self.envelope.production_parameter_bounds, "parameters")
        check_tree(
            initial_condition,
            self.envelope.production_initial_bounds,
            "initial condition",
        )
        check_tree(
            protocol_values, self.envelope.production_protocol_bounds, "protocol values"
        )
        if protocol_values.protocol_id != protocol.protocol_id:
            raise ValueError(
                "Production protocol values belong to another held schedule."
            )
        boundaries = np.asarray(protocol.boundary_times_s)
        if (
            boundaries.dtype != np.dtype("float64")
            or not np.all(np.isfinite(boundaries))
            or not np.all(np.diff(boundaries) > 0)
        ):
            raise ValueError(
                "Production protocol boundaries require finite ordered float64 times."
            )
        declared_layout = {
            name: (lower, upper)
            for name, lower, upper in self.envelope.production_layout_bounds
        }
        required_layout = {
            f"plan.{name}"
            for name, value in vars(prepared_model.plan).items()
            if type(value) in (int, float)
        }
        required_layout.add("protocol.duration_s")
        if not required_layout <= declared_layout.keys():
            raise ValueError(
                "Prepared static layout or total horizon lacks finite precommitted limits."
            )
        for name, (lower, upper) in declared_layout.items():
            if name == "protocol.duration_s":
                value = boundaries[-1] - boundaries[0]
            else:
                value = prepared_model
                for component in name.split("."):
                    if not component or component.startswith("_"):
                        raise ValueError(
                            "Production layout paths must name public numeric attributes."
                        )
                    value = getattr(value, component)
                value = np.asarray(value)
            if not np.all(np.isfinite(value) & (value >= lower) & (value <= upper)):
                raise ValueError(f"Prepared layout {name} exceeds its release envelope.")
        if self.predictive_profile_id is not None:
            # The product manifest must bind actual numeric parameters, not a
            # structural parameter_id shared by multiple parameterizations.
            attributes = dict(self.predictive_support.attributes)
            if (
                attributes.get("parameter_digest")
                != array_tree_fingerprint(parameters)["sha256"]
            ):
                raise ValueError(
                    "Predictive admission does not bind actual parameter bytes."
                )
        if (
            self.envelope.model_id
            == "battery:spme:marquis-2019:isothermal-prescribed-current"
        ):
            from ._spme_marquis2019 import validate_marquis2019_spme_execution

            validate_marquis2019_spme_execution(
                prepared_model, parameters, initial_condition, protocol, protocol_values
            )
        elif self.envelope.model_id in (
            "battery:ecm:thermal-prescribed-current",
            "battery:ecm:circuit-connected-electrothermal",
        ):
            _validate_ecm_run(
                prepared_model, parameters, initial_condition, protocol, protocol_values
            )
        else:
            raise ValueError(
                "This numerical model has no installed production preflight implementation."
            )


def _verify_admission(admission: BatteryExecutionAdmission, at_time: int) -> int:
    import jax

    if not jax.core.trace_ctx.is_top_level():
        raise ValueError(
            "Production admission requires a fresh host dispatch; "
            "transformed/cached whole-run execution is not authorized."
        )
    if not isinstance(admission.trust_policy, AsymmetricReleaseTrustPolicy):
        raise TypeError("Production execution requires asymmetric release trust.")
    if (
        type(at_time) is not int
        or type(admission.issued_at) is not int
        or type(admission.expires_at) is not int
        or not admission.issued_at <= at_time < admission.expires_at
    ):
        raise ValueError("Execution admission is stale or not active.")
    if not isinstance(admission.runtime_attestation, RuntimeDistributionAttestation):
        raise ValueError(
            "Production admission requires an authenticated installed-runtime attestation."
        )
    runtime_expiry = admission.runtime_attestation.verify(
        admission.trust_policy.roles,
        distribution_id=admission.distribution_id,
        at_time=at_time,
    )
    profile = require_profile(
        admission.index,
        admission.profile_id,
        admission.support_tuple,
        admission.trust_policy,
        at_time=at_time,
    )
    validate_battery_candidate_profile(profile, admission.support_tuple)
    if not profile.released or profile.to_record() != admission.profile.to_record():
        raise ValueError("Admission profile was substituted.")
    proofs = [
        proof
        for proof in admission.trust_policy.proofs
        if isinstance(proof, BatteryReleaseRecord)
        and proof.profile.profile_id == profile.profile_id
    ]
    if len(proofs) != 1:
        raise ValueError(
            "Production admission requires the retained typed release proof."
        )
    proof = proofs[0]
    proof.verify(admission.trust_policy, at_time=at_time)
    if (
        proof.distribution_id != admission.distribution_id
        or proof.envelope_id != admission.envelope_id
    ):
        raise ValueError("Admission distribution or validity envelope is mismatched.")
    cap = min(
        proof.expires_at, admission.index.issued_at + admission.trust_policy.max_index_age
    )
    roots = {
        record.key_id: record for record in admission.trust_policy.roles.store.records()
    }
    cap = min(cap, roots[admission.index.signer_id].expires_at)
    if admission.predictive_profile_id is None:
        if (
            admission.predictive_support is not None
            or admission.parameter_manifest_id is not None
        ):
            raise ValueError(
                "Predictive coordinates cannot be attached to equation-only admission."
            )
    else:
        support = admission.predictive_support
        if (
            not isinstance(support, SupportTuple)
            or support.capability != "battery.predictive-simulation"
        ):
            raise ValueError("A separate exact predictive support profile is required.")
        attributes = dict(support.attributes)
        required = {
            "model_id",
            "numerical_profile_id",
            "parameter_manifest_id",
            "parameter_digest",
            "data_manifest_id",
            "chemistry",
            "form_factor",
            "calibration_manifest_id",
            "locked_validation_split_id",
            "uncertainty_model_id",
            "model_discrepancy_id",
            "rights_id",
            "validity_envelope_id",
        }
        if (
            set(attributes) != required
            or attributes["numerical_profile_id"] != profile.profile_id
            or attributes["parameter_manifest_id"] != admission.parameter_manifest_id
            or attributes["model_id"] != admission.envelope.model_id
        ):
            raise ValueError(
                "Predictive support lacks exact product/calibration/validation/rights bindings."
            )
        predictive = require_profile(
            admission.index,
            admission.predictive_profile_id,
            support,
            admission.trust_policy,
            at_time=at_time,
        )
        dependency = SupportDependency(profile.profile_id, admission.support_tuple_id)
        if not any(
            isinstance(item, SupportDependency)
            and item.dependency_id == dependency.dependency_id
            for item in predictive.dependencies
        ):
            raise ValueError("Predictive profile lacks the exact numerical dependency.")
        for gate in predictive.release_evidence:
            cap = min(cap, gate.expires_at)
    cap = min(cap, runtime_expiry)
    if admission.expires_at > cap:
        raise ValueError("Execution admission outlives a cited proof or signing root.")
    return cap


def issue_battery_execution_admission(
    index: ReleaseIndex,
    profile_id: str,
    support_tuple: SupportTuple,
    trust_policy: AsymmetricReleaseTrustPolicy,
    /,
    *,
    envelope: BatteryValidityEnvelope,
    distribution_id: str,
    at_time: int,
    expires_at: int,
    predictive_profile_id: str | None = None,
    predictive_support: SupportTuple | None = None,
    parameter_manifest_id: str | None = None,
    runtime_attestation: RuntimeDistributionAttestation | None = None,
) -> BatteryExecutionAdmission:
    if not isinstance(trust_policy, AsymmetricReleaseTrustPolicy) or not isinstance(
        envelope, BatteryValidityEnvelope
    ):
        raise TypeError(
            "Admission requires asymmetric release trust and an exact typed envelope."
        )
    profile = require_profile(
        index, profile_id, support_tuple, trust_policy, at_time=at_time
    )
    admission = BatteryExecutionAdmission(
        index,
        profile,
        support_tuple,
        envelope,
        distribution_id,
        at_time,
        expires_at,
        trust_policy,
        predictive_profile_id,
        predictive_support,
        parameter_manifest_id,
        runtime_attestation,
    )
    _verify_admission(admission, at_time)
    return admission


def validate_battery_execution_admission(
    profile: CapabilityProfile,
    support_tuple: SupportTuple,
    admission: BatteryExecutionAdmission | None,
    /,
    *,
    distribution_id: str | None,
    at_time: int | None = None,
) -> BatteryExecutionAdmission | None:
    validate_battery_candidate_profile(profile, support_tuple)
    if not profile.released:
        if admission is not None:
            raise ValueError(
                "Candidate execution can never report numerical or predictive release admission."
            )
        return None
    if not isinstance(admission, BatteryExecutionAdmission):
        raise ValueError(
            "Production battery execution requires a verified time-scoped admission."
        )
    if (profile.profile_id, support_tuple.support_tuple_id, distribution_id) != (
        admission.profile_id,
        admission.support_tuple_id,
        admission.distribution_id,
    ):
        raise ValueError(
            "Execution does not match the admitted profile/support/distribution."
        )
    _verify_admission(admission, int(time.time()) if at_time is None else at_time)
    return admission


def battery_release_deployment_record(
    index: ReleaseIndex, proofs, /
) -> dict[str, object]:
    """Serialize an index and its retained typed proof DAG, never private keys."""
    return {
        "kind": "battery-release-deployment",
        "index": index.to_record(),
        "proofs": [proof.to_record() for proof in proofs],
    }


def load_battery_execution_admission(
    deployment,
    public_trust_config,
    /,
    *,
    profile_id: str,
    support_tuple_id: str,
    distribution_id: str,
    at_time: int,
    expires_at: int,
    max_index_age: int,
    runtime_attestation: RuntimeDistributionAttestation | None = None,
) -> BatteryExecutionAdmission:
    """Load offline retained proofs against separately provisioned public roots."""
    from ...qualification._trust import QualificationRoleTrust
    from ._dfn_entry import DfnEntryReleaseRecord
    from ._pack_entry import SeriesPackEntryReleaseRecord

    if (
        set(deployment) != {"kind", "index", "proofs"}
        or deployment["kind"] != "battery-release-deployment"
    ):
        raise ValueError("Invalid deployment record; a signed index alone is not proof.")
    proof_types = {
        "battery-release-record": BatteryReleaseRecord,
        "battery-dfn-entry-release": DfnEntryReleaseRecord,
        "series-pack-entry-release-record": SeriesPackEntryReleaseRecord,
    }
    proofs = []
    for record in deployment["proofs"]:
        kind = proof_types.get(record.get("kind"))
        if kind is None:
            raise ValueError("Unsupported or opaque deployment proof.")
        proofs.append(kind.from_record(record))
    if len({proof.gate_id for proof in proofs}) != len(proofs):
        raise ValueError("Deployment contains duplicate release proofs.")
    roles = QualificationRoleTrust.from_public_config(public_trust_config)
    policy = AsymmetricReleaseTrustPolicy(
        roles, proofs=proofs, max_index_age=max_index_age
    )
    index = ReleaseIndex.from_record(deployment["index"])
    matches = [
        proof
        for proof in proofs
        if isinstance(proof, BatteryReleaseRecord)
        and proof.profile.profile_id == profile_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "Requested numerical profile has no exact retained release proof."
        )
    proof = matches[0]
    support = next(
        (
            item
            for item in proof.profile.support_tuples
            if item.support_tuple_id == support_tuple_id
        ),
        None,
    )
    if support is None:
        raise ValueError("Requested support is absent from the retained release proof.")
    return issue_battery_execution_admission(
        index,
        profile_id,
        support,
        policy,
        envelope=proof.bundle.envelope,
        distribution_id=distribution_id,
        at_time=at_time,
        expires_at=expires_at,
        runtime_attestation=runtime_attestation,
    )


def _validate_ecm_run(prepared_model, parameters, initial, protocol, values):
    import numpy as np

    from ._circuit_ecm import PreparedCircuitConnectedEcm
    from ._ecm import PreparedThermalEquivalentCircuit, ThermalEquivalentCircuitParameters
    from ._properties import ConstantPropertyLaw, TabulatedPropertyLaw

    if type(prepared_model) not in (
        PreparedThermalEquivalentCircuit,
        PreparedCircuitConnectedEcm,
    ):
        raise TypeError(
            "Production ECM preparation must use the exact installed implementation."
        )
    if (
        not isinstance(parameters, ThermalEquivalentCircuitParameters)
        or prepared_model.plan.branch_count != 1
    ):
        raise ValueError(
            "Production ECM requires the exact one-branch thermal parameter owner."
        )
    for name in (
        "series_resistance_ohm",
        "branch_resistances_ohm",
        "branch_capacitances_f",
        "reference_capacity_c",
        "heat_capacity_j_per_k",
        "thermal_conductance_w_per_k",
        "ambient_temperature_k",
        "reference_temperature_k",
    ):
        if not np.all(np.asarray(getattr(parameters, name)) > 0):
            raise ValueError(
                f"Production ECM parameter {name} must remain strictly positive."
            )
    if parameters.branch_resistances_ohm.shape != (
        1,
    ) or parameters.branch_capacitances_f.shape != (1,):
        raise ValueError("Production ECM parameter branch topology was substituted.")
    charge = np.asarray(initial.charge_c)
    temperature = np.asarray(initial.temperature_k)
    if charge.shape != () or temperature.shape != () or not temperature > 0:
        raise ValueError(
            "Production ECM requires one physical positive-temperature cell."
        )
    if not initial.relaxed and initial.polarization_voltages_v.shape != (1,):
        raise ValueError("Production ECM initial RC topology is mismatched.")
    boundaries = np.asarray(protocol.boundary_times_s)
    indices = np.asarray(protocol.interval_current_indices)
    amplitudes = np.asarray(values.current_amplitudes_a)
    currents = np.asarray([0.0 if index < 0 else amplitudes[index] for index in indices])
    prescribed = (
        type(prepared_model) is PreparedThermalEquivalentCircuit
        or prepared_model.plan.prescribed_current
    )
    charges = (
        np.concatenate(
            (charge.reshape(1), charge + np.cumsum(currents * np.diff(boundaries)))
        )
        if prescribed
        else charge.reshape(1)
    )
    soc = charges / np.asarray(parameters.reference_capacity_c)
    if not np.all((soc >= 0) & (soc <= 1)):
        raise ValueError(
            "The complete prescribed-current schedule leaves the cell inventory envelope."
        )
    lower, upper = float(np.min(soc)), float(np.max(soc))
    if not np.array_equal(
        np.asarray(parameters.open_circuit_voltage.support_bounds),
        np.asarray(parameters.entropic_coefficient.support_bounds),
    ):
        raise ValueError("ECM property support intervals are mismatched.")
    for law in (parameters.open_circuit_voltage, parameters.entropic_coefficient):
        queries = [lower, upper]
        if isinstance(law, TabulatedPropertyLaw):
            nodes = np.asarray(law.nodes)
            if not np.all(np.diff(nodes) > 0):
                raise ValueError("ECM property nodes are not ordered.")
            interior = nodes[(nodes > lower) & (nodes < upper)]
            points = np.concatenate(([lower], interior, [upper]))
            queries.extend(interior.tolist())
            queries.extend((0.5 * (points[:-1] + points[1:])).tolist())
            active_values = np.asarray(law.values)[np.asarray(law.source_mask)]
        elif isinstance(law, ConstantPropertyLaw):
            active_values = np.asarray(law.value)
        else:
            raise TypeError("Production ECM properties require a supported bounded law.")
        bounds = np.asarray(law.value_bounds)
        if not np.all(
            np.isfinite(active_values)
            & (active_values >= bounds[0])
            & (active_values <= bounds[1])
        ):
            raise ValueError("ECM property values violate their declared bounds.")
        if not np.all(np.asarray(law.evaluate(np.asarray(queries)).support)):
            raise ValueError(
                "The current schedule crosses unsupported property intervals or masked holes."
            )
