#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed model-owned scientific contracts shared by release verification and runners."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..._fingerprint import canonical_fingerprint
from ...qualification._criterion import QualificationCriterion
from ...qualification._evidence import QualificationMatrix


@dataclass(frozen=True, slots=True)
class CampaignMetric:
    name: str
    unit: str
    aggregation: str
    uncertainty: str
    comparison: str
    target_limit: float
    formula: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in (
                self.name,
                self.unit,
                self.aggregation,
                self.uncertainty,
                self.comparison,
                self.formula,
            )
        ):
            raise ValueError(
                "Campaign metric declarations must be canonical identifiers."
            )
        if self.comparison not in (
            "equal",
            "less-than-or-equal",
            "greater-than-or-equal",
        ):
            raise ValueError("Unsupported campaign comparison.")
        if isinstance(self.target_limit, bool) or not math.isfinite(self.target_limit):
            raise ValueError("Campaign targets must be finite, ex-ante numbers.")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "uncertainty": self.uncertainty,
            "comparison": self.comparison,
            "target_limit": self.target_limit,
            "formula": self.formula,
        }

    def validate(self, criterion: QualificationCriterion, /) -> None:
        if (
            criterion.unit,
            criterion.aggregation,
            criterion.uncertainty,
            criterion.comparison,
        ) != (self.unit, self.aggregation, self.uncertainty, self.comparison):
            raise ValueError(
                "Criterion units, aggregation, uncertainty or comparison differ from its registered metric."
            )
        target = criterion.target
        valid = math.isfinite(target)
        if self.comparison == "less-than-or-equal":
            valid = valid and 0.0 <= target <= self.target_limit
        elif self.comparison == "greater-than-or-equal":
            valid = valid and target >= self.target_limit
        else:
            valid = valid and target == self.target_limit
        if not valid:
            raise ValueError("Criterion target violates its finite precommitted limit.")


@dataclass(frozen=True, slots=True)
class CampaignCase:
    case_id: str
    inputs: tuple[tuple[str, object], ...]
    metrics: tuple[CampaignMetric, ...]

    def __post_init__(self) -> None:
        if not self.case_id or self.case_id != self.case_id.strip():
            raise ValueError("Campaign case IDs must be canonical identifiers.")
        if not self.metrics or len({metric.name for metric in self.metrics}) != len(
            self.metrics
        ):
            raise ValueError("Campaign cases require unique nonempty metric matrices.")
        if len(dict(self.inputs)) != len(self.inputs):
            raise ValueError("Campaign case input names must be unique.")
        canonical_fingerprint(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "inputs": dict(self.inputs),
            "metrics": [metric.to_record() for metric in self.metrics],
        }


RESOURCE_METRICS = (
    ("prepare-seconds", "s", "single", "host preparation wall duration"),
    ("lower-seconds", "s", "single", "JAX lowering wall duration"),
    ("compile-seconds", "s", "single", "JAX executable compilation wall duration"),
    (
        "first-seconds",
        "s",
        "single",
        "first complete execution synchronized on all leaves",
    ),
    (
        "warm-p95-seconds",
        "s",
        "p95",
        "linear-interpolated p95 of all synchronized warm samples",
    ),
    ("host-python-peak-bytes", "byte", "maximum", "tracemalloc peak during preparation"),
    (
        "host-process-peak-bytes",
        "byte",
        "maximum",
        "OS process-lifetime peak resident set including the complete workload",
    ),
    (
        "device-executable-bytes",
        "byte",
        "maximum",
        "compiler estimated argument + output + temporary bytes",
    ),
    ("scratch-bytes", "byte", "maximum", "compiler temporary_size_in_bytes"),
    (
        "output-bytes",
        "byte",
        "sum",
        "unique logical bytes in the complete returned trajectory",
    ),
    (
        "checkpoint-bytes",
        "byte",
        "maximum",
        "measured native replay checkpoint payload bytes",
    ),
    ("solver-iterations", "1", "sum", "native attempted step count across all segments"),
    (
        "nonlinear-iterations",
        "1",
        "sum",
        "native nonlinear iteration count across all roots",
    ),
    ("linear-iterations", "1", "sum", "native iterative linear solver iteration count"),
    ("numeric-refreshes", "1", "sum", "native numeric preconditioner refresh count"),
    ("archive-capacity", "1", "maximum", "allocated native accepted-step archive slots"),
    ("event-capacity", "1", "maximum", "allocated native event archive slots"),
    (
        "unsuccessful-executions",
        "1",
        "maximum",
        "1 iff any timed workload execution reports failure",
    ),
    (
        "global-dense-arrays",
        "1",
        "sum",
        "square arguments, constants and intermediates above tiny-oracle dimension "
        "cap 64 in recursively traversed JAXPR",
    ),
)

THERMAL_ECM_SCIENTIFIC_METRICS = (
    CampaignMetric(
        "maximum-normalized-analytic-residual",
        "1",
        "maximum",
        "deterministic",
        "less-than-or-equal",
        1.0e-4,
        "max over saved outputs, states and ledgers of |observed−closed-form|/declared-scale",
    ),
    CampaignMetric(
        "application-status-success",
        "1",
        "all",
        "deterministic",
        "equal",
        1.0,
        "1 iff application status is SUCCESS",
    ),
    CampaignMetric(
        "model-ledger-success",
        "1",
        "all",
        "deterministic",
        "equal",
        1.0,
        "1 iff the scalar model ledger is successful",
    ),
)


def _spme_metric(name, unit, ceiling, formula):
    return CampaignMetric(
        name, unit, "maximum", "deterministic", "less-than-or-equal", ceiling, formula
    )


SPME_SCIENTIFIC_METRICS = {
    "equilibrium": (
        _spme_metric("voltage-drift", "V", 1e-10, "max_t|V(t)-(Up-Un)|"),
        _spme_metric("inventory-drift", "mol", 1e-12, "max_{shell,cell,t}|n(t)-n(0)|"),
    ),
    "current-rest": (
        _spme_metric(
            "transition-current-error", "A", 1e-12, "max_t|I_observed-I_left_hold|"
        ),
        _spme_metric(
            "charge-error", "C", 1e-6, "max_k|F*delta_n_k-sign_k*integral(I dt)|"
        ),
        _spme_metric(
            "execution-contract-failures",
            "1",
            0.0,
            "count(status!=SUCCESS,invalid_output,ledger_failed,missing_provenance,candidate_claims_release)",
        ),
    ),
    "conservation": (
        _spme_metric("solid-lithium-error", "mol", 1e-10, "max_t|n_s(t)-n_s(0)|"),
        _spme_metric("electrolyte-lithium-error", "mol", 1e-10, "max_t|n_e(t)-n_e(0)|"),
        _spme_metric(
            "total-lithium-error", "mol", 1e-10, "max_t|n_s(t)+n_e(t)-n_total(0)|"
        ),
        _spme_metric("collector-flux-error", "mol/m2/s", 1e-12, "max_collector,t|N_e|"),
        _spme_metric(
            "interface-concentration-error",
            "mol/m3",
            1e-8,
            "max_interface,t|ce_left-ce_right|",
        ),
        _spme_metric(
            "equation-mapping-error", "mol/s", 1e-12, "max_cell,t|FV_rate-Eq48_rate|"
        ),
        _spme_metric("current-split-error", "A/m2", 1e-8, "max_face,t|i_s+i_e-i_app|"),
    ),
    **{
        f"{axis}-refinement": (
            _spme_metric(
                "fine-difference", "1", 1e-4, "max_{V,csn,csp,t}|fine-medium|/SI_scale"
            ),
            _spme_metric(
                "contraction-excess",
                "1",
                1e-10,
                "max(0,fine_difference-0.9*coarse_difference)",
            ),
        )
        for axis in ("x", "r", "time")
    },
    "eq49-table6": (
        _spme_metric(
            "eq49-formula-error",
            "1",
            1e-12,
            "max_t|Eq49-(|I|*L/(A*De*F*cmax_n))^2|; affine OCP curvature=0",
        ),
        _spme_metric(
            "applicability-failures",
            "1",
            0.0,
            "count(!Eq49_applicable,!Table6_conditions)",
        ),
    ),
    "fixed-path-jvp-vjp": (
        _spme_metric("duality-error", "1", 1e-9, "|w.Jv-v.J^T w|/max(1,|w.Jv|)"),
        _spme_metric(
            "finite-difference-error",
            "1",
            1e-6,
            "max|Jv-central_difference(h=1e-4)|/SI_scale",
        ),
        _spme_metric(
            "inventory-derivative-error",
            "mol/A",
            1e-10,
            "|d n_negative_final/d I_first-0.5/F|",
        ),
    ),
    "references": (
        _spme_metric("paper-discrepancy", "1", 1e-3, "max|native-paper_fine|/SI_scale"),
        _spme_metric(
            "pybamm-discrepancy",
            "1",
            1e-3,
            "max|native-pybamm_fine_Eq48_projection|/SI_scale",
        ),
        _spme_metric(
            "paper-self-convergence", "1", 1e-4, "max|paper_fine-paper_medium|/SI_scale"
        ),
        _spme_metric(
            "pybamm-self-convergence",
            "1",
            1e-4,
            "max|pybamm_fine-pybamm_medium|/SI_scale",
        ),
        _spme_metric(
            "paper-contraction-excess",
            "1",
            1e-12,
            "max(0,paper_fine_difference-0.9*paper_coarse_difference)",
        ),
        _spme_metric(
            "pybamm-contraction-excess",
            "1",
            1e-12,
            "max(0,pybamm_fine_difference-0.9*pybamm_coarse_difference)",
        ),
        _spme_metric(
            "cross-reference-agreement", "1", 1e-4, "max|paper_fine-pybamm_fine|/SI_scale"
        ),
    ),
}

CIRCUIT_ECM_CASE_IDS = (
    "charge-rest",
    "discharge-rest",
    "resistive-load",
    "voltage-clamp",
)
CIRCUIT_ECM_SCIENTIFIC_METRICS = tuple(
    CampaignMetric(
        name,
        unit,
        "maximum-over-declared-case",
        "closed-form-floating-point-reference",
        comparison,
        target,
        formula,
    )
    for name, unit, comparison, target, formula in (
        (
            "application-status",
            "1",
            "equal",
            0.0,
            "native battery application disposition",
        ),
        (
            "maximum-normalized-analytic-error",
            "1",
            "less-than-or-equal",
            2e-5,
            "max abs(native-closed-form)/declared SI scales",
        ),
        (
            "maximum-kcl-defect-a",
            "A",
            "less-than-or-equal",
            1e-6,
            "max abs(native nodal current balance)",
        ),
        (
            "maximum-power-defect-w",
            "W",
            "less-than-or-equal",
            1e-6,
            "max abs(cell plus boundary terminal power)",
        ),
        (
            "maximum-thermal-defect-w",
            "W",
            "less-than-or-equal",
            1e-6,
            "max abs(augmented thermal residual)",
        ),
        (
            "charge-balance-defect-c",
            "C",
            "less-than-or-equal",
            2e-5,
            "abs(charge change minus integrated actual current)",
        ),
        (
            "thermal-balance-defect-j",
            "J",
            "less-than-or-equal",
            1e-4,
            "abs(stored heat change minus generated heat plus ambient loss)",
        ),
        (
            "physical-initialization-correction",
            "1",
            "equal",
            0.0,
            "max physical state correction during consistency",
        ),
        ("ledger-success", "1", "equal", 1.0, "scalar battery model ledger disposition"),
    )
)

CIRCUIT_ECM_ADVANCED_METRICS = {
    "circuit:standalone-parity": (
        CampaignMetric(
            "maximum-normalized-parity-error",
            "1",
            "maximum",
            "independent-standalone-reference",
            "less-than-or-equal",
            2e-5,
            "max|circuit−standalone|/declared SI scales",
        ),
    ),
    "circuit:terminal-event": (
        CampaignMetric(
            "event-time-refinement-error",
            "s",
            "maximum",
            "independent-time-refinement",
            "less-than-or-equal",
            1e-4,
            "|terminal_event_time(fine)−terminal_event_time(coarse)|",
        ),
        CampaignMetric(
            "event-contract-failures",
            "1",
            "maximum",
            "independent-time-refinement",
            "equal",
            0.0,
            "count(wrong terminal guard,status,bracket,threshold,state continuity or restart)",
        ),
    ),
    "circuit:fixed-path-jvp-vjp": (
        CampaignMetric(
            "duality-error",
            "1",
            "maximum",
            "fixed-accepted-topology",
            "less-than-or-equal",
            1e-9,
            "|w.Jv−v.Jᵀw|/max(1,|w.Jv|)",
        ),
        CampaignMetric(
            "finite-difference-error",
            "1",
            "maximum",
            "fixed-accepted-topology",
            "less-than-or-equal",
            1e-6,
            "max|Jv−central_difference|/declared SI scales",
        ),
        CampaignMetric(
            "derivative-contract-failures",
            "1",
            "maximum",
            "fixed-accepted-topology",
            "equal",
            0.0,
            "count(invalid native derivative,changed accepted/event topology or nonfinite derivative)",
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class BatteryReleaseContract:
    model_id: str
    scientific_cases: tuple[tuple[str, tuple[CampaignMetric, ...]], ...]
    reference_cases: tuple[str, ...]
    reference_count: int
    replay_observables: tuple[tuple[str, str], ...]
    workload_cases: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "battery-required-release-contract",
            "model_id": self.model_id,
            "scientific_cases": {
                case: [metric.to_record() for metric in metrics]
                for case, metrics in self.scientific_cases
            },
            "reference_cases": list(self.reference_cases),
            "reference_count": self.reference_count,
            "replay_observables": dict(self.replay_observables),
            "workload_cases": list(self.workload_cases),
            "resource_metrics": [list(row) for row in RESOURCE_METRICS],
        }

    @property
    def contract_id(self) -> str:
        return canonical_fingerprint(self.to_record())

    @property
    def replay_case_ids(self) -> tuple[str, ...]:
        return tuple(case for case, _ in self.scientific_cases)

    @property
    def replay_keys(self) -> tuple[str, ...]:
        return tuple(
            f"{case}/{name}"
            for case in self.replay_case_ids
            for name, _ in self.replay_observables
        )

    def requirements(self, envelope, replay_limits):
        if set(replay_limits) != set(self.replay_keys):
            raise ValueError(
                "Clean replay must compare every fixed model case and observable."
            )
        required = {}
        for case, metrics in self.scientific_cases:
            for metric in metrics:
                required[(case, metric.name)] = (
                    metric,
                    "reference" if case in self.reference_cases else "scientific",
                )
        for case in self.replay_case_ids:
            for name, unit in self.replay_observables:
                key = f"{case}/{name}"
                limit = replay_limits[key]
                metric = CampaignMetric(
                    f"replay:{name}",
                    unit,
                    "maximum",
                    "independent-clean-execution",
                    "less-than-or-equal",
                    limit,
                    f"max_t|primary({name})−independent_clean_execution({name})| for exact case {case}",
                )
                required[(f"replay:{case}", metric.name)] = (metric, "operational")
        resource_limits = dict(envelope.resource_limits)
        if set(resource_limits) != {name for name, _, _, _ in RESOURCE_METRICS}:
            raise ValueError(
                "Every fixed absolute resource observable needs an ex-ante finite limit."
            )
        for case in self.workload_cases:
            for name, unit, aggregation, formula in RESOURCE_METRICS:
                limit = (
                    0.0
                    if name in ("global-dense-arrays", "unsuccessful-executions")
                    else resource_limits[name]
                )
                required[(case, name)] = (
                    CampaignMetric(
                        name,
                        unit,
                        aggregation,
                        "observed-single-environment",
                        "less-than-or-equal",
                        limit,
                        formula,
                    ),
                    "performance",
                )
        return required

    def validate(
        self, criteria, matrix, envelope, replay, /, *, support_tuple_id, build_id
    ):
        required = self.requirements(envelope, dict(replay.observable_limits))
        if tuple(replay.case_ids) != self.replay_case_ids:
            raise ValueError(
                "Independent replay substituted or omitted a required scientific case."
            )
        actual = {(item.applicability, item.metric): item for item in criteria}
        if len(actual) != len(criteria) or set(actual) != set(required):
            raise ValueError(
                "Release criteria omit, duplicate or substitute fixed model case/metric requirements."
            )
        predicates = {}
        for (case, name), (metric, kind) in required.items():
            criterion = actual[(case, name)]
            if criterion.support_tuple_id != support_tuple_id:
                raise ValueError("Required criterion uses another support tuple.")
            metric.validate(criterion)
            predicates[f"{case}/{name}"] = {
                "evidence_kind": kind,
                "criterion_id": criterion.criterion_id,
                "subject_id": support_tuple_id,
                "build_id": build_id,
                "topology": self.model_id,
            }
        expected = QualificationMatrix(predicates)
        if matrix.matrix_id != expected.matrix_id:
            raise ValueError(
                "Release matrix is not the exact required model case/metric matrix."
            )
        return required


_RELEASE_CONTRACTS = (
    BatteryReleaseContract(
        "battery:ecm:thermal-prescribed-current",
        (("ecm-analytic", THERMAL_ECM_SCIENTIFIC_METRICS),),
        (),
        0,
        (
            ("voltage_v", "V"),
            ("current_a", "A"),
            ("charge_c", "C"),
            ("polarization_voltage_sum_v", "V"),
            ("temperature_k", "K"),
        ),
        ("ecm:scalar", "ecm:batch", "ecm:adjoint"),
    ),
    BatteryReleaseContract(
        "battery:spme:marquis-2019:isothermal-prescribed-current",
        tuple(
            (f"spme:{case}", metrics) for case, metrics in SPME_SCIENTIFIC_METRICS.items()
        ),
        ("spme:references",),
        2,
        (
            ("voltage_v", "V"),
            ("negative_surface_concentration_mol_m3", "mol/m3"),
            ("positive_surface_concentration_mol_m3", "mol/m3"),
            ("total_lithium_mol", "mol"),
            ("eq49:error_estimate", "1"),
            ("current_a", "A"),
        ),
        (
            "spme:small-mesh",
            "spme:medium-mesh",
            "spme:production-mesh",
            "spme:batch",
            "spme:adjoint",
        ),
    ),
    BatteryReleaseContract(
        "battery:ecm:circuit-connected-electrothermal",
        (
            *tuple(
                (case, CIRCUIT_ECM_SCIENTIFIC_METRICS) for case in CIRCUIT_ECM_CASE_IDS
            ),
            *tuple(CIRCUIT_ECM_ADVANCED_METRICS.items()),
        ),
        (),
        0,
        (
            ("voltage_v", "V"),
            ("current_a", "A"),
            ("charge_c", "C"),
            ("polarization_voltage_sum_v", "V"),
            ("temperature_k", "K"),
        ),
        ("circuit-ecm:scalar", "circuit-ecm:batch", "circuit-ecm:adjoint"),
    ),
)


def battery_release_contract(model_id: str, /) -> BatteryReleaseContract:
    for contract in _RELEASE_CONTRACTS:
        if contract.model_id == model_id:
            return contract
    raise ValueError(
        "No complete native release criterion contract is installed for this model."
    )
