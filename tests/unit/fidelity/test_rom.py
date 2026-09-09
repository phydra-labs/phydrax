#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np

import phydrax as phx


def test_rom_evaluator_exposes_only_reduced_result_as_fidelity():
    cases = tuple(
        phx.rom.ROMCaseSpec(f"train-{index}", (("mu", float(index)),))
        for index in range(3)
    )

    def truth(case):
        mu = dict(case.parameters)["mu"]
        state = np.asarray((1.0, mu))
        return phx.rom.TruthSample(
            state,
            f"truth-{case.case_id}",
            operator=np.eye(2),
            rhs=state,
            qoi_vector=np.asarray((0.0, 1.0)),
            qoi=999.0,
        )

    corpus = phx.rom.create_corpus(
        cases,
        truth,
        truth_model_id="linear-truth",
        truth_model_revision="revision",
        split=phx.rom.CorpusSplit(tuple(case.case_id for case in cases)),
    )
    artifact = phx.rom.train_profile(corpus, phx.rom.LinearPODProfile(2))
    level = phx.fidelity.FidelityLevelSpec(
        "rom",
        problem_id="linear-problem",
        observable_id="state",
        model_id=artifact.artifact_id,
        approximation_id="pod-2",
        observable_contract_id="state-vector",
    )
    evaluator = phx.rom.ROMFidelityEvaluator(
        artifact,
        level,
        truth,
        cost=1.0,
        observable="state",
    )
    evaluation = evaluator(
        phx.fidelity.FidelityCaseSpec(
            {"mu": np.asarray(1.0)},
            case_id="query",
        )
    )

    assert evaluation.result.source == "rom"
    np.testing.assert_allclose(evaluation.observable, np.asarray((1.0, 1.0)))
    assert evaluation.artifact_id == artifact.artifact_id

    qoi_level = phx.fidelity.FidelityLevelSpec(
        "rom-qoi",
        problem_id="linear-problem",
        observable_id="qoi",
        model_id=artifact.artifact_id,
        approximation_id="pod-2",
        observable_contract_id="scalar-qoi",
    )
    qoi_evaluation = phx.rom.ROMFidelityEvaluator(
        artifact,
        qoi_level,
        truth,
        cost=1.0,
        observable="qoi",
    )(
        phx.fidelity.FidelityCaseSpec(
            {"mu": np.asarray(1.0)},
            case_id="qoi-query",
        )
    )
    np.testing.assert_allclose(qoi_evaluation.observable, 1.0)
