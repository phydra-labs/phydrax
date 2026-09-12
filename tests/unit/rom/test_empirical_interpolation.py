import numpy as np

import phydrax as phx


def test_empirical_interpolation_reproduces_source_basis():
    cases = tuple(
        phx.rom.ROMCaseSpec(f"case-{index}", (("index", float(index)),))
        for index in range(4)
    )

    def truth(case):
        index = int(dict(case.parameters)["index"])
        state = np.eye(4)[:, index]
        return phx.rom.TruthSample(state, f"truth-{index}")

    corpus = phx.rom.create_corpus(
        cases,
        truth,
        truth_model_id="identity-truth",
        truth_model_revision="exact",
        split=phx.rom.CorpusSplit(tuple(case.case_id for case in cases)),
    )
    artifact = phx.rom.train_profile(corpus, phx.rom.LinearPODProfile(4))
    interpolation = phx.rom.prepare_empirical_interpolation(artifact)
    prepared = interpolation.prepare()
    values = np.asarray([2.0, -1.0, 0.5, 3.0])
    node_values = values[np.asarray(interpolation.node_indices)]

    np.testing.assert_allclose(prepared.interpolate(node_values), values, atol=1e-12)
    assert interpolation.condition_number <= 1.0e10
    assert interpolation.maximum_reproduction_error <= 1.0e-12
