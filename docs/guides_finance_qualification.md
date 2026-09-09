# Finance qualification and replay

Finance qualification is attached to an exact route coordinate, never to an entire product family by implication. The route builders return ordinary `phydrax.qualification.SupportTuple` records for market resolution, curve calibration, valuation, econometrics, portfolio construction, exposure/XVA, execution, and candidate advanced finance.

```python
from phydrax.finance.qualification import (
    build_finance_qualification_matrix,
    evaluate_finance_campaign,
    valuation_support,
)

support = valuation_support(
    "analytic",
    product="european-option",
    model="black-scholes",
    pricing_law="synthetic-usd-q",
)
matrix = build_finance_qualification_matrix((support,))
campaign = evaluate_finance_campaign(matrix, (), at_time=1_800_000_000)
assert campaign.coverage.outcome == "inconclusive"
```

The empty evidence sequence is explicitly inconclusive. Replace it with existing reviewed `QualificationEvidence` records; evidence is data, not an executable callback. Every support tuple receives four independent predicates:

1. reference evidence for data and source claims;
2. scientific evidence for model claims;
3. unit evidence for numerical behavior;
4. operational evidence for the named use.

Missing, future, expired, or superseded evidence is inconclusive. An active failure is failed. Only a complete conjunction passes. Performance records may be retained as observations, but timing never decides finance qualification.

## Result archives

`finance_result_manifest` binds each named physical array to an explicit unit and canonical payload digest. `archive_finance_result` then binds that lifecycle manifest to a replay ID, exact support tuples, and any P/Q/stress law IDs. `reopen_finance_result` performs bounded, pickle-free archive admission and independently checks array, manifest, replay, support, and content identities.

```python
import jax.numpy as jnp

from phydrax.finance.qualification import (
    archive_finance_result,
    finance_result_manifest,
    reopen_finance_result,
)

arrays = {
    "present_value": jnp.asarray([10.5]),
    "valid": jnp.asarray([True]),
}
manifest = finance_result_manifest(
    "synthetic-result",
    "synthetic-run",
    arrays,
    {"present_value": "USD", "valid": "1"},
)
archive_finance_result(
    "synthetic-result.phx",
    result_manifest=manifest,
    arrays=arrays,
    replay_id="synthetic-replay",
    support_tuples=(support,),
    law_ids=("synthetic-usd-q",),
)
reopened = reopen_finance_result("synthetic-result.phx")
```

The archive contains arrays and inert records only. It does not contain Python objects, code, callbacks, external commands, feeds, or network locations.

## Local runner

`tools/finance_qualification.py` reads one local JSON object containing `support_tuples`, `evidence`, and `evaluated_at`. It writes the complete campaign record and exits nonzero unless the campaign passes. The runner never invokes an oracle or external executable.
