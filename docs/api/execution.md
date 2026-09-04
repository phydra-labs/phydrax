# Execution worksets

`phydrax.execution` groups exact homogeneous execution signatures into deterministic,
fixed-capacity worksets. It owns canonical ordering, reversible gather/scatter,
serial/vectorized equivalence, semantic restartable RNG keys, finite/coverage evidence,
and content-addressed checkpoints. It does not expose a distributed path when no real
multi-device qualification exists.

::: phydrax.execution.ExecutionWorksetPlan

---

::: phydrax.execution.PreparedExecutionWorksets

---

::: phydrax.execution.evaluate_execution_worksets_serial

---

::: phydrax.execution.evaluate_execution_worksets_vmap

---

::: phydrax.execution.ExecutionWorksetCheckpoint

---

::: phydrax.execution.restore_execution_workset_checkpoint
