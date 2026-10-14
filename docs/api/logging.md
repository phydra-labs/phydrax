# Logging

PhydraX event logging is disabled by default and never owns application handlers.
See [Structured logging and observability](../guides_logging.md) for the event,
privacy, and distributed-execution contracts.

::: phydrax.logging.LogValue

---

Emission is process-wide configuration; sinks and correlation stay here:

```python executable
import phydrax as phx

phx.config.configure(phx.config.LoggingConfig(enabled=True))
assert phx.config.logging_enabled()
phx.config.configure(phx.config.LoggingConfig(enabled=False))
```

See [Configuration](configuration.md) for the configuration records and observers.

::: phydrax.logging.context

---

::: phydrax.logging.emit

---

::: phydrax.logging.add_text_sink

---

::: phydrax.logging.add_json_sink

---

::: phydrax.logging.remove_sink
