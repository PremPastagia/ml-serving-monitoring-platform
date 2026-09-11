# Rollback

Rollback is documented in full in [RETRAINING.md](RETRAINING.md#rollback-verified-through-the-serving-layer),
alongside the promotion path it mirrors. This file is the short operational summary.

## Mechanism

Registry **aliases**, not MLflow stages. An alias is a mutable pointer to an immutable
version, so a rollback is a single atomic repoint rather than a multi-step transition
that can be observed half-applied.

| Alias | Meaning |
|---|---|
| `production` | The version the API serves |
| `previous` | What `production` pointed at before the last change — the known rollback target |
| `candidate` | A freshly retrained model awaiting a decision |

`rollback()` updates `previous` to the version being rolled *away from* before moving
`production`, so a rollback can itself be rolled back. A mistaken rollback is therefore
not a dead end.

## Verified through the running service

```bash
python scripts/rollback_through_serving.py
```

Starts the real uvicorn server, drives four concurrent probe threads issuing
predictions, then promotes and rolls back **while traffic is in flight**:

| Step | From → To | End to end | Requests in flight | Failed |
|---|---|---:|---:|---:|
| Promote | v1 → v2 | 0.063 s | 14 | **0** |
| Rollback | v2 → v1 | 0.082 s | 18 | **0** |

Across both switches: **1,119 requests, 0 failures**, both versions observed in
responses, production correctly restored to v1.

Registry-only alias repoint, measured separately: **0.0055 s**.

## What is claimed

- Rollback works end to end through the live HTTP service — the API reports and serves
  the earlier version afterwards.
- No request failed during the measured switches.

## What is not claimed

**Zero-downtime deployment is not verified.** That is a claim about sustained
production traffic across many switches, restarts and partial failures. What was
measured is 1,119 requests across two switches on one machine.

## Failure paths

Each raises a specific `RegistryError` rather than silently doing nothing, and each is
covered by a test in `tests/test_registry.py`:

| Situation | Behaviour |
|---|---|
| No `production` alias set | `RegistryError: alias 'production' is not set` |
| No `previous` alias recorded | `RegistryError: no 'previous' alias recorded` |
| Target version does not exist | Raises before any alias is moved |
| Target is already the production version | `RegistryError: ... is already the 'production' version` |
| Model reload fails after the alias moves | Previous model keeps serving; `/admin/reload` returns 503 |

## Runbook

```bash
python -c "from mlserve.models.registry import ModelRegistry; print(ModelRegistry().production().to_dict())"
python -c "from mlserve.models.registry import ModelRegistry; print(ModelRegistry().rollback())"
curl -X POST http://127.0.0.1:8077/admin/reload
curl -s http://127.0.0.1:8077/model-info
```
