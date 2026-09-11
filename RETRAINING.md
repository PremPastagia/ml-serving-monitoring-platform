# Retraining, Promotion and Rollback

## Workflow

```
current window
  → drift detection
  → (if drift) validate the proposed training data
  → retrain a candidate
  → score candidate AND incumbent on the SAME holdout
  → time both models by interleaving their requests
  → apply six acceptance criteria
  → promote (alias moves, previous kept) or reject (registered, not aliased)
```

**Triggered explicitly**, by `scripts/retrain_experiment.py` or a test. There is no
scheduler and no automatic production trigger — that is a deliberate scope boundary,
not an omission waiting to be filled in silently.

## Two design decisions worth defending

**The incumbent is re-scored, not remembered.** The comparison uses the incumbent's
score on the *same* holdout as the candidate, computed now. Comparing against the
number recorded at the incumbent's training time would compare two different test sets,
which is the most common way an automated promotion gate quietly promotes a worse model.

**A rejected candidate is still registered.** It gets a version and a `decision=reject`
tag but no alias. Discarding it would throw away the evidence for why the loop did
nothing, and a rejection is exactly the event someone will want to inspect later.

## Acceptance criteria

Each exists because of a specific way an automated retraining loop goes wrong.

| Criterion | Value | Why |
|---|---|---|
| `min_absolute_improvement` | 0.002 ROC-AUC | Retraining always moves the score a little. Promoting on any improvement means promoting on noise, and the registry fills with churn. |
| `max_allowed_degradation` | 0.0 | An explicit ceiling on how much worse a candidate may be and still ship. Zero here; it is a separate knob because a team needing fresher data may accept a bounded loss. |
| `min_candidate_roc_auc` | 0.85 | An absolute floor. If the incumbent has already decayed, "better than the incumbent" is a very low bar, and without a floor the loop ratchets downwards. |
| `max_latency_ratio` | 1.5× | A model that is accurate and far slower is not deployable. |
| `max_p95_latency_ms` | 50 ms | Applied **only** when there is no incumbent, where a ratio is undefined. |
| `require_clean_validation` | true | Never train on data that failed validation. This is what stops a corrupt upstream feed being laundered into a promoted model. |
| `min_training_rows` | 5,000 | A drift window can be small; fitting on 200 rows and promoting is how a loop destroys a good model. |

### Why latency is a ratio, not a millisecond budget

An absolute wall-clock threshold is not portable between machines and is not stable on
one machine. During development a 50 ms absolute budget **rejected a candidate that was
2.4 ROC-AUC points better and no slower** — the host was simply oversubscribed and every
measurement was inflated. A ratio cancels that, because both models pay the same tax.

Two further refinements were needed to make the ratio trustworthy, both found by
investigating implausible numbers:

1. **Interleaved measurement.** Timing one model fully and then the other does not
   cancel contention, because the load changes between the windows. That produced a
   measured 10.5× ratio (216 ms vs 21 ms) between two models whose real cost differs by
   under 2×. Requests are now alternated candidate/incumbent on the same rows.
2. **The ratio is taken on the median, not p95.** On a shared host the tail is dominated
   by scheduler outliers belonging to neither model. Both p95 figures are still
   reported, because the tail is what a latency budget is about and a reviewer must be
   able to see it.

A related finding, kept because it is genuinely useful: a 200-tree model really *is*
~4.5× slower per single-row prediction than a 5-tree one. When the experiment's weak
incumbent was created with `max_iter: 5`, the gate correctly rejected the more accurate
candidate for being slower — a real accuracy/latency trade-off, not a bug. The
experiment now weakens the incumbent by learning rate and tree depth instead, holding
inference cost constant so the accuracy decision is isolated.

## Measured results

```bash
python scripts/retrain_experiment.py
```

Four cases against a scratch registry, so the real one is untouched:

| Case | Triggered | Decision | Failed criterion | Incumbent | Candidate | Δ ROC-AUC | Production after |
|---|---|---|---|---:|---:|---:|---|
| `no_drift` | **no** | — | — | — | — | — | v1 (unchanged) |
| `reject_no_gain` | yes | **reject** | `min_absolute_improvement` | 0.926784 | 0.926784 | 0.000000 | v1 (unchanged) |
| `promote_better` | yes | **promote** | — | 0.897678 | 0.926784 | **+0.029106** | **v2** |
| `rollback_after_promotion` | yes | **rollback** | — | — | — | — | **v1 (restored)** |

| Timing | Value |
|---|---|
| Retraining on 26,048 rows | 1.08 – 1.25 s |
| Promotion (registry alias repoint) | 0.0043 s |
| Rollback (registry alias repoint) | 0.0055 s |

All four cases assert their expected outcome, and the script exits non-zero if any
differs — so this table cannot drift away from reality.

Additionally, `tests/test_retraining.py` covers all six criteria individually plus
promote, reject, rollback, schema-triggered retraining, forced retraining and a
training failure that must promote nothing — 28 tests.

## Rollback verified through the serving layer

Moving a registry alias proves nothing about the running service. `scripts/rollback_through_serving.py`
starts the real uvicorn server, drives **four concurrent probe threads** issuing
predictions continuously, then promotes and rolls back while traffic is in flight.

| Step | From → To | End to end | Reload | Requests in flight | Failed |
|---|---|---:|---:|---:|---:|
| Promote | v1 → v2 | 0.063 s | 0.037 s | 14 | **0** |
| Rollback | v2 → v1 | 0.082 s | 0.056 s | 18 | **0** |

Across the whole experiment: **1,119 probe requests, 0 failed**, both model versions
observed in responses, and the production alias correctly back at v1 at the end.

The concurrent probes exist because a single serial prober issues a request only every
few hundred milliseconds, and a sub-second alias switch can complete with no request in
flight at all — "0 of 0 requests failed" is not evidence of anything. The script now
*requires* in-flight traffic during both switches or it fails.

### What this does and does not prove

**Proven:** rollback works end to end through the live HTTP service; the API served the
earlier version afterwards; no request failed during the measured switches.

**Not proven — and not claimed:** zero-downtime deployment. That is a claim about
sustained production traffic across many switches, node restarts and partial failures.
What was measured is 1,119 requests across two switches on one machine with no failures.
That is real evidence, and it is stated as exactly that.

## Rollback mechanics

Rollback is deliberately symmetric with promotion: it also updates `previous` to the
version being rolled *away from*, so a rollback can itself be rolled back. Without
that, a mistaken rollback is a dead end.

| Alias | Meaning |
|---|---|
| `production` | The version the API serves |
| `previous` | What `production` pointed at before the last change — the known rollback target |
| `candidate` | A freshly retrained model awaiting the decision |

Error paths covered by tests: rollback with no `production` alias set, rollback with no
`previous` recorded, rollback to a non-existent version, and rollback to the version
already in production — each raises a specific `RegistryError` rather than silently
doing nothing.

## Operational runbook

```bash
# Where is production pointing?
python -c "from mlserve.models.registry import ModelRegistry; \
           print(ModelRegistry().production().to_dict())"

# Roll back and make the running service pick it up
python -c "from mlserve.models.registry import ModelRegistry; \
           print(ModelRegistry().rollback())"
curl -X POST http://127.0.0.1:8077/admin/reload
curl -s http://127.0.0.1:8077/model-info | python -m json.tool | head -5
```

`/admin/reload` is **unauthenticated** in this local platform. A deployment would put it
behind authentication or move it off the public listener entirely.
