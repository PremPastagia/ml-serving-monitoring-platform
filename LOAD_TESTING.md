# Load Testing

## Method

`scripts/load_test.py` is deliberately self-contained:

- **It starts and stops its own server.** Benchmarking whatever happens to be running is
  how a number becomes unreproducible. The script launches uvicorn, waits for `/health`
  to report `model_loaded`, runs, then shuts down — and records the exact command line.
- **Payloads are drawn from the real held-out test split with a fixed seed**, so two
  runs send the same bytes. Sending one repeated record would let every layer cache and
  would measure the cache.
- **A warm-up phase is excluded.** The first requests pay lazy imports and first-call
  allocation inside scikit-learn; folding those into p99 makes the tail meaningless.
  Cold start is measured separately.
- **Host CPU is recorded with every result**, and the script refuses to present figures
  as clean when other work is using more than 35% of the machine.
- **It refuses to reuse a busy port.** A leftover server from an interrupted run would
  answer `/health` while the new uvicorn exited immediately — the benchmark would
  silently measure a stale build. This actually happened during development.

```bash
python scripts/load_test.py --thread-pin 1 --duration 20 --warmup 5 \
       --concurrency 1 2 4 8 16 --batch-size 1 32
```

## Environment

| | |
|---|---|
| Machine | Apple Silicon arm64, 10 CPUs, macOS 27.0 |
| Server | uvicorn, 1 worker, `OMP_NUM_THREADS=1` |
| Model | registry version 1, HistGradientBoosting, 200 trees |
| Measured | 20 s per point after 5 s warm-up |
| Host CPU used by other work | 12–15% |

## Results

| Concurrency | Batch | Requests/s | Records/s | p50 ms | p95 ms | p99 ms | Error rate | RSS MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | **256.4** | 256.4 | **3.68** | **4.73** | **9.18** | 0.000 | 282 |
| 2 | 1 | 247.6 | 247.6 | 7.80 | 9.16 | 13.69 | 0.000 | 280 |
| 4 | 1 | 247.6 | 247.6 | 15.85 | 23.03 | 25.92 | 0.000 | 275 |
| 8 | 1 | 242.4 | 242.4 | 32.61 | 46.79 | 53.57 | 0.000 | 273 |
| 16 | 1 | 236.5 | 236.5 | 66.63 | 92.97 | 111.49 | 0.000 | 272 |
| 1 | 32 | 205.2 | **6,564.7** | 4.67 | 5.58 | 10.16 | 0.000 | 272 |
| 2 | 32 | 196.2 | 6,276.8 | 9.98 | 11.97 | 16.49 | 0.000 | 271 |
| 4 | 32 | 192.7 | 6,166.4 | 20.36 | 27.13 | 31.32 | 0.000 | 271 |
| 8 | 32 | 192.7 | 6,166.1 | 40.92 | 60.89 | 71.65 | 0.000 | 271 |
| 16 | 32 | 193.8 | 6,201.6 | 82.15 | 136.05 | 149.86 | 0.000 | 271 |

**Zero failed requests in every configuration.**

Cold start (process launch → `/health` reporting `model_loaded`): **1.742 – 1.797 s**.

## Reading the numbers

Request throughput is **flat from concurrency 1 to 16** (256 → 236 req/s) while latency
rises roughly linearly. That is the signature of a saturated single worker with a
stable service time: the server is already at its throughput ceiling at concurrency 1,
so additional clients queue rather than add capacity. It is the expected and correct
shape, and it means **p50 at concurrency 1 (3.68 ms) is the number that describes the
service**; the higher-concurrency latencies describe queueing.

Batching is where the real capacity is: 32 records per request gives **6,565 records/s**
versus 256 records/s one at a time — a **25× improvement in record throughput** — because
the ~3.4 ms of pandas and ColumnTransformer overhead is paid once per request rather
than once per record.

Memory is flat at ~271–282 MB across every configuration, so there is no leak under
sustained load.

## The finding that justified the whole harness

The first benchmark runs showed throughput *falling* as concurrency rose and the entire
10-core machine at 99–100% CPU for one server process. Investigating rather than
publishing that produced the single largest performance result in the project:
scikit-learn's HistGradientBoosting predicts through OpenMP and fans even a single-row
request across every core.

| Configuration | Requests/s | p50 ms | p99 ms | Host CPU |
|---|---:|---:|---:|---:|
| Uncapped, concurrency 1 | 99.2 | 8.84 | 22.68 | 99.3% |
| **Pinned, concurrency 1** | **256.4** | **3.68** | **9.18** | **19.6%** |
| Uncapped, concurrency 8 | 47.4 | 164.35 | 196.16 | 99.8% |
| **Pinned, concurrency 8** | **242.4** | **32.61** | **53.57** | **25.3%** |

**2.6× throughput at concurrency 1, 5.1× at concurrency 8, p99 3.7× lower, host CPU
from 99.8% to 25.3%.** Full table in [BENCHMARKS.md](BENCHMARKS.md#thread-pinning).

Reproduce the comparison with `--thread-pin 0`.

## A measurement-methodology note

An earlier version of this harness gated on **load average** and flagged every run as
unusable. On macOS the load average counts threads blocked in uninterruptible I/O as
well as runnable ones; it read 30–40 on a machine that measurement showed was ~20%
busy. The guard now uses measured CPU utilisation, and the load average is recorded but
annotated as not being the gate. Calibration: one busy spin thread moved
`psutil.cpu_percent` from 16.1% to 22.2% on a 10-core machine, as expected.

## Limitations

- Single machine, single worker, loopback networking. No real network latency, no load
  balancer, no TLS.
- macOS on Apple Silicon. Linux x86 numbers will differ.
- 20 s per point. Long-run stability, memory growth over hours and GC behaviour are not
  measured.
- The client shares the host with the server, so the load generator competes for CPU.
- Multi-worker configurations were measured and did **not** help
  (74 req/s with 4 workers versus 112 with 1, and ~1 GB RSS versus 282 MB) — but that
  was measured *before* thread pinning, when each of the 4 workers was itself trying to
  use 10 cores. A pinned multi-worker measurement has not been taken, so no claim is
  made about how this scales across processes.

## Artefacts

| File | Contents |
|---|---|
| `SERVING_BENCHMARKS.csv` | Every configuration, pinned and unpinned |
| `results/serving/load_test_pin_w1.json` | Full run metadata including the exact server command |
| `results/serving/load_test_nopin_w1.json` | The comparison run |
