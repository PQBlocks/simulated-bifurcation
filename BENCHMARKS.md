# Benchmarks: pqblocks-bifurcation

Results for the bSB and dSB solvers against four classical baselines on four small Ising instances. Losing cases are included.
Raw data: [benchmarks/benchmark-card.json](benchmarks/benchmark-card.json).

## Setup

| Item | Value |
|---|---|
| Date of run | 2026-07-11 (UTC) |
| CPU | Intel Core i5-10300H @ 2.50 GHz, 8 threads, no GPU |
| Runs per instance | 100 (master seed 42), 1 warm-up run excluded |
| Confidence intervals | bootstrap, 10,000 resamples, 95 % |
| Metric | TTS(0.99) = t_run · ln(0.01) / ln(1 − p), where p is the observed success rate (reaching the known optimum) |

Solver settings:
- **SB-bSB / SB-dSB:** `num_sweeps=1000`, `num_reads=1`, `dt=0.5`, `c0` computed per instance (see the note below).
- **Neal-SA:** `num_reads=100`, `num_sweeps=1000`, `beta_range=[0.1, 5.0]`.
- **OpenJij-SQA:** `num_reads=100`, `trotter=4`, `beta=5.0`, `gamma=1.0`.
- **MQLib:** `BURER2002`, 0.1 s timeout.
- **CP-SAT:** 0.1 s limit.

## Results

### brute_force_n5 (N=5, 10 edges)

| Solver | Success | TTS(0.99) | 95 % CI | Best energy | Optimum |
|---|---|---|---|---|---|
| SB-bSB | 100 % | 0.0005 s | [0.0005, 0.0005] | -3.729609 | -3.729609 |
| SB-dSB | 100 % | 0.0006 s | [0.0005, 0.0006] | -3.729609 | -3.729609 |
| Neal-SA | 100 % | 0.0076 s | [0.0076, 0.0077] | -3.729609 | -3.729609 |
| OpenJij-SQA | 100 % | 0.2980 s | [0.2846, 0.3155] | -3.729609 | -3.729609 |
| MQLib | 100 % | 0.1018 s | [0.1018, 0.1019] | -3.729609 | -3.729609 |
| CP-SAT | 100 % | 0.0042 s | [0.0035, 0.0054] | -3.729609 | -3.729609 |

### brute_force_n6 (N=6, 15 edges)

| Solver | Success | TTS(0.99) | 95 % CI | Best energy | Optimum |
|---|---|---|---|---|---|
| SB-bSB | 100 % | 0.0006 s | [0.0006, 0.0006] | -3.612752 | -3.612752 |
| SB-dSB | 100 % | 0.0006 s | [0.0006, 0.0006] | -3.612752 | -3.612752 |
| Neal-SA | 100 % | 0.0107 s | [0.0107, 0.0107] | -3.612752 | -3.612752 |
| OpenJij-SQA | 100 % | 0.3397 s | [0.3234, 0.3620] | -3.612752 | -3.612752 |
| MQLib | 0 % | n/a | n/a | -2.205190 | -3.612752 |
| CP-SAT | 100 % | 0.0041 s | [0.0041, 0.0042] | -3.612752 | -3.612752 |

### planted_n7 (N=7, 21 edges)

| Solver | Success | TTS(0.99) | 95 % CI | Best energy | Optimum |
|---|---|---|---|---|---|
| SB-bSB | 100 % | 0.0006 s | [0.0006, 0.0006] | -4.078013 | -4.078013 |
| SB-dSB | 100 % | 0.0006 s | [0.0006, 0.0006] | -4.078013 | -4.078013 |
| Neal-SA | 100 % | 0.0128 s | [0.0128, 0.0129] | -4.078013 | -4.078013 |
| OpenJij-SQA | 100 % | 0.3340 s | [0.3246, 0.3492] | -4.078013 | -4.078013 |
| MQLib | 100 % | 0.1020 s | [0.1019, 0.1020] | -4.078013 | -4.078013 |
| CP-SAT | 100 % | 0.0046 s | [0.0046, 0.0047] | -4.078013 | -4.078013 |

### brute_force_n8 (N=8, 28 edges): losing case

| Solver | Success | TTS(0.99) | 95 % CI | Best energy | Optimum |
|---|---|---|---|---|---|
| SB-bSB | **0 %** | n/a | n/a | -5.761389 | -5.822944 |
| SB-dSB | **0 %** | n/a | n/a | -5.761389 | -5.822944 |
| Neal-SA | 100 % | 0.0141 s | [0.0140, 0.0141] | -5.822944 | -5.822944 |
| OpenJij-SQA | 100 % | 0.3546 s | [0.3385, 0.3763] | -5.822944 | -5.822944 |
| MQLib | 0 % | n/a | n/a | -3.409337 | -5.822944 |
| CP-SAT | 100 % | 0.0055 s | [0.0054, 0.0056] | -5.822944 | -5.822944 |

On this dense 8-spin instance both SB variants settle in a local minimum 1.06 % above the optimum. The exact solver (CP-SAT) and two other heuristics find the optimum.

## Read these numbers carefully

- **The timings are not like-for-like.** SB makes 1 read per call, while Neal-SA and OpenJij-SQA make 100. Where p = 100 %, TTS(0.99) equals one call's wall time. So SB's lower TTS mostly reflects doing less work per call; it is not evidence of a faster algorithm at equal work.
- **The instances are tiny (N ≤ 8)**, chosen so the optimum is known exactly. They show correctness and per-call latency, not scaling. Larger instances have not been benchmarked yet.
- **About `c0`:** the original card says `c0` is "computed from the J matrix spectral norm". The implementation (`kernel._compute_c0`) uses `c0 = 0.5 / (σ_J · √N)`, where σ_J is the standard deviation of the off-diagonal couplings, and falls back to `0.5 / √N` when σ_J ≈ 0. The implementation is authoritative.

## Reproducing

The benchmark driver uses the shared harness `pqblocks-qubo-bench` and the baseline solvers (dwave-neal, openjij, ortools for CP-SAT, and the MQLib binary). A one-command reproduction script will be added here when `pqblocks-qubo-bench` is published. Until then the raw results above, and in `benchmarks/benchmark-card.json`, are the record.
