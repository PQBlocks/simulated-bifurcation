# Use cases: pqblocks-bifurcation

Each runnable script below lives in `examples/` and prints its own check.

## 1. Max-Cut on a weighted graph

Split a graph's nodes into two groups so that the total weight of edges between the groups is as large as possible. This comes up in network partitioning, circuit layout and clustering.

```bash
python examples/maxcut.py
```

The script sets `J_ij = w_ij`, takes the best of 10 seeded reads, and compares the result with brute force on 8 nodes:

```
best cut found by SB over 10 reads: 44.0
optimal cut (brute force):         44.0
optimum reached: True
```

## 2. QUBO problems (scheduling, selection, assignment toy models)

Any problem written as "minimise xᵀQx over binary x" can be solved by converting it to Ising form: `J_ij = Q_ij/2` and `h_i = Q_ii/2 + Σ_{j≠i} Q_ij/2`.

```bash
python examples/qubo.py
```

```
best x found by SB: (1, 0, 1, 0) objective -7.0
brute-force optimum: (1, 0, 1, 0) objective -7.0
optimum reached: True
```

## 3. Ising spin-glass ground states

Pass the couplings `J` and fields `h` directly, as in the quickstart (`examples/quickstart.py`).

## When to use something else

- **You need a proof of optimality.** Use an exact solver (MIP / CP-SAT). SB is a heuristic, and on one dense 8-spin instance it misses the optimum by about 1 %.
- **Your problem has more than about 1000 variables.** Behaviour at that size is untested here.
- **Anything security-related.** This package is optimisation only and offers no cryptographic protection.
