# Kokkos Tuning Resume Follow-Up

## Full Search Checkpoint/Resume

Best-so-far replay avoids restarting from defaults, but it does not preserve the
full search state. Add per-strategy checkpoint/resume so restarted runs avoid
reevaluating parameter combinations where possible.

Prioritize exhaustive search first because it can resume deterministically from
iteration counters and parameter indexes. Then extend support to simulated
annealing, random search, genetic search, and Nelder-Mead with their
strategy-specific state.

## Restart Regression Tests

Add tests that simulate two APEX runs using the same Kokkos tuning cache:

- Run 1 stops before convergence and writes partial state.
- Run 2 resumes from the saved state.
- For exhaustive search, verify already evaluated combinations are not repeated.
- Verify `BestSoFar` contexts are not reported as converged.
- Verify cache-only replay still only applies fully converged contexts.

## Context Definition

Give APEX a context that accurately reflects the workload without exploding the
number of contexts.

The context key should include enough workload information to make cached tuning
decisions valid, but avoid including overly specific values that create too many
near-duplicate contexts and prevent convergence.

## Invalid Configurations

Account for invalid configurations and improve fault tolerance.

Some parameter combinations may be unsupported, fail at runtime, or produce
invalid execution configurations. Tuning should record these combinations as
invalid, avoid retrying them after restart, and fall back to a valid default or
best-known configuration when needed.
