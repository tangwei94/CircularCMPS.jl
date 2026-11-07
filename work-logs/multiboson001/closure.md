# multiboson001 closure

Status at original closure: **closed positively — completed successfully**, at the owner's request.

**Subsequently reopened by the owner.** The preparation-phase diagnostic extension
is complete; see the current [work log](README.md). This document preserves the
original closure and roadmap entry, rather than declaring the reopened work closed.

The requested simulation comparison is complete and recorded in commit `ddd0ed2`.
All three left-canonical trajectories, step-aligned diagnostics, scientific TSVs,
comparison plots, execution log, and local HDF5 checkpoints are retained.
The prepared case converged in 481 iterations; both unprepared cases reached the
1000-iteration budget without meeting the gradient tolerance. No further simulation
work is pending within this scope. Manuscript edits remain deferred and outside
this work's completion criteria.

Validation and limitations are documented in [the work log](README.md), including
the nonmonotonic condition number in the unpreconditioned trajectory, single-seed
scope, solver warnings, and ambient RNG used by internal Krylov solves.

## Roadmap entry

## multiboson001 - Left-canonical initialization and environment conditioning
- motivation and setup: Collect numerical evidence for Referee 1, Comment 2 by comparing three left-canonical two-component cMPS optimizations at chi=8, c=10, c12=-7, mu=0 from the same seeded initial tensors: without preparation with/without preconditioning, and after 100 diagonal-gauge preparation steps with preconditioning. Execute locally in the existing cmps-repl, recording energy, gradient norm, and current-state rho_R condition number.
- conclusion: Completed successfully. The prepared trajectory converged in 481 iterations (gradient norm 7.82e-7); both unprepared trajectories exhausted 1000 iterations with large gradients and ill-conditioned environments. Disabling preconditioning did not resolve poor convergence for this initial state and budget. The unpreconditioned condition number initially decreases and later rises; these results do not establish monotonic growth, an underlying mechanism, or asymptotic nonconvergence. Tables, plots, logs, and checkpoints are retained; manuscript changes are deferred.
- details: [work log](README.md).

The roadmap entry is retained here for publication during integration. This work
was performed and committed directly on main; closure does not require a merge.
