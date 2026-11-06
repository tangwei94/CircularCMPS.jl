# multiboson001 — left-canonical random-initialization diagnostics

**Status: closed positively — completed successfully.** See the
[closure record and roadmap entry](closure.md). Simulation results are committed
in `ddd0ed2`; manuscript edits remain deferred outside this work's scope.

Work performed directly in the existing CircularCMPS.jl checkout, starting at
`5af47a4`. No manuscript changes or DATASTORE writes. No additional agents.
The earlier, clean worktree was removed after confirming all async agents stopped.

## Design

Single recorded seed (1), chi=8, two species, c=10, c12=-7, mu=0.
Three left-canonical trajectories: no preparation with hybrid preconditioner
(type 3), no preparation without preconditioner (type 0), and after 100 diagonal
L-BFGS preparation steps with hybrid preconditioner. Each left-canonical run has
maxiter=1000 and gradtol=1e-6; preparation has maxiter=100 and gradtol=1e-12.
No preparation-phase condition-number trajectory is collected.

All three derive from the same saved random diagonal state. The constructor is
passed a closure using MersenneTwister(1) and rand(ComplexF64,...), matching the
original benchmark distribution. Each optimizer receives a separate copy.

## Execution

Only the designated existing Julia REPL cmps-repl (herdr wT:pA) is used.
Task-local environment: `work-logs/multiboson001/environment/`, with local
CircularCMPS developed by absolute path and HDF5 added. Missing package dependencies
were installed through Pkg in that REPL; root Project.toml is unchanged.
Automatic Pkg precompilation was disabled for installation; normal package loading
performs Julia's dependency compilation. Environment Manifest.toml records versions
(locally retained, ignored by repository policy).

Driver: `driver.jl`, isolated module Multiboson001. Raw stdout/stderr retained in
`execution.log`. Scientific TSVs are under `results/<timestamp>-seed1/`.
Checkpoints are native HDF5 under root `data/multiboson001/<run-id>/checkpoints/`
(ignored, never committed). Q and Lambda reconstruct diagonal states; Q, M, Minv,
and D1/D2 diagonals reconstruct left-canonical states. No optimizer history is
checkpointed; these are state snapshots, not exact L-BFGS restart checkpoints.

## Measurement and conventions

At mu=0 raw Hamiltonian coefficients are twice the manuscript convention, so
energy_paper=energy_raw/2. gnorm is the optimizer's raw-coordinate gradient norm.
The callback precedes the package's cached rho_R update. Therefore the diagnostic
recomputes right_env for current x.data, using a copy of the cached initial guess,
without changing optimizer state. It trace-normalizes rho_R before diagnostics.
cond_rhoR is sigma_max/sigma_min (2-norm); eigen_min is computed from the Hermitian
part, alongside Hermiticity and left-canonical residuals. Near machine precision,
condition numbers cannot resolve arbitrarily small eigenvalues reliably.

OptimKit 0.3.1 records initialization in history but invokes finalize only after
accepted iterations. Initial diagnostics are recorded separately; initial energy
and gradient are taken from history[1,:]. Every later row is asserted to match
history[k+1,:] and the callback iteration counter. Live TSVs flush every callback
so failures retain evidence; failed trajectories get their own error file and do
not prevent independent trajectories from running.

## Results — run `20261106-202812-seed1`

Completed all three trajectories in the designated REPL. Initial state is saved;
all 2484 trajectory rows (including iteration zero per run) are finite and contiguous.
Driver assertions verified callback energy and gradient against OptimKit history.

| Trajectory | Iterations | Final energy (paper) | Final gradient norm | Final cond(rho_R) |
|---|---:|---:|---:|---:|
| No preparation, preconditioned | 1000 | -8.58388608388821 | 75.6205981269 | 7.94100914874e11 |
| No preparation, unpreconditioned | 1000 | -8.926721854246923 | 547.110869376 | 3.14096183871e8 |
| After preparation, preconditioned | 481 | -180.07883852694263 | 7.82013616318e-7 | 49298.1429557 |

The prepared trajectory meets gradtol; both unprepared trajectories terminate at
maxiter. This demonstrates failure to converge within the tested budget even with
preconditioning disabled, but does not prove asymptotic nonconvergence or identify
the cause. The prepared finite-chi stationary point supplies the energy reference;
we do not assert a proof of global optimality.

The unprepared initial environment is already ill-conditioned: cond approximately
1.77524e9. Preconditioned optimization reaches a maximum 1.23504e12. Unpreconditioned
optimization initially *reduces* the condition number, reaching 2.70850e6 at iteration
474, then rises to 3.14096e8. Thus this seed does NOT support a statement that the
condition number grows throughout both unprepared runs, nor that it ends above its
initial value in the unpreconditioned run. It supports persistent ill-conditioning
and poor convergence in both. Preparation starts the left-canonical trajectory at
energy -167.40473111954384 and cond 295609; its peak cond is 664751.

Maximum gauge residual across all rows is 1.19887e-10 (initial state); maximum
Hermiticity residual is 1.53559e-12. Hermitian-part eigenvalues remain positive in
all recorded rows. Minimum eigenvalue for the preconditioned unprepared trajectory
is 7.75942e-13. These diagnostics do not replace rigorous error estimates for the
near-singular environments. Numerous GMRES/backward-environment nonconvergence
warnings are retained in execution.log; no driver exception occurred.

`audit.py` validates the TSVs, writes `energy-errors.tsv`, and regenerates
`summary.tsv` with extrema INCLUDING iteration zero. The driver's original summary
maximum excludes iteration zero; use the audited summary for all reported extrema.
`plot.py` writes `trajectories.pdf` and `trajectories.png`: only the left-canonical
segments appear, each starting at local iteration zero. The TSV also retains total
iteration counts including preparation.

## Reproducibility and limitations

The initial tensors are reproducible from the explicitly seeded local generator and
saved exactly in HDF5. CircularCMPS internally randomizes some Krylov initial guesses
using the ambient RNG (including backward solves); this run did not seed that RNG.
Therefore the recorded driver/environment/checkpoints specify the calculation, but
bitwise reproduction of the entire optimization trajectory is not claimed. All
comparisons share identical initial tensors; small differences in independently
computed initial diagnostics are numerical, not different initial states.

The resolved environment is also archived as `environment/resolved-environment.toml`
(copy to Manifest.toml in that environment if recreating it). Package source remains
unchanged at 5af47a4. Exact artifact SHA-256 values are in `sha256sums.txt`.
Execution: Julia 1.11.9, one Julia thread, four BLAS threads. No separate simulation
process was launched; Julia itself launched normal dependency-precompile helpers
when packages were first imported. CircularCMPS caching failed on its pre-existing
duplicate similar method; Julia successfully loaded it without that cache.

Seven HDF5 checkpoints were reopened and their Q matrices checked for dimensions and
finite entries; `checkpoint-verification.txt` records dataset names. Source checkpoint
fields preserve the complete state tensors, but no exact optimizer restart is offered.
The original active project was restored at completion and cmps-repl was confirmed
alive and idle at `julia>`. Review was performed directly, with no subagents.

Reproduction commands (Julia runs only in designated REPL):

```julia
using Pkg
Pkg.activate("/home/tangwei/Workspace/research/CircularCMPS.jl/work-logs/multiboson001/environment")
include("/home/tangwei/Workspace/research/CircularCMPS.jl/work-logs/multiboson001/driver.jl")
Multiboson001.run(seed=1, maxiter=1000, prepsteps=100)
```

Each invocation creates a fresh timestamped run directory; retain the recorded run
as evidence rather than replacing it with a numerically different replay.
