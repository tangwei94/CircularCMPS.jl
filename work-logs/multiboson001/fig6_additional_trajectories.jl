using CircularCMPS, Dates, HDF5, LinearAlgebra, Logging, Random

const CHECKPOINTS = normpath(joinpath(@__DIR__, "../../data/multiboson001",
    "20261106-202812-seed1/checkpoints"))
const SOLVER_SEED = 20261107
const E_REF = -180.07883852694263
const COUPLINGS = [10.0 -7.0; -7.0 10.0]
const CHEMICAL_POTENTIALS = [0.0, 0.0]

function load_diagonal(path)
    h5open(path, "r") do f
        return MultiBosonCMPSData_diag(read(f["Q"]), read(f["Lambda"]))
    end
end

function load_left_canonical(path)
    h5open(path, "r") do f
        return MultiBosonCMPSData_MDMinv(read(f["Q"]), read(f["M"]), read(f["Minv"]),
            [Diagonal(read(f["D$j"])) for j in eachindex(CHEMICAL_POTENTIALS)])
    end
end

function run_trajectory(hamiltonian, out, run)
    state = run.load(run.checkpoint)
    @assert size(state.Q) == (8, 8)
    accepted = Tuple{Int,Float64,Float64}[]
    function record(x, f, g, k)
        push!(accepted, (k, f, sqrt(real(dot(g, g)))))
        return x, f, g, k
    end

    started = now()
    println("START $(run.label) $started")
    flush(stdout)
    # The preparation rerun seeds the optimizer from its own Julia task.
    result, elapsed_ns = fetch(schedule(Task() do
        Random.seed!(SOLVER_SEED)
        t0 = time_ns()
        result = ground_state(hamiltonian, state; maxiter=run.maxiter, gradtol=run.gradtol,
            run.options..., _finalize! = record)
        return result, time_ns() - t0
    end))
    finished = now()

    _, final_raw, _, evaluations, history = result
    iterations = size(history, 1) - 1
    @assert iterations == length(accepted)
    @assert all(k == i && isapprox(f, history[i+1, 1]; rtol=1e-12) &&
        isapprox(gnorm, history[i+1, 2]; rtol=1e-10)
        for (i, (k, f, gnorm)) in enumerate(accepted))
    @assert all(isfinite, history) && final_raw == history[end, 1]

    open(joinpath(out, run.label * ".tsv"), "w") do io
        println(io, "iter\ttotal_iter\tenergy_raw\tenergy_paper\tgnorm")
        for (i, (raw, gnorm)) in enumerate(eachrow(history))
            iter = i - 1
            println(io, join((iter, iter + run.offset, raw, raw / 2, gnorm), '\t'))
        end
    end

    final_energy = final_raw / 2
    final_gnorm = history[end, 2]
    termination = if final_gnorm <= run.gradtol
        "gradient_tolerance"
    elseif iterations == run.maxiter
        "iteration_limit"
    else
        "stopped_before_limit"
    end
    println("FINISH $(run.label) $finished $termination $(elapsed_ns) ns")
    flush(stdout)
    return (run.label, started, finished, elapsed_ns, termination, iterations, evaluations,
        final_energy, abs(final_energy - E_REF), final_gnorm)
end

function main()
    @assert VERSION == v"1.11.9" && Threads.nthreads() == 1
    BLAS.set_num_threads(4)
    runid = Dates.format(now(), "yyyymmdd-HHMMSS") * "-fig6-additional"
    out = mkdir(joinpath(@__DIR__, "results", runid))
    runs = (
        (label="diag_500", load=load_diagonal, checkpoint=joinpath(CHECKPOINTS, "initial-diagonal.h5"),
            offset=0, maxiter=500, gradtol=1e-12, options=(do_preconditioning=false,)),
        (label="prep100_noprecond", load=load_left_canonical, checkpoint=joinpath(CHECKPOINTS, "prepared-left.h5"),
            offset=100, maxiter=400, gradtol=1e-6, options=(preconditioner_type=0,)),
    )
    open(joinpath(out, "configuration.txt"), "w") do io
        println(io, "Julia=$VERSION\nthreads=$(Threads.nthreads())\nBLAS_threads=$(BLAS.get_num_threads())")
        println(io, "project=$(Base.active_project())\nsolver_seed=$SOLVER_SEED\nchi=8")
        println(io, "couplings=$COUPLINGS\nchemical_potentials=$CHEMICAL_POTENTIALS\nL=Inf")
        println(io, "energy_paper=energy_raw/2\nreference_energy=$E_REF")
        println(io, "wall_ns=elapsed wall time of ground_state including any Julia compilation during the call")
        for run in runs
            println(io, "\n[$(run.label)]\ncheckpoint=$(run.checkpoint)\noffset=$(run.offset)")
            println(io, "maxiter=$(run.maxiter)\ngradtol=$(run.gradtol)")
            for (key, value) in pairs(run.options)
                println(io, "$key=$value")
            end
        end
    end

    println("RESULTS $out")
    flush(stdout)
    open(joinpath(out, "execution.log"), "w") do log
        redirect_stdio(stdout=log, stderr=log) do
            with_logger(ConsoleLogger(log)) do
                hamiltonian = MultiBosonLiebLiniger(COUPLINGS, CHEMICAL_POTENTIALS, Inf)
                open(joinpath(out, "summary.tsv"), "w") do io
                    println(io, "trajectory\tstarted_local\tfinished_local\twall_ns\ttermination\titerations\tenergy_gradient_evaluations\tfinal_energy_paper\tfinal_abs_error\tfinal_gnorm")
                    for run in runs
                        println(io, join(run_trajectory(hamiltonian, out, run), '\t'))
                        flush(io)
                    end
                end
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
