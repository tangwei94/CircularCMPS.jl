module Preparation001
using CircularCMPS, HDF5, LinearAlgebra, Random, Dates
include("driver.jl")
const ROOT = normpath(joinpath(@__DIR__, "../.."))
const SOURCE = joinpath(ROOT,"data","multiboson001","20261106-202812-seed1","checkpoints","initial-diagonal.h5")
function execute(; solver_seed=20261107)
    runid=Dates.format(now(),"yyyymmdd-HHMMSS")*"-preparation"
    out=joinpath(@__DIR__,"results",runid)
    chk=joinpath(ROOT,"data","multiboson001",runid,"checkpoints")
    mkpath(out); mkpath(chk)
    phi=h5open(SOURCE,"r") do f
        MultiBosonCMPSData_diag(read(f["Q"]),read(f["Lambda"]))
    end
    H=MultiBosonLiebLiniger([10.0 -7.0;-7.0 10.0],[0.0,0.0],Inf)
    states=[deepcopy(phi)]
    rows=NamedTuple[]
    cb=function(x,f,g,k)
        push!(states,deepcopy(x.data))
        push!(rows,(iter=k,energy=f,gnorm=sqrt(real(dot(g,g)))))
        return x,f,g,k
    end
    open(joinpath(out,"configuration.txt"),"w") do io
        println(io,"source_checkpoint=$SOURCE\nsolver_seed=$solver_seed\nmaxiter=100\ngradtol=1e-12\ndo_preconditioning=false\nJulia=$VERSION\nthreads=$(Threads.nthreads())\nBLAS_threads=$(BLAS.get_num_threads())\nproject=$(Base.active_project())\ndiagnostics=postprocess exact saved accepted iterates; no extra solves inside optimizer")
    end
    Random.seed!(solver_seed)
    res=ground_state(H,deepcopy(phi);maxiter=100,gradtol=1e-12,do_preconditioning=false,_finalize! = cb)
    hist=res[5]
    @assert length(states)==size(hist,1)==length(rows)+1
    @assert all(r.iter==i && isapprox(r.energy,hist[i+1,1];rtol=1e-12) && isapprox(r.gnorm,hist[i+1,2];rtol=1e-10) for (i,r) in enumerate(rows))
    h5open(joinpath(chk,"diagonal-iterates.h5"),"w") do f
        for (i,p) in enumerate(states)
            g=create_group(f,"iter_$(i-1)")
            g["Q"]=p.Q; g["Lambda"]=p.Λs
        end
    end
    open(joinpath(out,"preparation.tsv"),"w") do io
        println(io,"iter\tenergy_raw\tenergy_paper\tgnorm_diagonal\tcond_rhoR\tsigma_min\tsigma_max\teigen_min\thermiticity\tgauge_residual\tenergy_canonical_paper\tenergy_conversion_abs_error_paper")
        for (i,p) in enumerate(states)
            qcopy=copy(p.Q); lcopy=copy(p.Λs)
            psi=left_canonical(MultiBosonCMPSData_MDMinv(deepcopy(p)))
            d=Multiboson001.diagnostic(psi)
            ec=energy(H,psi)/2
            @assert isequal(p.Q,qcopy) && isequal(p.Λs,lcopy)
            println(io,join((i-1,hist[i,1],hist[i,1]/2,hist[i,2],values(d)...,ec,abs(ec-hist[i,1]/2)),'\t')); flush(io)
            if i==1 || i==length(states)
                Multiboson001.checkpoint(joinpath(chk,"left-iter_$(i-1).h5"),psi)
            end
        end
    end
    println("PREPARATION_DONE ",out); flush(stdout)
    return out
end
# Seeding occurs in a dedicated task, preserving the interactive task's RNG state.
run(;kwargs...)=fetch(schedule(Task(()->execute(;kwargs...))))
end
