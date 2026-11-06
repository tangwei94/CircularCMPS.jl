module Multiboson001
using CircularCMPS, HDF5, LinearAlgebra, Random, Dates
const ROOT = normpath(joinpath(@__DIR__, "../.."))
function checkpoint(path, psi)
    h5open(path, "w") do f
        f["Q"] = psi.Q
        if psi isa MultiBosonCMPSData_diag
            f["Lambda"] = psi.Λs
        else
            f["M"] = psi.M
            f["Minv"] = psi.Minv
            for j in eachindex(psi.Ds)
                f["D$j"] = diag(psi.Ds[j])
            end
        end
    end
end
function diagnostic(psi; init=missing)
    rho = convert(Array, right_env(psi; init=init))
    rho /= tr(rho)
    s = svdvals(rho)
    ev = eigvals(Hermitian((rho+rho')/2))
    rs = [psi.M * d * psi.Minv for d in psi.Ds]
    gauge = norm(psi.Q + psi.Q' + sum(r' * r for r in rs))
    return (cond_rhoR=s[1]/s[end], sigma_min=s[end], sigma_max=s[1],
            eigen_min=minimum(ev), hermiticity=norm(rho-rho'), gauge_residual=gauge)
end
function run(; seed=1, maxiter=1000, prepsteps=100)
    runid = Dates.format(now(), "yyyymmdd-HHMMSS") * "-seed$seed"
    out = joinpath(@__DIR__, "results", runid)
    chk = joinpath(ROOT, "data", "multiboson001", runid, "checkpoints")
    mkpath(out); mkpath(chk)
    open(joinpath(out,"configuration.txt"),"w") do io
        println(io, "seed=$seed\nmaxiter=$maxiter\nprepsteps=$prepsteps\nchi=8\nc=10\nc12=-7\nmu=0\ngradtol=1e-6\nJulia=$VERSION\nthreads=$(Threads.nthreads())\nBLAS_threads=$(BLAS.get_num_threads())\nproject=$(Base.active_project())")
    end
    rng=MersenneTwister(seed)
    phi=MultiBosonCMPSData_diag((args...)->rand(rng,args...),8,2)
    checkpoint(joinpath(chk,"initial-diagonal.h5"),phi)
    H=MultiBosonLiebLiniger([10.0 -7.0; -7.0 10.0],[0.0,0.0],Inf)
    psi0=left_canonical(MultiBosonCMPSData_MDMinv(deepcopy(phi)))
    checkpoint(joinpath(chk,"initial-left.h5"),psi0)
    summaries=[]
    for (label,ptype,offset) in (("noprep_precond",3,0),("noprep_noprecond",0,0),("prep100_precond",3,prepsteps))
        println("START ",label," ",now()); flush(stdout)
        try
            if offset>0
                prep=ground_state(H,deepcopy(phi); gradtol=1e-12,maxiter=prepsteps,do_preconditioning=false)
                checkpoint(joinpath(chk,"prepared-diagonal.h5"),prep[1])
                psi=left_canonical(MultiBosonCMPSData_MDMinv(prep[1]))
                checkpoint(joinpath(chk,"prepared-left.h5"),psi)
            else
                psi=deepcopy(psi0)
            end
            rows=Any[]
            d0=diagnostic(psi)
            partial=joinpath(out,label*"-live.tsv")
            open(partial,"w") do io
                println(io,"iter\tenergy_raw\tgnorm\tcond_rhoR\tsigma_min\tsigma_max\teigen_min\thermiticity\tgauge_residual")
                callback=function(x,f,g,k)
                    d=diagnostic(x.data;init=deepcopy(x.ρR))
                    gn=sqrt(real(dot(g,g)))
                    push!(rows,(k=k,f=f,gn=gn,d=d))
                    println(io,join((k,f,gn,values(d)...),'\t')); flush(io)
                    return x,f,g,k
                end
                res=ground_state(H,psi; preconditioner_type=ptype,gradtol=1e-6,maxiter=maxiter,_finalize! = callback)
                history=res[5]
                @assert size(history,1)==length(rows)+1
                @assert all(r.k==i && isapprox(r.f,history[i+1,1];rtol=1e-12) && isapprox(r.gn,history[i+1,2];rtol=1e-10) for (i,r) in enumerate(rows))
                checkpoint(joinpath(chk,label*"-final.h5"),res[1])
                open(joinpath(out,label*".tsv"),"w") do tab
                    println(tab,"iter\ttotal_iter\tenergy_raw\tenergy_paper\tgnorm\tcond_rhoR\tsigma_min\tsigma_max\teigen_min\thermiticity\tgauge_residual")
                    println(tab,join((0,offset,history[1,1],history[1,1]/2,history[1,2],values(d0)...),'\t'))
                    for r in rows
                        println(tab,join((r.k,r.k+offset,r.f,r.f/2,r.gn,values(r.d)...),'\t'))
                    end
                end
                push!(summaries,(label,length(rows),res[2]/2,history[end,2],d0.cond_rhoR,rows[end].d.cond_rhoR,maximum(r.d.cond_rhoR for r in rows)))
            end
        catch e
            open(joinpath(out,label*"-failure.txt"),"w") do io
                showerror(io,e,catch_backtrace())
            end
            println("FAILED ",label,": ",sprint(showerror,e)); flush(stdout)
        end
    end
    open(joinpath(out,"summary.tsv"),"w") do io
        println(io,"trajectory\titerations\tfinal_energy_paper\tfinal_gnorm\tinitial_cond\tfinal_cond\tmax_cond")
        for r in summaries; println(io,join(r,'\t')); end
    end
    println("MULTIBOSON001_DONE ",out); flush(stdout)
    return out
end
end
