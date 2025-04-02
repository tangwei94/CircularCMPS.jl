using Pkg;
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
using CairoMakie 
using JLD2 
using OptimKit 
using Revise 
using CircularCMPS 

# parameters for the model
c1, c2 = 1.0, 1.0 
c12 = 0.5 
μ1, μ2 = 2.0, 2.0 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf) 

function optimization_precondition(χ::Integer)
    ys, gs = Float64[], Float64[]
    myfinalize!(x, f, g, numiter) = begin

        if x isa MultiBosonCMPSData_diag
            _f_MDMinv(x1::MultiBosonCMPSData_MDMinv) = energy(Hm, x1)

            x1 = left_canonical(MultiBosonCMPSData_MDMinv(x))
            _, ∂E_MDMinv = withgradient(_f_MDMinv, x1)
            g_MDMinv = CircularCMPS.diff_to_grad(x1, ∂E_MDMinv[1])
            push!(ys, f)
            push!(gs, norm(g_MDMinv))
        end

        if x isa CircularCMPS.OptimState
            push!(ys, f)
            push!(gs, norm(g))
        end

        return x, f, g, numiter
    end

    ψ0 = MultiBosonCMPSData_diag(rand, χ, 2);
    res0 = ground_state(Hm, ψ0; gradtol=1e-3, maxiter=10000, do_preconditioning=false, _finalize! =myfinalize!);
    ψ1 = MultiBosonCMPSData_MDMinv(res0[1])
    res_opt = ground_state(Hm, ψ1; gradtol=1e-6, maxiter=1000000, preconditioner_type=1, _finalize! =myfinalize!);
    return ys, gs
end

χ = parse(Int, ARGS[1])
ix = parse(Int, ARGS[2])
res = optimization_precondition(χ);
#@save "multiboson_scripts_for_paper/benchmark/data/benchmark_precondition_opt_$(χ)_$(ix).jld2" res
#@save "benchmark_precondition_opt_$(χ)_$(ix).jld2" res
