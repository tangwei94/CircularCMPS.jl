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

function optimization_plain(χ::Integer)
    ys, gs = Float64[], Float64[]
    myfinalize!(x, f, g, numiter) = begin
        push!(ys, f)
        push!(gs, norm(g))
        return x, f, g, numiter
    end
    ψ0 = MultiBosonCMPSData_diag(rand, χ, 2);
    res = ground_state(Hm, ψ0; gradtol=1e-3, maxiter=1000000, do_preconditioning=false, _finalize! =myfinalize!);
    return ys, gs
end

χ = parse(Int, ARGS[1])
ix = parse(Int, ARGS[2])
res = optimization_plain(χ);
@save "multiboson_scripts_for_paper/benchmark/data/benchmark_plain_opt_$(χ)_$(ix).jld2" res




