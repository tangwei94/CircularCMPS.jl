using Pkg;
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
using JLD2 
using OptimKit 
using Revise 
using CircularCMPS 

# parameters for the model
c1, c2 = 1.0, 1.0
c12 = 0.5 
μ1, μ2 = 2.0, 2.0 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf) 
_f_MDMinv(_x1::MultiBosonCMPSData_MDMinv) = energy(Hm, _x1)

function optimization_plain(χ::Integer)
    ys, gs = Float64[], Float64[]
    myfinalize_for_diag!(x, f, g, numiter) = begin
        x1 = left_canonical(MultiBosonCMPSData_MDMinv(x))
        _, ∂E_MDMinv = withgradient(_f_MDMinv, x1)
        g_MDMinv = CircularCMPS.diff_to_grad(x1, ∂E_MDMinv[1])

        #println("f = $f, norm(g) = $(norm(g)), norm(g_MDMinv) = $(norm(g_MDMinv))")
        push!(ys, f)
        push!(gs, norm(g_MDMinv))
        
        if norm(g_MDMinv) < 1e-5 # exit condition for optimization 1
            g = 0.0 * g
        end
        return x, f, g, numiter
    end
    ψ0 = MultiBosonCMPSData_diag(rand, χ, 2);
    res = ground_state(Hm, ψ0; gradtol=0.0, maxiter=1000000, do_preconditioning=false, _finalize! =myfinalize_for_diag!);
    return ys, gs
end

χ = parse(Int, ARGS[1])
ix = parse(Int, ARGS[2])
res = optimization_plain(χ);
#@save "multiboson_scripts_for_paper/benchmark/data/benchmark_plain_opt_$(χ)_$(ix).jld2" res
@save "benchmark_plain_opt_$(χ)_$(ix).jld2" res
