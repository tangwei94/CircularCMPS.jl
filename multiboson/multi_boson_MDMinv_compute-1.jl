using LinearAlgebra, TensorKit, KrylovKit
using ChainRules, Zygote 
using CairoMakie
using JLD2 
using OptimKit
using Revise
using CircularCMPS

c1, μ1 = 1., 2.
c2, μ2 = 1.5, 2.5
c12 = 0.5
#c2, μ2 = parse(Float64, ARGS[1]), parse(Float64, ARGS[2])
#c12 = parse(Float64, ARGS[3]) 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf);

χ, Δχ = 16, 8

if Δχ > 0 
    lgΛmin, lgΛmax, steps = 2, 5, 10
    #@load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ-Δχ)-$(lgΛmin)_$(lgΛmax)_$(steps).jld2" res_wp
    @load "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ-Δχ).jld2" res_wp
    ϕ = expand(res_wp[1], χ, perturb = 1e-3);
else
    ϕ = nothing
end

lgΛmin, lgΛmax, steps = 2, 10, 81
ΔlgΛ = (lgΛmax - lgΛmin) / (steps - 1)
Λs = 10 .^ (lgΛmin:ΔlgΛ:lgΛmax)

# initialization with lagrange multipiler
ψ = left_canonical(CMPSData(ϕ))[2];
res_lm = ground_state(Hm, ψ; Λs = Λs, gradtol=1e-2, maxiter=1000, do_prerun=true);
@save "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-$(lgΛmin)_$(lgΛmax)_$(steps).jld2" res_lm

#CircularCMPS.convert_to_MultiBosonCMPSData_MDMinv(res_lm[1])

@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-$(lgΛmin)_$(lgΛmax)_$(steps).jld2" res_lm
ϕ = left_canonical(MultiBosonCMPSData_MDMinv(res_lm[1]));

# optimization
res_wp = ground_state(Hm, ϕ; do_preconditioning=true, maxiter=2000);
@save "multiboson/results/preconditioned_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-$(lgΛmin)_$(lgΛmax)_$(steps).jld2" res_wp
