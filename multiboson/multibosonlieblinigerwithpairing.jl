using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote  
using CairoMakie 
using JLD2 
using OptimKit 
using Revise 
using CircularCMPS 

c1, c2 = 1.0, 1.0
μ1, μ2 = 2.0, 2.0 
u1, u2 = 0.5, 0.5 
#c12 = 0.5 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf) 

χ = 4
ψ0 = MultiBosonCMPSData_MDMinv(rand, χ, 2);
ψ0 = left_canonical(ψ0);
ψ0.Ds[1]

lgΛmin, lgΛmax, steps = 2, 6, 5
ΔlgΛ = (lgΛmax - lgΛmin) / (steps - 1)
Λs = 10 .^ (lgΛmin:ΔlgΛ:lgΛmax)

# initialization with lagrange multipiler
ψ = left_canonical(CMPSData(ψ0))[2];
res_lm = ground_state(Hm, ψ; Λs = Λs, gradtol=1e-2, maxiter=1000, do_prerun=true, order=1);
@save "multiboson/new_results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-$(lgΛmin)_$(lgΛmax)_$(steps).jld2" res_lm

ϕ = left_canonical(MultiBosonCMPSData_MDMinv(res_lm[1]));
res1 = ground_state(Hm, left_canonical(ϕ); gradtol=1e-6, maxiter=1000, preconditioner_type=1);

