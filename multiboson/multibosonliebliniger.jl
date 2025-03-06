using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
using CairoMakie 
using JLD2 
using OptimKit 
using Revise 
using CircularCMPS 

# parameters for the model
c1, μ1 = 1, 2 
c2, μ2 = 1, 2 
c12 = 0.5 
Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf) 

# parameters for penalty term calculation
lgΛmin, lgΛmax, steps = 1, 6, 11
ΔlgΛ = (lgΛmax - lgΛmin) / (steps - 1)
Λs = 10 .^ (lgΛmin:ΔlgΛ:lgΛmax)
#Λs = Vector{Float64}(2:10)

χ = 4

# optimization procedure
function optimization_starting_from(ψ; penalty_order=1)
    # initialization with penalty term
    ϕ = left_canonical(ψ)[2];
    res_lm = ground_state(Hm, ϕ; Λs = Λs, gradtol=1e-2, maxiter=500, do_prerun=true, order=penalty_order);
   
    # convert to MDMinv parametrization and continue
    Ψ = left_canonical(CircularCMPS.convert_to_MultiBosonCMPSData_MDMinv(res_lm[1])[1]);
    res = ground_state(Hm, Ψ; gradtol=1e-6, maxiter=1000, preconditioner_type=1);

    return res, res_lm
end
#function optimization_starting_from_MDMinv(ψ; penalty_order=1)
#    # initialization with penalty term
#    ϕ = left_canonical(ψ)[2];
#    Λs = [1e4, 1e51e6, 1e7]
#    res_lm = ground_state(Hm, ϕ; Λs = Λs, gradtol=1e-2, maxiter=1000, do_prerun=false, order=penalty_order);
#   
#    # convert to MDMinv parametrization and continue
#    Ψ = left_canonical(CircularCMPS.convert_to_MultiBosonCMPSData_MDMinv(res_lm[1])[1]);
#    res = ground_state(Hm, Ψ; gradtol=1e-6, maxiter=1000, preconditioner_type=1);
#
#    return res, res_lm
#end

ψ0 = CMPSData(rand, χ, 2);
res, res_lm = optimization_starting_from(ψ0);

ψ1 = expand(res_lm[2], 2*χ, 10; perturb = 1e-3);
#ψ1 = CMPSData((expand(res[1], 2*χ; perturb = 1e-3), res[2])[1]);
res1, res_lm1 = optimization_starting_from(ψ1);

ψ2 = expand(res_lm1[2], 3*χ, 10; perturb = 1e-3);
res2, res_lm2 = optimization_starting_from(ψ2);

ψ3 = expand(res_lm2[2], 4*χ, 10; perturb = 1e-3);
res3, res_lm3 = optimization_starting_from(ψ3);

ψ4 = expand(res_lm3[2], 5*χ, 10; perturb = 1e-3);
res4, res_lm4 = optimization_starting_from(ψ4);

ψ5 = expand(res_lm4[2], 6*χ, 10; perturb = 1e-3);
res5, res_lm5 = optimization_starting_from(ψ5);
