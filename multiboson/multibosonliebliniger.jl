using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
using CairoMakie 
using JLD2 
using OptimKit 
using Revise 
using CircularCMPS 

# parameters for the model
#c1, μ1 = 1, 2 
#c2, μ2 = 1, 2 
#c12 = 0.5 
#Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf) 
c1, c2 = 1.0, 1.0
c12 = 0.5 
μ1, μ2 = 2.0, 2.0 
u1, u2 = 0.5, 0.5 
#c12 = 0.5 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf) 

# optimization procedure
function optimization_starting_from(ψ; penalty_order=1)
    # initialization with penalty term
    ϕ = left_canonical(ψ)[2];
    res_lm = ground_state(Hm, ϕ; Λs = 10 .^ (1:(1/4):6), gradtol=1e-3, maxiter=250, do_prerun=true, order=penalty_order);
    #res = ground_state(Hm, res_lm[1]; Λs = 10 .^ (6:(1/4):8), gradtol=1e-4, maxiter=1000, do_prerun=true, order=penalty_order);
   
    # convert to MDMinv parametrization and continue
    Ψ = left_canonical(CircularCMPS.convert_to_MultiBosonCMPSData_MDMinv(res_lm[1])[1]);
    res = ground_state(Hm, Ψ; gradtol=1e-4, maxiter=1000, preconditioner_type=1);

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

χ = 4

ψ0 = CMPSData(rand, χ, 2);
res, res_lm = optimization_starting_from(ψ0);

ψ1 = res[1]
ϕ2 = CircularCMPS.jordan_expand(ψ1)

ϕ1 = CMPSData(ψ1)

real.(eigen(K_mat(ϕ1, ϕ1))[1].data) |> sort
real.(eigen(K_mat(ϕ2, ϕ2))[1].data) |> sort

function ln_ovlp1(a::CMPSData, b::CMPSData)
    K = K_mat(a, b)
    a = eigen(K)[1].data

    return maximum(real.(a))
end

ln_ovlp1(ϕ1, ϕ1)
2*ln_ovlp1(ϕ1, ϕ2)-ln_ovlp1(ϕ2, ϕ2)