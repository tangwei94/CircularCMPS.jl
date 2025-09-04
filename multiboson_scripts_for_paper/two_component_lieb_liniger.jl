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

# parameters for the model, only consider equal mass
c = 1.0
μ = 0.5
c12 = -0.4

folder_name = "data_for_two_component_lieb_liniger_c$(c)_mu$(μ)_c12$(c12)"

c1, c2 = c, c
μ1, μ2 = μ, μ 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf);

ϕ1 = MultiBosonCMPSData_diag(rand, 4, 2);
res_d1 = ground_state(Hm, ϕ1; gradtol=1e-3, maxiter=1000, do_preconditioning=false);
ψ1 = left_canonical(MultiBosonCMPSData_MDMinv(res_d1[1]))
res1 = ground_state(Hm, ψ1; gradtol=1e-6, maxiter=2500, preconditioner_type=1);

ψ2 = left_canonical(expand(res1[1]; perturb=0.05));
ϕ2 = MultiBosonCMPSData_diag(ψ2)
res_d2 = ground_state(Hm, ϕ2; gradtol=1e-3, maxiter=1000, do_preconditioning=false);
ψ2 = left_canonical(MultiBosonCMPSData_MDMinv(res_d2[1]))
res2 = ground_state(Hm, ψ2; gradtol=1e-6, maxiter=2500, preconditioner_type=1);

ψ3 = left_canonical(expand(res2[1]; perturb=0.05));
ϕ3 = MultiBosonCMPSData_diag(ψ3)
res_d3 = ground_state(Hm, ϕ3; gradtol=1e-3, maxiter=1000, do_preconditioning=false);
ψ3 = left_canonical(MultiBosonCMPSData_MDMinv(res_d3[1]))
res3 = ground_state(Hm, ψ3; gradtol=1e-6, maxiter=2500, preconditioner_type=1);

ψ4 = left_canonical(expand(res3[1]; perturb=0.05));
ϕ4 = MultiBosonCMPSData_diag(ψ4)
res_d4 = ground_state(Hm, ϕ4; gradtol=1e-3, maxiter=1000, do_preconditioning=false);
ψ4 = left_canonical(MultiBosonCMPSData_MDMinv(res_d4[1]))
res4 = ground_state(Hm, ψ4; gradtol=1e-6, maxiter=2500, preconditioner_type=1);

@save joinpath(folder_name, "results_chi4.jld2") res=res1
@save joinpath(folder_name, "results_chi8.jld2") res=res2
@save joinpath(folder_name, "results_chi16.jld2") res=res3
@save joinpath(folder_name, "results_chi32.jld2") res=res4