using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
using CairoMakie 
using JLD2 
using OptimKit 
using Revise 
using CircularCMPS 

# parameters for the model, only consider equal mass for now
c = 10.0
μ, c12 = 0.0, -7.0

root_folder = "data_two_component_lieb_liniger_benchmark"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)_benchmark"
mkpath(root_folder)
mkpath(joinpath(root_folder, folder_name))

c1, c2 = c, c
μ1, μ2 = μ, μ 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf);
@info "Doing calculation for c1 = $c1, c2 = $c2, c12 = $c12, μ1 = $μ1, μ2 = $μ2 "

ϕ1 = MultiBosonCMPSData_diag(rand, 8, 2);
res_d1 = ground_state(Hm, ϕ1; gradtol=1e-9, maxiter=100, do_preconditioning=false);
ψ1 = left_canonical(MultiBosonCMPSData_MDMinv(res_d1[1]))
res1 = ground_state(Hm, ψ1; gradtol=1e-6, maxiter=5000, preconditioner_type=3);
@save joinpath(root_folder, folder_name, "results_chi4.jld2") res=res1

ϕ2 = MultiBosonCMPSData_diag(rand, 16, 2);
res_d2 = ground_state(Hm, ϕ2; gradtol=1e-9, maxiter=400, do_preconditioning=false);
ψ2 = left_canonical(MultiBosonCMPSData_MDMinv(res_d2[1]))
res2 = ground_state(Hm, ψ2; gradtol=1e-6, maxiter=5000, preconditioner_type=3);
@save joinpath(root_folder, folder_name, "results_chi16.jld2") res=res2

res2[5]

ψ2_compare = left_canonical(expand(res1[1]; perturb=0.05));
ϕ2_compare = MultiBosonCMPSData_diag(ψ2_compare)
res_d2_compare = ground_state(Hm, ϕ2_compare; gradtol=1e-9, maxiter=200, do_preconditioning=false);
ψ2_compare = left_canonical(MultiBosonCMPSData_MDMinv(res_d2_compare[1]))
res2_compare = ground_state(Hm, ψ2_compare; gradtol=1e-6, maxiter=5000, preconditioner_type=3);
@save joinpath(root_folder, folder_name, "results_chi16_compare.jld2") res=res2_compare

ψ3_compare = left_canonical(expand(res2_compare[1]; perturb=0.05));
ϕ3_compare = MultiBosonCMPSData_diag(ψ3_compare)
res_d3_compare = ground_state(Hm, ϕ3_compare; gradtol=1e-9, maxiter=400, do_preconditioning=false);
ψ3_compare = left_canonical(MultiBosonCMPSData_MDMinv(res_d3_compare[1]))
res3_compare = ground_state(Hm, ψ3_compare; gradtol=1e-6, maxiter=2000, preconditioner_type=3);
@save joinpath(root_folder, folder_name, "results_chi16_compare3.jld2") res=res3_compare
