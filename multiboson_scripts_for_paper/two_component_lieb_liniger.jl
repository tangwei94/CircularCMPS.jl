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
c = 1.0
#μ, c12 = 0.0, -0.7
μ = parse(Float64, ARGS[1])
c12 = parse(Float64, ARGS[2])

root_folder = "data_two_component_lieb_liniger"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"
mkpath(root_folder)
mkpath(joinpath(root_folder, folder_name))

c1, c2 = c, c
μ1, μ2 = μ, μ 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf);
@info "Doing calculation for c1 = $c1, c2 = $c2, c12 = $c12, μ1 = $μ1, μ2 = $μ2 "

ϕ1 = MultiBosonCMPSData_diag(rand, 4, 2);
res_d1 = ground_state(Hm, ϕ1; gradtol=1e-3, maxiter=1000, do_preconditioning=false);
ψ1 = left_canonical(MultiBosonCMPSData_MDMinv(res_d1[1]))
res1 = ground_state(Hm, ψ1; gradtol=1e-6, maxiter=2500, preconditioner_type=3);
@save joinpath(root_folder, folder_name, "results_chi4.jld2") res=res1

ψ2 = left_canonical(expand(res1[1]; perturb=0.05));
ϕ2 = MultiBosonCMPSData_diag(ψ2)
res_d2 = ground_state(Hm, ϕ2; gradtol=1e-3, maxiter=1000, do_preconditioning=false);
ψ2 = left_canonical(MultiBosonCMPSData_MDMinv(res_d2[1]))
res2 = ground_state(Hm, ψ2; gradtol=1e-6, maxiter=2500, preconditioner_type=3);
@save joinpath(root_folder, folder_name, "results_chi8.jld2") res=res2

ψ3 = left_canonical(expand(res2[1]; perturb=0.05));
ϕ3 = MultiBosonCMPSData_diag(ψ3)
res_d3 = ground_state(Hm, ϕ3; gradtol=1e-3, maxiter=1000, do_preconditioning=false);
ψ3 = left_canonical(MultiBosonCMPSData_MDMinv(res_d3[1]))
res3 = ground_state(Hm, ψ3; gradtol=1e-6, maxiter=2500, preconditioner_type=3);
@save joinpath(root_folder, folder_name, "results_chi16.jld2") res=res3

ψ4 = left_canonical(expand(res3[1]; perturb=0.075));
ϕ4 = MultiBosonCMPSData_diag(ψ4)
res_d4 = ground_state(Hm, ϕ4; gradtol=1e-3, maxiter=1000, do_preconditioning=false);
ψ4 = left_canonical(MultiBosonCMPSData_MDMinv(res_d4[1]))
res4 = ground_state(Hm, ψ4; gradtol=1e-6, maxiter=2500, preconditioner_type=3);
@save joinpath(root_folder, folder_name, "results_chi32.jld2") res=res4

# some basic measurements 
open(joinpath(root_folder, folder_name, "basic_measurements.txt"), "w") do f
    println(f, "chi, energy, gnorm, n1, n2, num_iter")
    for (res, χ) in zip([res1, res2, res3, res4], [4, 8, 16, 32])
        n1 = particle_density(res[1], 1)
        n2 = particle_density(res[1], 2)
        num_iter = size(res[5])[1]
        msg = "$χ, $(res[2]), $(norm(res[3])), $n1, $n2, $num_iter"
        println(f, msg)
        println(msg)
    end
end