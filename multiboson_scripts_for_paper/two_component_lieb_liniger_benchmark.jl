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

root_folder = "data_two_component_lieb_liniger"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)_benchmark"
mkpath(root_folder)
mkpath(joinpath(root_folder, folder_name))

c1, c2 = c, c
μ1, μ2 = μ, μ 

Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf);
@info "Doing calculation for c1 = $c1, c2 = $c2, c12 = $c12, μ1 = $μ1, μ2 = $μ2 "

ϕ1 = MultiBosonCMPSData_diag(rand, 8, 2);
res_d1 = ground_state(Hm, ϕ1; gradtol=1e-12, maxiter=100, do_preconditioning=false);
ψ1 = left_canonical(MultiBosonCMPSData_MDMinv(res_d1[1]))
res1 = ground_state(Hm, ψ1; gradtol=1e-6, maxiter=1000, preconditioner_type=1);

res1[2]

res_d2 = ground_state(Hm, ϕ1; gradtol=1e-12, maxiter=2000, do_preconditioning=false);
res_d2[2]

ψ3 = left_canonical(MultiBosonCMPSData_MDMinv(ϕ1))
res3 = ground_state(Hm, ψ3; gradtol=1e-6, maxiter=1000, preconditioner_type=3);

@save joinpath(root_folder, folder_name, "results_chi8.jld2") res1 res_d2 res3

open(joinpath(root_folder, folder_name, "optim_history1_chi8.txt"), "w") do f
    println(f, "energy, gnorm")
    for (y, g) in zip(res1[5][:, 1], res1[5][:, 2])
        println(f, "$y, $g")
    end
end

open(joinpath(root_folder, folder_name, "optim_history2_chi8.txt"), "w") do f
    println(f, "energy, gnorm")
    for (y, g) in zip(res_d2[5][:, 1], res_d2[5][:, 2])
        println(f, "$y, $g")
    end
end

open(joinpath(root_folder, folder_name, "optim_history3_chi8.txt"), "w") do f
    println(f, "energy, gnorm")
    for (y, g) in zip(res3[5][:, 1], res3[5][:, 2])
        println(f, "$y, $g")
    end
end