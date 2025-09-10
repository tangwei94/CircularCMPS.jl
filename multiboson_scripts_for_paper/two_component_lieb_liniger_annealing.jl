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
#μ, c12 = 0.02, -0.4
μ = parse(Float64, ARGS[1])
c12 = parse(Float64, ARGS[2])

c1, c2 = c, c
μ1, μ2 = μ, μ 
Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf);

root_folder = "data_two_component_lieb_liniger"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"
mkpath(root_folder)
mkpath(joinpath(root_folder, folder_name))
# prepare the initial guess in this folder
init_data_folder = "init_for_c$(c)_mu$(μ)_coupling$(c12)" 

@load joinpath(init_data_folder, "results_chi4.jld2") res
ψ1 = deepcopy(res[1])
res1 = ground_state(Hm, ψ1; gradtol=1e-6, maxiter=2500, preconditioner_type=1);
@save joinpath(root_folder, folder_name, "results_chi4.jld2") res=res1

@load joinpath(init_data_folder, "results_chi8.jld2") res
ψ2 = deepcopy(res[1])
res2 = ground_state(Hm, ψ2; gradtol=1e-6, maxiter=2500, preconditioner_type=1);
@save joinpath(root_folder, folder_name, "results_chi8.jld2") res=res2

@load joinpath(init_data_folder, "results_chi16.jld2") res
ψ3 = deepcopy(res[1])
res3 = ground_state(Hm, ψ3; gradtol=1e-6, maxiter=2500, preconditioner_type=1);
@save joinpath(root_folder, folder_name, "results_chi16.jld2") res=res3

@load joinpath(init_data_folder, "results_chi32.jld2") res
ψ4 = deepcopy(res[1])
res4 = ground_state(Hm, ψ4; gradtol=1e-6, maxiter=2500, preconditioner_type=1);
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