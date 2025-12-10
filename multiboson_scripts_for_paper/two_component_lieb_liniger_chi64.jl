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
@info "Continue simulation for c1 = $c1, c2 = $c2, c12 = $c12, μ1 = $μ1, μ2 = $μ2"

root_folder = "data_two_component_lieb_liniger"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"
mkpath(root_folder)
mkpath(joinpath(root_folder, folder_name))

open(joinpath(root_folder, folder_name, "basic_measurements_chi64.txt"), "w") do f
    println(f, "chi, energy, gnorm, n1, n2, num_iter")
end

# before running the job, delete all the directories in root_folder except folder_name
# otherwise, different jobs might conflict with eachother when copying data to the management node
@info "Cleaning up directories in $root_folder..."
if isdir(root_folder)
    all_dirs = filter(isdir, readdir(root_folder, join=true))
    for dir_path in all_dirs
        dir_name = basename(dir_path)
        if dir_name != folder_name
            rm(dir_path, recursive=true)
        end
    end
    all_dirs = filter(isdir, readdir(root_folder, join=true))
    @info "Cleanup completed. Now the directories in $root_folder are: $(all_dirs)"
end

@load joinpath(root_folder, folder_name, "results_chi32.jld2") res
ψ5 = left_canonical(expand(res[1]; perturb=0.05));
ϕ5 = MultiBosonCMPSData_diag(ψ5)
res_d5 = ground_state(Hm, ϕ5; gradtol=1e-6, maxiter=2500, do_preconditioning=false);
ψ5 = left_canonical(MultiBosonCMPSData_MDMinv(res_d5[1]))
res5 = ground_state(Hm, ψ5; gradtol=1e-6, maxiter=2500, preconditioner_type=3);

@save joinpath(root_folder, folder_name, "results_chi64.jld2") res=res5
open(joinpath(root_folder, folder_name, "basic_measurements_chi64.txt"), "a") do f
    n1 = particle_density(res5[1], 1)
    n2 = particle_density(res5[1], 2)
    num_iter = size(res5[5])[1]
    msg = "64, $(res5[2]), $(norm(res5[3])), $n1, $n2, $num_iter"
    println(f, msg)
end

