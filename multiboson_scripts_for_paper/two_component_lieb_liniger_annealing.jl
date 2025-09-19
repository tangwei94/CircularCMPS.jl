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

μ_old = parse(Float64, ARGS[3])
c12_old = parse(Float64, ARGS[4])

c1, c2 = c, c
μ1, μ2 = μ, μ 
Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf);
@info "Doing calculation for c1 = $c1, c2 = $c2, c12 = $c12, μ1 = $μ1, μ2 = $μ2 "
@info "Initialized with results of c1 = $c1, c2 = $c2, c12 = $c12_old, μ1 = $μ_old, μ2 = $μ_old "

root_folder = "data_two_component_lieb_liniger"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"
folder_name_old = "results_c$(c)_mu$(μ_old)_coupling$(c12_old)"
mkpath(root_folder)
mkpath(joinpath(root_folder, folder_name))

open(joinpath(root_folder, folder_name, "basic_measurements.txt"), "w") do f
    println(f, "chi, energy, gnorm, n1, n2, num_iter")
end

# before running the job, delete all the directories in root_folder except folder_name
# otherwise, different jobs might conflict with eachother when copying data to the management node
@info "Cleaning up directories in $root_folder..."
mkpath("tmp_init")
# copy jld2 files in folder_name_old to tmp_init 
for file in readdir(joinpath(root_folder, folder_name_old))
    if endswith(file, ".jld2")
        cp(joinpath(root_folder, folder_name_old, file), joinpath("tmp_init", file))
    end
end
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

for (χ, file) in zip([4, 8, 16, 32], ["results_chi4.jld2", "results_chi8.jld2", "results_chi16.jld2", "results_chi32.jld2"])
    @load joinpath("tmp_init", file) res
    ψ1 = deepcopy(res[1])
    res1 = ground_state(Hm, ψ1; gradtol=1e-6, maxiter=2500, preconditioner_type=3);
    @save joinpath(root_folder, folder_name, file) res=res1

    open(joinpath(root_folder, folder_name, "basic_measurements.txt"), "a") do f
        n1 = particle_density(res1[1], 1)
        n2 = particle_density(res1[1], 2)
        num_iter = size(res1[5])[1]
        msg = "$χ, $(res1[2]), $(norm(res1[3])), $n1, $n2, $num_iter"
        println(f, msg)
        println(msg)
    end
end

