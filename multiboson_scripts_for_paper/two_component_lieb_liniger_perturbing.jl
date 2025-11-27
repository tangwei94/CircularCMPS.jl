# for the computation of Luttinger parameters and velocities

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
μ, c12 = 1.0, -0.4
μ = parse(Float64, ARGS[1])
c12 = parse(Float64, ARGS[2])

δμ = 0.001

c1, c2 = c, c
μ1, μ2 = μ, μ 

@info "Perturbing the state for c1 = $c1, c2 = $c2, c12 = $c12, μ1 = $μ1, μ2 = $μ2 "

root_folder = "data_two_component_lieb_liniger"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"
mkpath(root_folder)

label_meaning = Dict('N' => -4.0, 'n' => -3.0, 'M' => -2.0, 'm' => -1.0, '0' => 0.0, 'p' => 1.0, 'P' => 2.0, 'q' => 3.0, 'Q' => 4.0)

# before running the job, delete all the directories in root_folder except folder_name
# otherwise, different jobs might conflict with eachother when copying data to the management node
# TODO. read the datafile first, and then delete everything
labels = ["NN", "nn", "MM", "mm", "pp", "PP", "qq", "QQ", "mp", "MP", "nq", "NQ"]
files_to_keep = ["perturbing_$(perturbing_label)_1e-3_results_c$(c)_mu$(μ)_coupling$(c12)" for perturbing_label in labels]
@info "Cleaning up directories in $root_folder..."
mkpath("tmp_init")
# copy jld2 files in folder_name to tmp_init 
for file in readdir(joinpath(root_folder, folder_name))
    if endswith(file, ".jld2")
        cp(joinpath(root_folder, folder_name, file), joinpath("tmp_init", file))
    end
end
if isdir(root_folder)
    all_dirs = filter(isdir, readdir(root_folder, join=true))
    for dir_path in all_dirs
        dir_name = basename(dir_path)
        if !(dir_name in files_to_keep)
            rm(dir_path, recursive=true)
        end
    end
    all_dirs = filter(isdir, readdir(root_folder, join=true))
    @info "Cleanup completed. Now the directories in $root_folder are: $(all_dirs)"
end

for perturbing_label in labels
    s1 = label_meaning[perturbing_label[1]]
    s2 = label_meaning[perturbing_label[2]]
    μ1_perturbing = μ1 + s1 * δμ
    μ2_perturbing = μ2 + s2 * δμ
    Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1_perturbing, μ2_perturbing], Inf);
    
    folder_name_perturbing = "perturbing_$(perturbing_label)_1e-3_results_c$(c)_mu$(μ)_coupling$(c12)"
    mkpath(joinpath(root_folder, folder_name_perturbing))
    open(joinpath(root_folder, folder_name_perturbing, "basic_measurements.txt"), "w") do f
        println(f, "chi, energy, gnorm, n1, n2, num_iter")
    end
    
    for (χ, file) in zip([4, 8, 16, 32], ["results_chi4.jld2", "results_chi8.jld2", "results_chi16.jld2", "results_chi32.jld2"])
        @load joinpath("tmp_init", file) res
        ψ1 = deepcopy(res[1])
        #ψ1 = left_canonical(CircularCMPS.perturb(ψ1; perturb=1e-6))
        res1 = ground_state(Hm, ψ1; gradtol=1e-6, maxiter=10000, preconditioner_type=3);
        @save joinpath(root_folder, folder_name_perturbing, file) res=res1

        open(joinpath(root_folder, folder_name_perturbing, "basic_measurements.txt"), "a") do f
            n1 = particle_density(res1[1], 1)
            n2 = particle_density(res1[1], 2)
            num_iter = size(res1[5])[1]
            msg = "$χ, $(res1[2]), $(norm(res1[3])), $n1, $n2, $num_iter"
            println(f, msg)
            println(msg)
        end
    end
end



