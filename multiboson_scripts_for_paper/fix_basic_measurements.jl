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

root_folder = "data_two_component_lieb_liniger"

# List all directories in root_folder
println("Directories in $root_folder:")
if isdir(root_folder)
    directories = filter(isdir, readdir(root_folder, join=true))
    for (i, dir) in enumerate(directories)
        folder_name = basename(dir)
        println("fixing basic measurements for $folder_name")
        
        open(joinpath(root_folder, folder_name, "basic_measurements.txt"), "w") do f
            println(f, "chi, energy, gnorm, n1, n2, num_iter")
            for χ in [4, 8, 16, 32]
                required_file = joinpath(root_folder, folder_name, "results_chi$(χ).jld2")
                if isfile(required_file)
                    @load required_file res
                    n1 = particle_density(res[1], 1)
                    n2 = particle_density(res[1], 2)
                    num_iter = size(res[5])[1]
                    msg = "$χ, $(res[2]), $(norm(res[3])), $n1, $n2, $num_iter"
                    println(f, msg)
                    println(msg)
                end
            end
        end
    end
    println("Total directories found: $(length(directories))")
end