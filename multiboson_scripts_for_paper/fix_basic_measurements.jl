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
        dir_name = basename(dir)
        println("$i. $dir_name")
    end
    println("Total directories found: $(length(directories))")
else
    println("Root folder '$root_folder' does not exist!")
end

#Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1, μ2], Inf);
#
## some basic measurements 
#open(joinpath(root_folder, folder_name, "basic_measurements.txt"), "w") do f
#    println(f, "chi, energy, gnorm, n1, n2, num_iter")
#    for (res, χ) in zip([res1, res2, res3, res4], [4, 8, 16, 32])
#        n1 = particle_density(res[1], 1)
#        n2 = particle_density(res[1], 2)
#        num_iter = size(res[5])[1]
#        msg = "$χ, $(res[2]), $(norm(res[3])), $n1, $n2, $num_iter"
#        println(f, msg)
#        println(msg)
#    end
#end