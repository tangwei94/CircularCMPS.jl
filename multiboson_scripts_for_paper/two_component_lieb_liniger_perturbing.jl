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
#μ, c12 = 1, -0.4
μ = parse(Float64, ARGS[1])
c12 = parse(Float64, ARGS[2])

δμ = 0.01

c1, c2 = c, c
μ1, μ2 = μ, μ 

@info "Perturbing the state for c1 = $c1, c2 = $c2, c12 = $c12, μ1 = $μ1, μ2 = $μ2 "

root_folder = "data_two_component_lieb_liniger"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"
mkpath(root_folder)

label_meaning = Dict('m' => -1.0, '0' => 0.0, 'p' => 1.0)

for perturbing_label in ["mm", "m0", "mp", "0p", "pp"]
    s1 = label_meaning[perturbing_label[1]]
    s2 = label_meaning[perturbing_label[2]]
    μ1_perturbing = μ1 + s1 * δμ
    μ2_perturbing = μ2 + s2 * δμ
    Hm = MultiBosonLiebLiniger([c1 c12; c12 c2], [μ1_perturbing, μ2_perturbing], Inf);
    
    folder_name_perturbing = "perturbing_$(perturbing_label)_1e-2_results_c$(c)_mu$(μ)_coupling$(c12)"
    mkpath(joinpath(root_folder, folder_name_perturbing))
    open(joinpath(root_folder, folder_name_perturbing, "basic_measurements.txt"), "w") do f
        println(f, "chi, energy, gnorm, n1, n2, num_iter")
    end
    
    for (χ, file) in zip([4, 8, 16, 32], ["results_chi4.jld2", "results_chi8.jld2", "results_chi16.jld2", "results_chi32.jld2"])
        @load joinpath(root_folder, folder_name, file) res
        ψ1 = deepcopy(res[1])
        res1 = ground_state(Hm, ψ1; gradtol=1e-6, maxiter=2500, preconditioner_type=3);
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


