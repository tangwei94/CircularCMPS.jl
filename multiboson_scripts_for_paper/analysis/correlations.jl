using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
using CairoMakie 
using JLD2 
using OptimKit 
using CSV, DataFrames
using Revise 
using CircularCMPS 

c = 1.0
μ = 2.0
for c12 in -0.9:0.1:0.9
#c = 10.0
#μ = 0.0
#for c12 in [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0]
    root_folder = "data_two_component_lieb_liniger"
    folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"
    output_folder_name = "analysis_results_c$(c)_mu$(μ)_coupling$(c12)"
    mkpath(joinpath(root_folder, output_folder_name))
    
    c1, c2 = c, c
    μ1, μ2 = μ, μ 
    
    @load joinpath(root_folder, folder_name, "results_chi32.jld2") res
    ψ = CMPSData(res[1]);
    corr, λs = correlator(ψ, Inf);
    ξ = 1/abs(λs[end-1])
    
    On1 = particle_density(ψ, 1);
    On2 = particle_density(ψ, 2);
    
    ρp = real(measure_local_observable(ψ, On1 + On2, Inf))
    #pairing_term = norm(measure_local_observable(ψ, pairing12(ψ, false), Inf))
    #hopping_term = norm(measure_local_observable(ψ, hopping12(ψ, false), Inf))
    #field1_term = norm(measure_local_observable(ψ, field_operator(ψ, 1, false), Inf))
    #field2_term = norm(measure_local_observable(ψ, field_operator(ψ, 2, false), Inf))
    
    Δxs = (0.02:0.02:1) .* 2*ξ;
    
    open(joinpath(root_folder, output_folder_name, "correlations.txt"), "w") do f
        println(f, "Δx, ξ, corr_rhom, corr_rhop, coherence_plus, coherence_minus, coherence_1, coherence_2")
        for Δx in Δxs 
            corr_ρm = real(corr(On1 - On2, On1 - On2, Δx))
            corr_ρp = real(corr(On1 + On2, On1 + On2, Δx)) - ρp^2
            coherence_plus = real(corr(pairing12(ψ, false), pairing12(ψ, true), Δx)) #- pairing_term^2
            coherence_minus = real(corr(hopping12(ψ, false), hopping12(ψ, true), Δx)) #- hopping_term^2
        
            coherence_1 = real(corr(field_operator(ψ, 1, false), field_operator(ψ, 1, true), Δx)) #- field1_term^2
            coherence_2 = real(corr(field_operator(ψ, 2, false), field_operator(ψ, 2, true), Δx)) #- field2_term^2
            
            msg = "$Δx, $ξ, $corr_ρm, $corr_ρp, $coherence_plus, $coherence_minus, $coherence_1, $coherence_2"
            msg = println(f, msg)
            println(c12, " ", Δx / ξ)
        end
    end
end 