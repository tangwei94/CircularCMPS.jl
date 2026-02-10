using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
#using CairoMakie 
using JLD2 
using OptimKit 
using CSV, DataFrames 
using Revise 
using CircularCMPS 

χs = [4, 8, 16, 32, 64]
c = 1.0 
μ = 2.0
c12 = -0.3

k1 = 1/(sqrt(12/1)+1)
k2 = 1/(sqrt(12/2)+1)
@show k1, k2

# Temporary plotting disabled.
# fig = Figure(fontsize=18, size= (400, 400))
#     
# ax1 = Axis(fig[1, 1], 
#     xlabel = "ln(χ)",
#     ylabel = "S",
#     )

for c12 in [-0.6, -0.3, 0.0, 0.3, 0.6]
    root_folder = "data_two_component_lieb_liniger"
    folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"
    analysis_folder_name = "analysis_results_c$(c)_mu$(μ)_coupling$(c12)"
    mkpath(joinpath(root_folder, analysis_folder_name))
    
    EEs = map(χs) do χ
        @load joinpath(root_folder, folder_name, "results_chi$(χ).jld2") res
        ψ = CMPSData(res[1])
        EE = entanglement_entropy_inf(ψ)
    
        return EE
    end
    
    ξs = map(χs) do χ
        @load joinpath(root_folder, folder_name, "results_chi$(χ).jld2") res
        ψ = CMPSData(res[1])
        corr, λs = correlator(ψ, Inf);
        ξ = 1/abs(λs[end-1])
        return ξ
    end
    
    fit_range = 3:5
    X = hcat(ones(length(fit_range)), log.(χs[fit_range]))
    βs = X \ EEs[fit_range]
    println("k=$(βs[2]), C=$(βs[1])")
    
    X = hcat(ones(length(fit_range)), log.(ξs[fit_range]))
    γs = X \ EEs[fit_range]
    println("central charge=$(γs[2] * 6)")
    
    # scatter!(ax1, log.(ξs), EEs)
    #lines!(ax1, log.(ξs), log.(ξs) .* βs[2] .+ βs[1])

    open(joinpath(root_folder, analysis_folder_name, "entanglement.txt"), "w") do io
        println(io, "χ, EE, ξ")
        for (χ, EE, ξ) in zip(χs, EEs, ξs)
            println(io, "$χ, $EE, $ξ")
        end
    end
end
# @show fig
# save("tmp.pdf", fig)
