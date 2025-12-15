using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
using CairoMakie 
using JLD2 
using OptimKit 
using CSV, DataFrames 
using Revise 
using CircularCMPS 

χs = [4, 8, 16, 32, 64]
c = 1.0 
μ = 2.0
c12 = 0.0

root_folder = "data_two_component_lieb_liniger"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"

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

k1 = 1/(sqrt(12/1)+1)
k2 = 1/(sqrt(12/2)+1)
@show k1, k2

fit_range = 3:5
X = hcat(ones(length(fit_range)), log.(χs[fit_range]))
βs = X \ EEs[fit_range]
println("k=$(βs[2]), C=$(βs[1])")

X = hcat(ones(length(fit_range)), log.(ξs[fit_range]))
γs = X \ EEs[fit_range]
println("central charge=$(γs[2] * 6)")

fig = Figure(fontsize=18, size= (400, 400))

ax1 = Axis(fig[1, 1], 
    xlabel = "ln(χ)",
    ylabel = "S",
    )
scatter!(ax1, log.(χs), EEs)
lines!(ax1, log.(χs), log.(χs) .* βs[2] .+ βs[1])

ax2 = Axis(fig[2, 1], 
    xlabel = "log(ξ)",
    ylabel = "S",
    )
scatter!(ax2, log.(ξs), EEs)
lines!(ax2, log.(ξs), log.(ξs) .* γs[2] .+ γs[1])

@show fig