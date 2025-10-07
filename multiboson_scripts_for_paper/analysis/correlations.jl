using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
using CairoMakie 
using JLD2 
using OptimKit 
using Revise 
using CircularCMPS 

c = 1.0
μ, c12 = 0.0, -0.5
root_folder = "data_two_component_lieb_liniger"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"

c1, c2 = c, c
μ1, μ2 = μ, μ 

@load joinpath(root_folder, folder_name, "results_chi32.jld2") res
println(res[2], " ", norm(res[3]))
ψ = CMPSData(res[1]);
corr, λs = correlator(ψ, Inf);
λs
ξ = 1/abs(λs[end-1])

On1 = particle_density(ψ, 1);
On2 = particle_density(ψ, 2);
ρp = real(measure_local_observable(ψ, On1 + On2, Inf))
@show ρp * ξ

corr_m_s = Float64[];
corr_p_s = Float64[];
coherence_s = Float64[];
hopping_s = Float64[];
Δxs = (0.01:0.01:1) .* 2*ξ;
for Δx in Δxs
    Cm = real(corr(On1 - On2, On1 - On2, Δx))
    Cp = real(corr(On1 + On2, On1 + On2, Δx)) - ρp^2
    Cc = real(corr(pairing12(ψ, false), pairing12(ψ, true), Δx)) 
    Ch = real(corr(hopping12(ψ, false), hopping12(ψ, true), Δx)) 
    println(Δx, " ", Cm, " ", Cp, " ", Cc, " ", Ch)
    push!(corr_m_s, Cm)
    push!(corr_p_s, Cp)
    push!(coherence_s, Cc)
    push!(hopping_s, Ch)
end

let _ = 1 
    fig = Figure(fontsize=18, size= (600, 1200))

    ax1 = Axis(fig[1, 1], 
            xlabel = L"Δx",
            ylabel = L"C_m", 
            yscale = log10,
            xscale = log10,
            )
    ax2 = Axis(fig[2, 1], 
            xlabel = L"Δx",
            ylabel = L"C_p", 
            yscale = log10,
            xscale = log10,
            )
    ax3 = Axis(fig[3, 1], 
            xlabel = L"Δx",
            ylabel = L"C_c", 
            yscale = log10,
            xscale = log10,
            )
    ax4 = Axis(fig[4, 1], 
            xlabel = L"Δx",
            ylabel = L"C_h", 
            yscale = log10,
            xscale = log10,
            )
    lines!(ax1, Δxs, norm.(corr_m_s))
    ys = 1 ./ (Δxs .^ 2)
    lines!(ax1, Δxs, ys, color=:gray)
    lines!(ax2, Δxs, norm.(corr_p_s))
    lines!(ax2, Δxs, ys, color=:gray)
    lines!(ax3, Δxs, norm.(coherence_s) ./ (ρp^2))

    lines!(ax4, Δxs, norm.(hopping_s) ./ (ρp^2))
    @show fig
end 

χs = [4,8,16,32]
results = map(χs) do χ
    @load joinpath(root_folder, folder_name, "results_chi$(χ).jld2") res
    ψ = CMPSData(res[1]);
    E = entanglement_entropy_inf(ψ)
    corr, λs = correlator(ψ, Inf);
    ξ = 1/abs(λs[end-1])
    return E, ξ
end
Es = [result[1] for result in results]
ξs = [result[2] for result in results]

k1 = 1/(sqrt(12/1)+1)
k2 = 1/(sqrt(12/2)+1)
@show k1, k2

X = hcat(ones(2), log.(χs[end-1:end]))
βs = X \ Es[end-1:end]
println("k=$(βs[2]), C=$(βs[1])")

X = hcat(ones(2), log.(ξs[end-1:end]))
γs = X \ Es[end-1:end]
println("central charge=$(γs[2] * 6)")

fig = Figure(fontsize=18, size= (400, 400))

ax1 = Axis(fig[1, 1], 
    xlabel = "ln(χ)",
    ylabel = "S",
    )
scatter!(ax1, log.(χs), Es)
lines!(ax1, log.(χs), log.(χs) .* βs[2] .+ βs[1])
lines!(ax1, log.(χs), log.(χs) .* k2 .+ βs[1])

ax2 = Axis(fig[2, 1], 
    xlabel = "log.(ξ)",
    ylabel = "S",
    )
scatter!(ax2, log.(ξs), Es)
lines!(ax2, log.(ξs), log.(ξs) .* γs[2] .+ γs[1])
lines!(ax2, log.(ξs), log.(ξs) .* (2/6) .+ γs[1])


@show fig

