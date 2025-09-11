using LinearAlgebra, TensorKit, KrylovKit 
using ChainRules, Zygote 
using CairoMakie 
using JLD2 
using OptimKit 
using Revise 
using CircularCMPS 

c = 1.0
μ, c12 = 0.9, -0.8

root_folder = "tmpdata/data_two_component_lieb_liniger"
folder_name = "results_c$(c)_mu$(μ)_coupling$(c12)"

c1, c2 = c, c
μ1, μ2 = μ, μ 

@load joinpath(root_folder, folder_name, "results_chi32.jld2") res
println(res[2], " ", norm(res[3]))
ψ = CMPSData(res[1]);
corr, λs = correlator(ψ, Inf);
ξ = 1/abs(λs[end-1])

On1 = particle_density(ψ, 1);
On2 = particle_density(ψ, 2);
ρp = real(measure_local_observable(ψ, On1 + On2, Inf))

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
    lines!(ax2, Δxs, norm.(corr_p_s))
    lines!(ax3, Δxs, norm.(coherence_s))
    lines!(ax4, Δxs, norm.(hopping_s))
    @show fig
end 