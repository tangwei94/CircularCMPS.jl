using LinearAlgebra, TensorKit, KrylovKit 
using CairoMakie 
using JLD2 
using Polynomials 
using Revise 
using CircularCMPS 

βs = 32:16:192
χs = 4:2:20
χs = χs[3:end]

fig = Figure(backgroundcolor = :white, fontsize=18, resolution= (600, 2000))

ax1 = Axis(fig[1, 1], 
        xlabel = L"χ",
        ylabel = L"βΔE", 
        )
for β in βs
    ΔEs = map(χs) do χ
        @load "ising/results/ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" ψ
        Es = eigvals(convert(Array, K_mat(ψ, ψ)))
        abs(Es[end-1] - Es[end])
    end
    scatterlines!(ax1, χs, β .* ΔEs, label="β=$β")
end
lines!(ax1, χs, χs .* 0 .+ sqrt(2))
axislegend(ax1, position=:rt)
@show fig

ax2 = Axis(fig[2, 1], 
        xlabel = L"\mathrm{ln}(β)",
        ylabel = L"\mathrm{ln}(ΔE_1)", 
        )
for χ in χs[2:end]
    ΔEs = map(βs) do β
        @load "ising/results/ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" ψ
        Es = eigvals(convert(Array, K_mat(ψ, ψ)))
        abs(Es[end-1] - Es[end])
    end

    fit_results = Polynomials.fit(log.(βs)[2:end-2], log.(ΔEs)[2:end-2], 1)
    @show χ, fit_results

    scatterlines!(ax2, log.(βs), log.(ΔEs), label="χ=$χ")
end
#lines!(ax2, log.(βs), βs .* 0 .+ sqrt(2))
axislegend(ax2, position=:lb)
@show fig

ax3 = Axis(fig[3, 1], 
        xlabel = L"\mathrm{ln}(β)",
        ylabel = L"\mathrm{ln}(ΔE_2)", 
        )
for χ in χs[2:end]
    ΔEs = map(βs) do β
        @load "ising/results/ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" ψ
        Es = eigvals(convert(Array, K_mat(ψ, ψ)))
        abs(Es[end-2] - Es[end])
    end

    fit_results = Polynomials.fit(log.(βs)[2:end-2], log.(ΔEs)[2:end-2], 1)
    @show χ, fit_results

    scatterlines!(ax3, log.(βs), log.(ΔEs), label="χ=$χ")
end
#lines!(ax2, log.(βs), βs .* 0 .+ sqrt(2))
axislegend(ax3, position=:lb)

ax4 = Axis(fig[4, 1], 
        xlabel = L"β",
        ylabel = L"ΔE/ΔE_1", 
        )
for χ in χs[2:end]
    ΔEs = map(βs) do β
        @load "ising/results/ising_Gamma$(Γ)_beta$(β)-chi$(χ).jld2" ψ
        Es = eigvals(convert(Array, K_mat(ψ, ψ)))
        abs(Es[end-3] - Es[end])
    end

    fit_results = Polynomials.fit(log.(βs)[2:end-3], log.(ΔEs)[2:end-3], 1)
    @show χ, fit_results

    scatterlines!(ax4, log.(βs), log.(ΔEs), label="χ=$χ")
end
#lines!(ax2, log.(βs), βs .* 0 .+ sqrt(2))
axislegend(ax4, position=:lt)
@show fig
save("ising/results/ising_plot.pdf", fig)

