using LinearAlgebra, TensorKit, KrylovKit
using ChainRules, Zygote 
using CairoMakie
using JLD2 
using OptimKit
using Revise
using CircularCMPS

c1, μ1 = 1., 2.
c2, μ2 = 1.5, 2.5
c12 = 0.5
χ = 32
@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ).jld2" res_lm
Es, gnorms = res_lm[5][1000:end, 1], res_lm[5][1000:end, 2]
@load "multiboson/results/lagrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-try2.jld2" res_lm
Es2, gnorms2 = res_lm[5][1000:end, 1], res_lm[5][1000:end, 2]

fig = Figure(backgroundcolor = :white, fontsize=14, resolution= (400, 600))

gf = fig[1, 1] = GridLayout()

ax1 = Axis(gf[1, 1], 
        xlabel = "steps",
        ylabel = "energy",
        )
@show minimum(Es)
#ylims!(ax1, -2.13, -2.00)
lines!(ax1, 1:length(Es), Es, label="Λ=1e2->1e5, tol=1e-2")
lines!(ax1, 1:length(Es2), Es2, label="try2")
@show fig

ax2 = Axis(gf[2, 1], 
        xlabel = "steps",
        ylabel = "gnorm",
        yscale = log10,
        )
lines!(ax2, 1:length(gnorms),  gnorms, label="Λ=1e1->1e5, tol=1e-2")
lines!(ax2, 1:length(gnorms2),  gnorms2, label="try2")
#axislegend(ax2, position=:rb)
@show fig

save("multiboson/results/langrangian_multiplier_$(c1)_$(c2)_$(c12)_$(μ1)_$(μ2)_$(χ)-try2.pdf", fig)
